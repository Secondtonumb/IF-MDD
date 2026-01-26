import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model, LlamaForCausalLM, LlamaTokenizer
import speechbrain as sb
from speechbrain.utils.data_utils import undo_padding
import numpy as np
from tqdm import tqdm
import wandb
from types import SimpleNamespace
import pdb
import json
from pathlib import Path
from mpd_eval_v4 import MpdStats

try:
    from peft import PeftModel, get_peft_model_state_dict, set_peft_model_state_dict
except ImportError:
    PeftModel = None

class PeftAdapterRecoverable:
    """Wrapper to save ONLY the adapter weights of a PeftModel in SpeechBrain checkpoints"""
    def __init__(self, model):
        self.model = model
    
    def state_dict(self):
        return get_peft_model_state_dict(self.model)
    
    def load_state_dict(self, state_dict):
        # set_peft_model_state_dict handles loose matching nicely usually
        set_peft_model_state_dict(self.model, state_dict)
    
    def to(self, device):
        self.model.to(device)

import logging
logger = logging.getLogger(__name__)

def phn_list_to_seq(batch):
    """
    Args:
        batch:[["sil", "aa", "x"], ["sil", "xa", "th"]]
    return
        batch ["sil aa x", "sil xa th"]
    """
    result = []
    for phn_list in batch:
        result.append(" ".join(x for x in phn_list))
    return result
    
class SSL_LLM_PPATP(sb.Brain):
    def __init__(self, *args, patience=20, **kwargs):
        super().__init__(*args, **kwargs)
        self.patience = patience
        self.no_improve_epochs = 0
        self.best_per_list = []  # List of (PER, epoch, ckpt_name)
        self.best_mpd_f1_list = []  # List of (mpd_f1, epoch, ckpt_name)
        self.best_per = float('inf')
        self.best_mpd_f1 = float('-inf')
        self.last_improved_epoch = 0
        
        # 初始化设备（必须先于依赖device的模块创建）
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 从 hparams 读取 canonical phoneme length 显示选项
        self.show_canonical_phn_length = getattr(self.hparams, "show_canonical_phn_length", False)
        # 创建LayerNorm层用于特征归一化
        # self.embed_layer_norm = nn.LayerNorm(self.modules.LLM.config.hidden_size).to(self.device)
        
        # 训练追踪
        self.best_valid_loss = float('inf')
        self.best_valid_loss_list = []  # List of (valid_loss, epoch, ckpt_name)
        self.train_stats = {"ctc_loss": [], "ce_loss": [], "total_loss": []}
        self.valid_stats = {"ctc_loss": [], "ce_loss": [], "total_loss": [], "per": []}
        
        # 创建phoneme token掩码
        self.phoneme_bias = None
        self.setup_phoneme_mask()
        
        # 加载混淆矩阵用于生成潜在发音候选
        self.confusion_matrix = None
        self.load_confusion_matrix(getattr(self.hparams, "pronunciation_confusion_matrix_path", None))

    def setup_phoneme_mask(self):
        """创建一个掩码，只允许生成音素相关的token"""
        if getattr(self, "phoneme_bias", None) is not None:
            return

        vocab_size = self.modules.LLM.get_input_embeddings().weight.shape[0]
        # 创建一个全为 -inf 的掩码
        self.phoneme_bias = torch.full(
            (vocab_size,), float('-10e9'), device=self.device
        )
        # 将音素token的位置设为0（允许生成）
        valid_tokens = list(range(44))  # 0-43 是音素相关的token（包括blank, bos, eos）
        self.phoneme_bias[valid_tokens] = 0
    
    def load_confusion_matrix(self, confusion_path=None):
        """加载混淆矩阵用于生成潜在发音候选"""
        if confusion_path is None:
            # 默认路径
            confusion_path = Path("utils/l2_arctic_conf.json")
            logger.info(f"No confusion matrix path provided. Using default: {confusion_path}")
        
        if not Path(confusion_path).exists():
            logger.warning(f"Confusion matrix not found at {confusion_path}. Using empty matrix.")
            self.confusion_matrix = {}
            return
        
        try:
            with open(confusion_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.confusion_matrix = data.get("confusion_matrix", {})
            logger.info(f"Loaded confusion matrix with {len(self.confusion_matrix)} phonemes from {confusion_path}")
        except Exception as e:
            logger.error(f"Error loading confusion matrix: {e}")
            self.confusion_matrix = {}
    
    def generate_potential_pronunciations(self, canonical_phonemes, top_k=5):
        """根据混淆矩阵为canonical音素序列生成潜在发音候选
        
        Args:
            canonical_phonemes: list of phonemes, e.g., ['hh', 'aa', 'z']
            top_k: 每个音素取top K个混淆项
        
        Returns:
            candidates: list of potential pronunciation sequences
        """
        if not self.confusion_matrix:
            # 如果没有混淆矩阵，返回原始序列
            return [" ".join(canonical_phonemes)]
        
        # 为每个音素收集潜在替换项
        phoneme_candidates = []
        for phn in canonical_phonemes:
            # 默认包含原始音素
            candidates = [phn]
            
            # 如果在混淆矩阵中找到该音素
            if phn in self.confusion_matrix:
                confusions = self.confusion_matrix[phn].get("confusions", [])
                # 取前 top_k 个混淆项
                for item in confusions[:top_k]:
                    confused_phn = item.get("phoneme")
                    if confused_phn and confused_phn not in candidates:
                        candidates.append(confused_phn)
            
            phoneme_candidates.append(candidates)
        
        # 构建潜在发音字符串
        # 格式: "phn1候选1, 候选2 | phn2候选1, 候选2 | ..."
        potential_parts = []
        for phn_list in phoneme_candidates:
            potential_parts.append(" ".join(phn_list))
        
        potential_pronunciation = " | ".join(potential_parts)
        return potential_pronunciation
        
    def _ensure_initialized(self):
        """Lazy initialization for components that need LLM to be loaded.
        This ensures inference works even without calling on_fit_start."""
        # import pdb; pdb.set_trace()
        # Initialize llm_norm if not exists
        llm_handle = self.modules.LLM

        # 2. [关键修改] 自动判断是否被 DDP 包裹
        # 只要是 DDP 对象，它就没有 get_input_embeddings 方法，必须通过 .module 访问内部真身
        import torch
        if isinstance(llm_handle, torch.nn.parallel.DistributedDataParallel):
            llm_handle = llm_handle.module
        
        # 3. 现在 llm_handle 肯定是原始的 CausalLM class 了，可以放心调用
        embed_fn = llm_handle.get_input_embeddings()

        if not hasattr(self, "llm_norm") or self.llm_norm is None:
            # try:
            #     hidden_size = self.modules.LLM.config.hidden_size
            # except:
            #     hidden_size = self.modules.LLM.config.text_config.hidden_size
            hidden_size = self.hparams.LLM_DIM
            self.llm_norm = nn.LayerNorm(hidden_size).to(self.device)
            print(f"[Lazy Init] Created llm_norm with hidden_size={hidden_size}")
        
        # Initialize prompt_embed based on prompt_type if needed
        use_prompt = getattr(self.hparams, "use_prompt", False)
        if use_prompt and (not hasattr(self, "prompt_embed") or self.prompt_embed is None):
            prompt_type = getattr(self.hparams, "prompt_type", "soft")
            prompt_len = getattr(self.hparams, "prompt_len", 10)
            
            hidden_size = self.hparams.LLM_DIM
  
            
            if prompt_type == "soft":
                # Learnable soft prompt
                init_method = getattr(self.hparams, "prompt_init", "xavier")
                self.prompt_embed = nn.Parameter(
                    torch.zeros(prompt_len, hidden_size, device=self.device)
                )
                if init_method == "xavier":
                    nn.init.xavier_uniform_(self.prompt_embed)
                elif init_method == "normal":
                    nn.init.normal_(self.prompt_embed, mean=0.0, std=0.02)
                elif init_method == "zeros":
                    pass  # Already zeros
                
                print(f"[Lazy Init] Created soft prompt: {self.prompt_embed.shape}, init={init_method}")
            
            elif prompt_type in ["text", "discrete"]:
                tok = self.hparams.LLM_tokenizer
                PLACEHOLDER = "<<<SPEECH_EMBEDDING_HERE>>>"

                # 1. 定义对话结构，把占位符放在你想插入语音的地方
                #    注意：Llama 3 的 user content 通常紧跟在 header 之后
                chat_structure = [
                    {
                        "role": "system", 
                        "content": "You are a arabic phoneme transcriber."
                    },
                    {
                        "role": "user", 
                        # 语音在指令之前
                        "content": f"{PLACEHOLDER}\nTranscribe the preceding speech into phonemes."
                    }
                ]

                # 2. 使用官方模板渲染成字符串 (tokenize=False)
                #    add_generation_prompt=True 会自动加上 <|start_header_id|>assistant...
                full_prompt_str = tok.apply_chat_template(
                    chat_structure, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                
                # full_prompt_str 现在看起来大约是 (自动处理了 BOS 和 Header):
                # "<|begin_of_text|><|start...|>system...>...user... \n\n<<<SPEECH_EMBEDDING_HERE>>>\nTranscribe...assistant..."

                # 3. 根据占位符切割
                if PLACEHOLDER not in full_prompt_str:
                    raise ValueError("Chat template processing removed the placeholder!")
                
                prefix_str, suffix_str = full_prompt_str.split(PLACEHOLDER)
                
                # 4. 分别 Tokenize (注意不要再加 special tokens，因为模板里已经有了)
                prefix_tokens = tok(prefix_str, return_tensors="pt", add_special_tokens=False).to(self.device)
                suffix_tokens = tok(suffix_str, return_tensors="pt", add_special_tokens=False).to(self.device)
                
                prefix_ids = prefix_tokens["input_ids"].squeeze(0)
                suffix_ids = suffix_tokens["input_ids"].squeeze(0)

                # 下面的代码保持不变
                # embed_fn = self.modules.LLM.get_input_embeddings()
                with torch.no_grad():
                    self.prompt_prefix_embed = embed_fn(prefix_ids)
                    self.prompt_suffix_embed = embed_fn(suffix_ids)
                
                self.prompt_embed = torch.cat([self.prompt_prefix_embed, self.prompt_suffix_embed], dim=0)
                print(f"[Lazy Init] Generated Llama 3 Prompt via Template.")
        
        # ============ Initialize word and canonical phoneme embedding functions ============
        # These are used in compute_forward to embed batch.wrd and batch.phn_list_canonical
        # For mispronunciation detection: embed actual (GT) words and their canonical pronunciations
        if not hasattr(self, "_wrd_token_embedder") or self._wrd_token_embedder is None:
            self._wrd_token_embedder = embed_fn
            print(f"[Lazy Init] Registered word token embedder function")
        
        if not hasattr(self, "_phn_token_embedder") or self._phn_token_embedder is None:
            self._phn_token_embedder = embed_fn
            print(f"[Lazy Init] Registered phoneme token embedder function")

    def _build_dynamic_chat_embeddings(self, B, batch, device, tok, embed_fn, PAD_ID):
        """
        为split prompt构建动态的Chat embeddings，包含PP (Potential Pronunciations) context。
        
        对batch中的每个样本：
        1. 为该样本构建user content（包含word和canonical phoneme信息）
        2. 生成可能的发音候选
        3. 应用Chat template到placeholder
        4. Tokenize prefix和suffix
        5. 在batch维度上pad并embed
        
        Args:
            B: Batch size
            batch: 原始batch数据（包含wrd, phn_list_canonical等）
            device: 计算设备
            tok: LLM tokenizer
            embed_fn: Token embedding函数
            PAD_ID: Padding token ID
            
        Returns:
            dict: {
                "prefix_embed": [B, L_prefix, H] 前缀embedding
                "suffix_embed": [B, L_suffix, H] 后缀embedding
                "prefix_lens_tensor": [B] 实际前缀长度
                "suffix_lens_tensor": [B] 实际后缀长度
            }
        """
        PLACEHOLDER = "<<<SPEECH_EMBEDDING_HERE>>>"
        
        # 为batch中的每个样本构建chat结构
        prefix_ids_list = []
        suffix_ids_list = []
        prefix_lens = []
        suffix_lens = []
        
        # 日志记录：仅在第一次调用时输出一个样本prompt
        sample_prompt_logged = getattr(self, "_sample_prompt_logged", False)
        
        for b in range(B):
            # ===== 1. 为这个样本构建user content =====
            user_content = f"{PLACEHOLDER}"
            pp_context_lines = []
            
            # 获取 prompt 格式配置
            # ATP: word + canonical only
            # PPATP: word + canonical + potential pronunciation
            # STATLLM: word + canonical (interleaved potential)
            # SLAM: no context, just transcribe speech
            prompt_format = getattr(self.hparams, "prompt_format", "statllm")
            
            if prompt_format != "slam":
                # 添加word context（可选，基于hparams配置）
                include_wrd = getattr(self.hparams, "include_wrd_in_prompt", True)
                if include_wrd and hasattr(batch, "wrd") and batch.wrd is not None and len(batch.wrd) > b:
                    if isinstance(batch.wrd[b], list):
                        words_str = " ".join(batch.wrd[b])
                    else:
                        words_str = str(batch.wrd[b])
                    pp_context_lines.append(f"Word: {words_str}")
                
                # 根据 prompt_format 选择格式
                if hasattr(batch, "phn_list_canonical") and batch.phn_list_canonical is not None and len(batch.phn_list_canonical) > b:
                    can_phn_list = batch.phn_list_canonical[b]
                    
                    if prompt_format == "statllm":
                        # ===== STATLLM 格式：穿插的 mispronunciation hints =====
                        if isinstance(can_phn_list, list) and len(can_phn_list) > 0:
                            interleaved_phonemes = []
                            
                            # 检查是否使用Err_pos_aware模式
                            statllm_mode = getattr(self.hparams, "statllm_mode", "standard")  # "standard" or "err_pos_aware"
                            
                            # 获取perceived phonemes用于比较
                            perc_phn_list = None
                            if statllm_mode == "err_pos_aware" and hasattr(batch, "phn_list_perceived") and batch.phn_list_perceived is not None and len(batch.phn_list_perceived) > b:
                                perc_phn_list = batch.phn_list_perceived[b]
                                if not isinstance(perc_phn_list, list):
                                    perc_phn_list = [perc_phn_list]
                            
                            for idx, phn in enumerate(can_phn_list):
                                phoneme_str = phn
                                alternatives = []
                                
                                # 判断是否需要添加alternatives
                                should_add_alternatives = True
                                
                                if statllm_mode == "err_pos_aware" and perc_phn_list is not None:
                                    # 只在perceived与canonical不同的位置添加alternatives
                                    if idx < len(perc_phn_list) and perc_phn_list[idx] == phn:
                                        # 该位置相同，不添加alternatives
                                        should_add_alternatives = False
                                
                                if should_add_alternatives:
                                    if self.confusion_matrix and phn in self.confusion_matrix:
                                        confusions = self.confusion_matrix[phn].get("confusions", [])
                                        for item in confusions[:3]:
                                            confused_phn = item.get("phoneme")
                                            if confused_phn and confused_phn != phn:
                                                alternatives.append(confused_phn)
                                
                                if alternatives:
                                    phoneme_str += f" ({', '.join(alternatives)})"
                                interleaved_phonemes.append(phoneme_str)
                            
                            canonical_str = " ".join(interleaved_phonemes)
                            pp_context_lines.append(f"Canonical sequence: {canonical_str}")
                            if self.show_canonical_phn_length:
                                pp_context_lines.append(f"Length: {len(can_phn_list)} phonemes")
                            # import pdb; pdb.set_trace()
                        else:
                            can_phn_str = " ".join(can_phn_list) if isinstance(can_phn_list, list) else str(can_phn_list)
                            pp_context_lines.append(f"Canonical sequence: {can_phn_str}")
                    
                    elif prompt_format == "ppatp":
                        # ===== PPATP 格式：word + canonical + potential pronunciation =====
                        can_phn_str = " ".join(can_phn_list) if isinstance(can_phn_list, list) else str(can_phn_list)
                        can_phn_len = len(can_phn_list) if isinstance(can_phn_list, list) else 1
                        
                        if self.show_canonical_phn_length:
                            pp_context_lines.append(f"Canonical sequence (length={can_phn_len}): {can_phn_str}")
                        else:
                            pp_context_lines.append(f"Canonical sequence: {can_phn_str}")
                        
                        if isinstance(can_phn_list, list):
                            potential_pron = self.generate_potential_pronunciations(can_phn_list, top_k=5)
                        else:
                            potential_pron = can_phn_str
                        pp_context_lines.append(f"Potential pronunciation: {potential_pron}")
                    
                    elif prompt_format == "atp":
                        # ===== ATP 格式：word + canonical only (no potential) =====
                        can_phn_str = " ".join(can_phn_list) if isinstance(can_phn_list, list) else str(can_phn_list)
                        can_phn_len = len(can_phn_list) if isinstance(can_phn_list, list) else 1
                        
                        if self.show_canonical_phn_length:
                            pp_context_lines.append(f"Canonical sequence (length={can_phn_len}): {can_phn_str}")
                        else:
                            pp_context_lines.append(f"Canonical sequence: {can_phn_str}")
            
            # 构建完整的user content
            if prompt_format == "slam":
                # ===== SLAM 格式：无额外上下文 =====
                user_content += "\nTranscribe the preceding speech into phonemes."
                user_content += "\nReturn in CMUdict format with spaces between phonemes."
            elif pp_context_lines:
                if prompt_format == "statllm":
                    user_content += "\nHere is what the speaker read and the canonical phoneme sequence with potential mispronunciation hints:\n"
                    user_content += "\n".join(pp_context_lines)
                    user_content += "\n\nPlease predict the actual phonemes the speaker pronounced."
                    user_content += "\nReturn in CMUdict format with spaces between phonemes."
                elif prompt_format == "ppatp":
                    user_content += "\nHere I give you what the speaker read and the canonical phonemes, as well each canonical phonemes's potential pronunciations.\n"
                    user_content += "\n".join(pp_context_lines)
                    user_content += "\nYou need to predict what phoneme the speaker is actually said, rather than the canonical ones."
                    user_content += "\nReturn in CMUdict format with spaces between phonemes."
                elif prompt_format == "atp":
                    user_content += "\nHere is what the speaker read and the canonical phoneme sequence:\n"
                    user_content += "\n".join(pp_context_lines)
                    user_content += "\n\nPlease predict the actual phonemes the speaker pronounced."
                    user_content += "\nReturn in CMUdict format with spaces between phonemes."
            else:
                user_content += "\nTranscribe the preceding speech into phonemes."
            
            # 日志记录：仅在第一次（b=0）输出示例prompt
            if b == 0 and not sample_prompt_logged:
                logger.info("=" * 80)
                logger.info(f"[Prompt Format: {prompt_format.upper()}] Sample user content:")
                logger.info("-" * 80)
                logger.info(user_content.replace(PLACEHOLDER, "[SPEECH_AUDIO]"))
                logger.info("=" * 80)
                self._sample_prompt_logged = True
            
            # import pdb; pdb.set_trace()
            # ===== 2. 构建chat结构 =====
            chat_structure = [
                {"role": "system", "content": "You are a phoneme transcriber."},
                {"role": "user", "content": user_content}
            ]
            
            # ===== 3. 应用chat template =====
            full_prompt_str = tok.apply_chat_template(
                chat_structure,
                tokenize=False,
                add_generation_prompt=True
            )
            
            if PLACEHOLDER not in full_prompt_str:
                raise ValueError("Chat template processing removed the placeholder!")
            
            # ===== 4. 分割prefix和suffix =====
            prefix_str, suffix_str = full_prompt_str.split(PLACEHOLDER)
            
            # ===== 5. Tokenize prefix和suffix =====
            prefix_tokens = tok(prefix_str, return_tensors="pt", add_special_tokens=False).to(device)
            suffix_tokens = tok(suffix_str, return_tensors="pt", add_special_tokens=False).to(device)
            
            prefix_seq = prefix_tokens["input_ids"].squeeze(0)
            suffix_seq = suffix_tokens["input_ids"].squeeze(0)
            
            prefix_ids_list.append(prefix_seq)
            suffix_ids_list.append(suffix_seq)
            prefix_lens.append(len(prefix_seq))
            suffix_lens.append(len(suffix_seq))
        
        # ===== 6. Pad sequences到相同长度 =====
        from torch.nn.utils.rnn import pad_sequence
        prefix_ids = pad_sequence(prefix_ids_list, batch_first=True, padding_value=PAD_ID)  # [B, L_prefix]
        suffix_ids = pad_sequence(suffix_ids_list, batch_first=True, padding_value=PAD_ID)  # [B, L_suffix]
        
        # 转换长度为tensor
        prefix_lens_tensor = torch.tensor(prefix_lens, dtype=torch.long, device=device)  # [B]
        suffix_lens_tensor = torch.tensor(suffix_lens, dtype=torch.long, device=device)  # [B]
        
        # ===== 7. Embed prefix和suffix =====
        prefix_embed = embed_fn(prefix_ids)  # [B, L_prefix, H]
        suffix_embed = embed_fn(suffix_ids)  # [B, L_suffix, H]
        
        return {
            "prefix_embed": prefix_embed,
            "suffix_embed": suffix_embed,
            "prefix_lens_tensor": prefix_lens_tensor,
            "suffix_lens_tensor": suffix_lens_tensor,
            "prefix_ids": prefix_ids,
            "suffix_ids": suffix_ids,
        }

    def _build_input_embeddings(self, B, Z, phn_embed, SEP_embed, BOS_embed, EOS_embed, 
                                has_split_prompt=False, suffix_embed_enhanced=None,
                                prompt_embed_batch=None, prefix_embed_enhanced=None):
        """Build input embedding sequence for LLM forward pass.
        
        Consolidates all embedding concatenation logic:
        1. Split prompt: [prefix_enhanced] [speech] [suffix_enhanced] [phonemes] [EOS]
        2. With prompt: [prompt] [SEP] [speech] [BOS] [phonemes] [EOS]
        3. Without prompt: [SEP] [speech] [BOS] [phonemes] [EOS]
        
        Args:
            B: Batch size
            Z: Speech embeddings [B, Ts, H]
            phn_embed: Phoneme embeddings [B, L_phn, H]
            SEP_embed: SEP token embedding [B, 1, H]
            BOS_embed: BOS token embedding [B, 1, H]
            EOS_embed: EOS token embedding [B, 1, H]
            has_split_prompt: Whether using split prompt (prefix/suffix)
            suffix_embed_enhanced: Enhanced suffix embeddings with PP context [B, L_suffix, H] or None
            prompt_embed_batch: Prompt embeddings [B, P, H] or None
            prefix_embed_enhanced: Enhanced prefix embeddings with PP context [B, L_prefix, H] or None
            
        Returns:
            inputs_embeds: Concatenated input embeddings [B, seq_len, H]
        """
        if has_split_prompt:
            # [prefix_enhanced] [speech] [suffix_enhanced] [phonemes] [EOS]
            if prefix_embed_enhanced is not None:
                prefix_embed = prefix_embed_enhanced
            else:
                prefix_embed = self.prompt_prefix_embed.unsqueeze(0).expand(B, -1, -1)
            
            if suffix_embed_enhanced is not None:
                suffix_embed = suffix_embed_enhanced
            else:
                suffix_embed = self.prompt_suffix_embed.unsqueeze(0).expand(B, -1, -1)
            
            inputs_embeds = torch.cat([
                prefix_embed,
                Z,
                suffix_embed,
                phn_embed,
                EOS_embed
            ], dim=1)
        elif prompt_embed_batch is not None:
            # [prompt] [SEP] [speech] [BOS] [phonemes] [EOS]
            inputs_embeds = torch.cat([
                prompt_embed_batch,  # [B, P, H]
                SEP_embed,          # [B, 1, H]
                Z,                   # [B, Ts, H]
                BOS_embed,          # [B, 1, H]
                phn_embed,          # [B, L_phn, H]
                EOS_embed           # [B, 1, H]
            ], dim=1)
        else:
            # [SEP] [speech] [BOS] [phonemes] [EOS]
            inputs_embeds = torch.cat([
                SEP_embed,          # [B, 1, H]
                Z,                   # [B, Ts, H]
                BOS_embed,          # [B, 1, H]
                phn_embed,          # [B, L_phn, H]
                EOS_embed           # [B, 1, H]
            ], dim=1)
        
        return inputs_embeds

    def compute_forward(self, batch, stage):
        """Given an input batch it computes the model forward pass.
        
        Big-vocab causal LM approach using 128K LLaMA tokenizer.
        
        Returns:
            - p_ctc: [B, T, 41] - CTC logits 
            - ce_logits: [B, L, 128000] or None - LLM logits over big vocab
            - ce_targets: dict with target token IDs, or None
            - wav_lens: [B] - sequence lengths
        """
        # Ensure critical components are initialized (for inference compatibility)
        self._ensure_initialized()
        
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "augmentation"):
            wavs = self.hparams.augmentation(wavs)

        # AudioEncoder + Projector
        # import pdb; pdb.set_trace()
        # try:
        #     wav_feats = self.modules.perceived_ssl(wavs)  # [B, T, 1024]
        # except:
        #     # for whisper
        #     wav_feats = self.modules.perceived_ssl.forward_encoder(wavs) 
        
        # # pdb.set_trace()
        
        # AudioEncoder
        # import pdb; pdb.set_trace()
        try:
            # with Conformer it give extra
            Z, _ = self.hparams.audio_encoder_modules(wavs)
        except:
            # SSL projection only
            Z = self.hparams.audio_encoder_modules(wavs)
        # for whisper
        # CTC branch
        ctc_logits = self.modules.ctc_lin(Z)
        p_ctc = self.hparams.log_softmax(ctc_logits)
        # Z = self.modules.encoder_manager(wav_feats)  # [B, T, H]

        if hasattr(self.modules, "projector") and self.modules.projector is not None:
            # import pdb; pdb.set_trace()
            Z = self.modules.projector(Z) # [B, T, H]  
        # pdb.set_trace()
        B, Ts, H = Z.shape
        device = self.device
        tok = self.hparams.LLM_tokenizer
        embed_fn = self.modules.LLM.get_input_embeddings()
        
        # ===== Tokenize phoneme sequences =====
        if self.hparams.training_target == "target":
            phn_list_to_use = batch.phn_list_target
        elif self.hparams.training_target == "perceived":
            phn_list_to_use = batch.phn_list_perceived
        phn_seq = phn_list_to_seq(phn_list_to_use)
        phn_tokens = tok(phn_seq, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        # phn_tokens["input_ids"]: [B, L_phn]
        # phn_tokens["attention_mask"]: [B, L_phn]
        # tok.batch_decode(phn_tokens["input_ids"], skip_special_tokens=False)
        
        phn_ids = phn_tokens["input_ids"]
        phn_mask = phn_tokens["attention_mask"]
        L_phn = phn_ids.size(1)
        
        # ===== Prepare special tokens =====
        BOS_ID = tok.bos_token_id
        EOS_ID = tok.eos_token_id
        PAD_ID = tok.pad_token_id if tok.pad_token_id is not None else 0
        
        # SEP token (use reserved special token if not defined)
        if tok.sep_token is None or tok.sep_token_id is None:
            tok.sep_token = "<|reserved_special_token_0|>"
        SEP_ID = tok.sep_token_id
        
        def col_tokens(tok_id):
            """Create [B, 1] tensor of token ID"""
            return torch.full((B, 1), tok_id, dtype=torch.long, device=device)
        
        SEP = col_tokens(SEP_ID)  # [B, 1]
        BOS = col_tokens(BOS_ID)  # [B, 1]
        EOS = col_tokens(EOS_ID)  # [B, 1]
        
        # ===== Optional prompt tuning =====
        use_prompt = getattr(self.hparams, "use_prompt", False)
        prompt_len = 0
        
        # Check if we have prompt embeddings (either soft or text)
        has_prompt = use_prompt and hasattr(self, "prompt_embed") and self.prompt_embed is not None
        
        if has_prompt:
            prompt_embed = self.prompt_embed  # [P, H] soft/text embedding
            prompt_len = prompt_embed.size(0)
            prompt_embed_batch = prompt_embed.unsqueeze(0).expand(B, -1, -1)  # [B, P, H]
        else:
            prompt_embed = None
            prompt_embed_batch = None
        
        # ===== Embed phoneme tokens and special tokens =====
        phn_embed = embed_fn(phn_ids)  # [B, L_phn, H]
        SEP_embed = embed_fn(SEP)  # [B, 1, H]
        BOS_embed = embed_fn(BOS)  # [B, 1, H]
        EOS_embed = embed_fn(EOS)  # [B, 1, H]
        
        # ===== Build dynamic chat prompt with PP context (for split prompt) =====
        has_split_prompt = use_prompt and hasattr(self, "prompt_prefix_embed") and hasattr(self, "prompt_suffix_embed")
        suffix_embed_enhanced = None
        prefix_embed = None
        actual_prefix_lens = None
        actual_suffix_lens = None
        
        # Initialize as None
        self.prefix_ids = None
        self.suffix_ids = None
        
        if has_split_prompt:
            # 调用封装的函数来构建动态Chat embeddings
            result = self._build_dynamic_chat_embeddings(
                B=B, 
                batch=batch, 
                device=device, 
                tok=self.hparams.LLM_tokenizer,
                embed_fn=embed_fn,
                PAD_ID=PAD_ID
            )
            prefix_embed = result["prefix_embed"]
            suffix_embed_enhanced = result["suffix_embed"]
            actual_prefix_lens = result["prefix_lens_tensor"]
            actual_suffix_lens = result["suffix_lens_tensor"]
            self.prefix_ids = result["prefix_ids"]
            self.suffix_ids = result["suffix_ids"]
        
        # ===== Build input embedding sequence =====
        inputs_embeds = self._build_input_embeddings(
            B, Z, phn_embed, SEP_embed, BOS_embed, EOS_embed,
            has_split_prompt=has_split_prompt, 
            suffix_embed_enhanced=suffix_embed_enhanced,
            prompt_embed_batch=prompt_embed_batch,
            prefix_embed_enhanced=prefix_embed if has_split_prompt else None
        )
        
        # Store prefix/suffix lengths for attention mask and label calculation
        if has_split_prompt:
            actual_prefix_lens = actual_prefix_lens  # [B]
            actual_suffix_lens = actual_suffix_lens  # [B]
        
        # Persist LayerNorm (created in on_fit_start, not per-forward)
        # if hasattr(self, "llm_norm") and self.llm_norm is not None:
        #     inputs_embeds = self.llm_norm(inputs_embeds)
        
        # Align dtype with LLM weights
        llm_dtype = embed_fn.weight.dtype
        if inputs_embeds.dtype != llm_dtype:
            inputs_embeds = inputs_embeds.to(llm_dtype)
        
        # ===== Build attention mask =====
        seq_len = inputs_embeds.size(1)
        
        if stage == sb.Stage.TRAIN:
            # Build attention mask: 1 for valid positions, 0 for padding
            # Note: LLM internally applies causal mask, so we only mark valid/invalid positions
            attention_mask = torch.ones(B, seq_len, dtype=torch.long, device=device)
            
            if has_split_prompt:
                # For split prompt: [prefix_padded] [speech] [suffix_padded] [phn] [EOS]
                # Need to mask padding in prefix, suffix, and phonemes
                L_prefix_padded = self.prefix_ids.size(1)
                L_suffix_padded = self.suffix_ids.size(1)
                
                for b in range(B):
                    # Mask prefix padding
                    actual_prefix_len = actual_prefix_lens[b].item()
                    if actual_prefix_len < L_prefix_padded:
                        attention_mask[b, actual_prefix_len:L_prefix_padded] = 0
                    
                    # Speech is always valid (no masking needed)
                    
                    # Mask suffix padding
                    suffix_start = L_prefix_padded + Ts
                    actual_suffix_len = actual_suffix_lens[b].item()
                    if actual_suffix_len < L_suffix_padded:
                        attention_mask[b, suffix_start + actual_suffix_len:suffix_start + L_suffix_padded] = 0
                    
                    # Mask phoneme padding
                    text_start = suffix_start + L_suffix_padded
                    num_phn = int(phn_mask[b].sum().item())
                    if num_phn < L_phn:
                        attention_mask[b, text_start + num_phn + 1:] = 0
            else:
                # Mask out padding in phoneme sequence
                # Sequence: [prompt(P)] [SEP(1)] [speech(Ts)] [BOS(1)] [phn(L_phn)] [EOS(1)]
                sep_pos = prompt_len
                speech_start = sep_pos + 1
                speech_end = speech_start + Ts
                bos_pos = speech_end
                text_start = bos_pos + 1
                
                for b in range(B):
                    num_phn = int(phn_mask[b].sum().item())
                    # Mask positions after actual phonemes + EOS
                    if num_phn < L_phn:
                        # text_start + num_phn is EOS position
                        # Everything after EOS should be masked (padding)
                        attention_mask[b, text_start + num_phn + 1:] = 0
                        # Ignore index for masked positions
            ignore_idx = -100
            labels = torch.full((B, seq_len), ignore_idx, dtype=torch.long, device=device)
            if has_split_prompt:
                # For split prompt, suffix contains BOS at the end (before phonemes)
                # Sequence: [prefix_padded] [speech] [suffix_padded] [phn] [EOS]
                L_prefix_padded = self.prefix_ids.size(1)
                L_suffix_padded = self.suffix_ids.size(1)
                suffix_start = L_prefix_padded + Ts
                text_start = suffix_start + L_suffix_padded
                bos_pos = text_start - 1  # Last position of suffix is BOS
            else:
                # Sequence: [prompt(P)] [SEP(1)] [speech(Ts)] [BOS(1)] [phn(L_phn)] [EOS(1)]
                sep_pos = prompt_len  # SEP位置
                speech_start = sep_pos + 1
                speech_end = speech_start + Ts
                bos_pos = speech_end  # BOS位置
                text_start = bos_pos + 1  # phoneme开始位置
            
            text_end = text_start + L_phn
            eos_pos = text_end  # EOS位置

            # Set phoneme labels (left-shifted for causal LM)
            # hidden[i] predicts token[i+1]
            # pdb.set_trace()
            for b in range(B):
                num_phn = int(phn_mask[b].sum().item())  # actual phoneme length
                if num_phn > 0:
                    # BOS predicts first phoneme, phonemes predict next phoneme
                    labels[b, bos_pos : text_start + num_phn - 1] = phn_ids[b, : num_phn]
                    # Last phoneme position predicts EOS
                    labels[b, text_start + num_phn - 1] = EOS_ID
            
            labels_noshift = torch.full((B, seq_len), ignore_idx, dtype=torch.long, device=device)
            for b in range(B):
                num_phn = int(phn_mask[b].sum().item())
                if num_phn > 0:
                    labels_noshift[b, bos_pos] = BOS_ID
                    labels_noshift[b, text_start : text_start + num_phn] = phn_ids[b, :num_phn]
                    labels_noshift[b, text_start + num_phn] = EOS_ID
            
            # Run forward pass
            # Note: LlamaForCausalLM internally applies causal attention mask
            # (position i can only attend to positions 0...i-1)
            # We only need to provide padding mask via attention_mask
            # mask
            # self.modules.LLM.loss_function = torch.nn.CrossEntropyLoss(ignore_index=ignore_idx)
            llm_out = self.modules.LLM(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels_noshift,
                output_hidden_states=False,
                return_dict=True,
            )
            # llm_out.logits: [B, seq_len, 128000]
            # pdb.set_trace()
            loss = llm_out.loss  # CrossEntropyLoss if labels provided
            logger.info(f"LLM Loss: {loss.item():.4f}")
            # pdb.set_trace()
            # pdb.set_trace()
            ce_logits = llm_out.logits
            
            # Debug: verify causal masking (only run once)
            if not hasattr(self, '_causal_check_done'):
                print(f"[INFO] LLM type: {type(self.modules.LLM).__name__}")
                print(f"[INFO] LLM should use causal attention by default")
                print(f"[INFO] Logits shape: {ce_logits.shape}")
                self._causal_check_done = True

            return p_ctc, ce_logits, {"labels": labels}, wav_lens
        
        else:
            # ===== Validation/Test Stage =====
            if stage == sb.Stage.VALID:
                # Teacher-forcing (use full target): same as training
                # Build attention mask: 1 for valid positions, 0 for padding
                # Note: LLM internally applies causal mask, so we only mark valid/invalid positions
                
                attention_mask = torch.ones(B, seq_len, dtype=torch.long, device=device)
                
                if has_split_prompt:
                    # For split prompt with variable-length prefix/suffix
                    L_prefix_padded = self.prefix_ids.size(1)
                    L_suffix_padded = self.suffix_ids.size(1)
                    
                    for b in range(B):
                        # Mask prefix padding
                        actual_prefix_len = actual_prefix_lens[b].item()
                        if actual_prefix_len < L_prefix_padded:
                            attention_mask[b, actual_prefix_len:L_prefix_padded] = 0
                        
                        # Mask suffix padding
                        suffix_start = L_prefix_padded + Ts
                        actual_suffix_len = actual_suffix_lens[b].item()
                        if actual_suffix_len < L_suffix_padded:
                            attention_mask[b, suffix_start + actual_suffix_len:suffix_start + L_suffix_padded] = 0
                        
                        # Mask phoneme padding
                        text_start = suffix_start + L_suffix_padded
                        num_phn = int(phn_mask[b].sum().item())
                        if num_phn < L_phn:
                            attention_mask[b, text_start + num_phn + 1:] = 0
                else:
                    # Mask out padding in phoneme sequence
                    # Sequence: [prompt(P)] [SEP(1)] [speech(Ts)] [BOS(1)] [phn(L_phn)] [EOS(1)]
                    sep_pos = prompt_len
                    speech_start = sep_pos + 1
                    speech_end = speech_start + Ts
                    bos_pos = speech_end
                    text_start = bos_pos + 1
                    
                    for b in range(B):
                        num_phn = int(phn_mask[b].sum().item())
                        # Mask positions after actual phonemes + EOS
                        if num_phn < L_phn:
                            # text_start + num_phn is EOS position
                            # Everything after EOS should be masked (padding)
                            attention_mask[b, text_start + num_phn + 1:] = 0
                
                llm_out = self.modules.LLM(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    output_hidden_states=False,
                    return_dict=True,
                )
                ce_logits = llm_out.logits
                
                # Labels same as training
                ignore_idx = -100
                labels = torch.full((B, seq_len), ignore_idx, dtype=torch.long, device=device)
                
                # Calculate positions (same as training)
                if has_split_prompt:
                    L_prefix_padded = self.prefix_ids.size(1)
                    L_suffix_padded = self.suffix_ids.size(1)
                    suffix_start = L_prefix_padded + Ts
                    text_start = suffix_start + L_suffix_padded
                    bos_pos = text_start - 1
                else:
                    sep_pos = prompt_len
                    speech_start = sep_pos + 1
                    speech_end = speech_start + Ts
                    bos_pos = speech_end
                    text_start = bos_pos + 1
                
                for b in range(B):
                    num_phn = int(phn_mask[b].sum().item())
                    if num_phn > 0:
                        labels[b, bos_pos : text_start + num_phn - 1] = phn_ids[b, : num_phn]
                        labels[b, text_start + num_phn - 1] = EOS_ID
                
                return p_ctc, ce_logits, {"labels": labels}, wav_lens
            
            elif stage == sb.Stage.TEST:
                # ===== Autoregressive generation (no teacher forcing) =====
                # Only provide: [prompt?] [SEP] [speech] [BOS]
                # Model generates: [phn[0]] [phn[1]] ... [EOS]
                
                if has_split_prompt:
                    # Use the enhanced prefix and suffix embeddings from training
                    # These already have sample-specific PP context
                    inputs_embeds_inference = torch.cat([
                        prefix_embed,  # [B, L_prefix, H] (already padded and embedded)
                        Z,             # [B, Ts, H]
                        suffix_embed_enhanced   # [B, L_suffix, H] (already padded and embedded)
                    ], dim=1)
                    
                    # Build attention mask for inference (mask paddings)
                    attention_mask_inference = torch.ones(B, inputs_embeds_inference.size(1), dtype=torch.long, device=device)
                    L_prefix_padded = self.prefix_ids.size(1)
                    L_suffix_padded = self.suffix_ids.size(1)
                    
                    for b in range(B):
                        # Mask prefix padding
                        actual_prefix_len = actual_prefix_lens[b].item()
                        if actual_prefix_len < L_prefix_padded:
                            attention_mask_inference[b, actual_prefix_len:L_prefix_padded] = 0
                        
                        # Mask suffix padding
                        suffix_start = L_prefix_padded + Ts
                        actual_suffix_len = actual_suffix_lens[b].item()
                        if actual_suffix_len < L_suffix_padded:
                            attention_mask_inference[b, suffix_start + actual_suffix_len:suffix_start + L_suffix_padded] = 0
                
                elif prompt_embed_batch is not None:
                    inputs_embeds_inference = torch.cat([
                        prompt_embed_batch,  # [B, P, H]
                        SEP_embed,          # [B, 1, H]
                        Z,                   # [B, Ts, H]
                        BOS_embed           # [B, 1, H]
                    ], dim=1)  # [B, P+1+Ts+1, H]
                    attention_mask_inference = torch.ones(B, inputs_embeds_inference.size(1), dtype=torch.long, device=device)
                else:
                    inputs_embeds_inference = torch.cat([
                        SEP_embed,          # [B, 1, H]
                        Z,                   # [B, Ts, H]
                        BOS_embed           # [B, 1, H]
                    ], dim=1)  # [B, 1+Ts+1, H]
                    attention_mask_inference = torch.ones(B, inputs_embeds_inference.size(1), dtype=torch.long, device=device)
                
                # Align dtype with LLM weights
                if inputs_embeds_inference.dtype != llm_dtype:
                    inputs_embeds_inference = inputs_embeds_inference.to(llm_dtype)
                
                # print(f"[DEBUG] Generate input shape: {inputs_embeds_inference.shape}")
                # print(f"[DEBUG] max_new_tokens: {L_phn + 10}, BOS_ID: {BOS_ID}, EOS_ID: {EOS_ID}, PAD_ID: {PAD_ID}")
                # print(f"[DEBUG] Using full 128K vocab (no constraint)")
                
                # Generate phoneme tokens with full vocabulary
                # import pdb; pdb.set_trace()

                gen_out = self.modules.LLM.generate(
                    inputs_embeds=inputs_embeds_inference,
                    attention_mask=attention_mask_inference,
                    num_return_sequences=1,
                    pad_token_id=PAD_ID,
                    eos_token_id=EOS_ID,
                    bos_token_id=BOS_ID,
                    do_sample=True,
                    use_cache=True,
                    num_beams=1,
                    # top_k = 71,
                    # top_p = 0.9,
                    max_new_tokens=L_phn + 10, 
                    temperature=1.0,
                    # repetition_penalty=1.00,
                    # no_repeat_ngram_size=4,
                )
                # from matplotlib import pyplot as plt
                # plt.imshow(inputs_embeds_inference[0].cpu().numpy(), aspect='auto')
                # plt.savefig('input_embed.png')
                # plt.imshow(attention_mask_inference.cpu().numpy(), aspect='auto')
                # plt.savefig('attention_mask.png')
                # max_new_tokens=L_phn + 10,  # 增加生成长度

                # gen_out: [B, generated_len] 
                # 注意：当使用 inputs_embeds 时，generate() 只返回新生成的 tokens，不包含 prompt
                # print(f"[DEBUG] Generate output shape: {gen_out.shape}")
                # print(f"[DEBUG] First sample tokens: {gen_out[0].tolist()}")
                
                # 使用 inputs_embeds 时，generate 已经只返回新生成的部分
                gen_tokens = gen_out  # [B, generated_len]
                # pdb.set_trace()
                
                # print(f"[DEBUG] Generated tokens shape: {gen_tokens.shape}")
                # print(f"[DEBUG] Generated tokens[0]: {gen_tokens[0].tolist()}")
                # if gen_tokens.size(1) > 0:
                #     print(f"[DEBUG] Decoded: {self.hparams.LLM_tokenizer.decode(gen_tokens[0], skip_special_tokens=False)}")
                # else:
                #     print(f"[DEBUG] Decoded: <empty>")
                
                # Return for metrics calculation
                return p_ctc, None, {"generated_ids": gen_tokens, "target_phonemes": phn_list_to_use}, wav_lens

    def compute_objectives(self, predictions, batch, stage):
        """Compute training objectives: CTC loss + LLM loss (big-vocab, 128K tokens)"""
        ids = batch.id
        wavs, wav_lens = batch.sig
        targets, target_lens = batch.phn_encoded_target  # CTC targets (phoneme indices)
        
        # Unpack predictions from compute_forward
        if len(predictions) == 4:
            p_ctc, ce_logits, ce_targets, lens_for_ctc = predictions
        else:
            raise ValueError(f"Expected 4 values from compute_forward, got {len(predictions)}")
        
        # ===== CTC Loss =====
        T = p_ctc.size(1)
        clipped_target_lens = torch.minimum(target_lens, torch.full_like(target_lens, T))
        # import pdb; pdb.set_trace()
        loss_ctc = self.hparams.ctc_cost(p_ctc.float(), targets, lens_for_ctc, clipped_target_lens)
        
        # ===== LLM Loss (big-vocab CrossEntropyLoss) =====
        loss_ce = torch.tensor(0.0, device=self.device)
        
         # --- 1. Compute CE Loss (TRAIN & VALID) ---
        if stage != sb.Stage.TEST:
            if ce_logits is not None and isinstance(ce_targets, dict) and "labels" in ce_targets:
                labels = ce_targets["labels"]  # [B, seq_len]
                if labels is not None:
                    B, seq_len, vocab_size = ce_logits.shape
                    ignore_idx = -100
                    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_idx)
                    loss_ce = ce_loss_fn(
                        ce_logits.reshape(-1, vocab_size),
                        labels.reshape(-1)
                    )

        # --- 2. Compute Metrics (VALID & TEST) ---
        if stage != sb.Stage.TRAIN:
            # 2.1 CTC Metrics (Common)
            ctc_sequence = sb.decoders.ctc_greedy_decode(
                p_ctc, wav_lens, blank_id=self.hparams.blank_index
            )
            self.ctc_metrics.append(ids, p_ctc, targets, lens_for_ctc, clipped_target_lens)
            # remove id==70 in ctc sequence
            ctc_sequence = [[phn for phn in seq if phn != 70] for seq in ctc_sequence]
            self.per_metrics.append(
                ids=ids,
                predict=ctc_sequence,
                target=targets,
                predict_len=None,
                target_len=target_lens,
                ind2lab=self.label_encoder.decode_ndim,
            )
            
            # 计算CTC的MPD F1
            try:
                canonical = batch.phn_list_canonical
                perceived = batch.phn_list_perceived
            except:
                canonical = None
                perceived = None
            
            if canonical is not None and perceived is not None:
                # 将CTC序列转换为phoneme列表
                ctc_pred_phonemes = []
                for seq in ctc_sequence:
                    phn_list = self.label_encoder.decode_ndim(seq)
                    ctc_pred_phonemes.append(phn_list)
                
                self.ctc_mpd_f1_metrics.append(
                    ids=ids,
                    predict=ctc_pred_phonemes,
                    canonical=canonical,
                    perceived=perceived,
                    predict_len=None,
                    canonical_len=None,
                    perceived_len=None,
                    ind2lab=lambda x: x
                )

            # 2.2 LLM Metrics (Stage Dependent)
            if stage == sb.Stage.VALID:
                # VALID: Teacher-Forcing Evaluation
                # Use logits to compute perplexity-like metrics and greedy decoding accuracy

                try:
                    canonical = batch.phn_list_canonical
                    perceived = batch.phn_list_perceived
                except:
                    canonical = None
                    perceived = None

                if ce_logits is not None and isinstance(ce_targets, dict) and "labels" in ce_targets:
                    labels = ce_targets["labels"]
                    
                    llm_predictions = ce_logits.argmax(dim=-1)
                    p_llm = F.log_softmax(ce_logits, dim=-1)
                    
                    valid_mask = (labels != -100)
                    B = labels.size(0)
                    for b in range(B):
                        valid_pos = valid_mask[b]
                        num_valid = valid_pos.sum().item()
                        if num_valid > 0:
                            # Log Probablity Metric
                            self.llm_metrics.append(
                                ids=[ids[b]],
                                log_probabilities=p_llm[b, valid_pos, :].unsqueeze(0),
                                targets=labels[b, valid_pos].unsqueeze(0),
                                length=torch.tensor([num_valid], device=labels.device)
                            )
                            # PER Metric (Teacher Forced)
                            pred_tokens = llm_predictions[b, valid_pos]
                            valid_labels = labels[b, valid_pos]
                            
                            pred_text = self.hparams.LLM_tokenizer.decode(pred_tokens, skip_special_tokens=True)
                            target_text = self.hparams.LLM_tokenizer.decode(valid_labels, skip_special_tokens=True)
                            
                            self.llm_per_metrics.append(
                                ids=[ids[b]],
                                predict=[pred_text.split()],
                                target=[target_text.split()],
                                predict_len=None,
                                target_len=None,
                                ind2lab=lambda x: x 
                            )
                            self.mpd_f1_metrics.append(
                                ids=[ids[b]],
                                predict=[pred_text.split()],
                                canonical=[canonical[b]],
                                perceived=[perceived[b]],
                                predict_len=None,
                                canonical_len=None,
                                perceived_len=None,
                                ind2lab=lambda x: x
                            )

 
            elif stage == sb.Stage.TEST:
                # TEST: Autoregressive Generation Evaluation
                # Use generated IDs to compute real PER
                canonical = batch.phn_list_canonical
                _, canonical_lens = batch.phn_encoded_canonical
                perceived = batch.phn_list_perceived
                _, perceived_lens = batch.phn_encoded_perceived
                
                if isinstance(ce_targets, dict) and "generated_ids" in ce_targets:
                    gen_ids = ce_targets["generated_ids"]
                    
                    # 1. Decode Hypotheses (Generated)
                    hyps = self.hparams.LLM_tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
                    
                    # 2. Decode References (Targets)
                    if self.hparams.training_target == "target":
                        phn_list_to_use = batch.phn_list_target
                    elif self.hparams.training_target == "perceived":
                        phn_list_to_use = batch.phn_list_perceived
                    refs = phn_list_to_seq(phn_list_to_use)
                    
                    # 3. Compute PER
                    self.llm_per_metrics.append(
                        ids=ids,
                        predict=[hyp.split() for hyp in hyps],
                        target=[ref.split() for ref in refs],
                        predict_len=None,
                        target_len=None,
                        ind2lab=lambda x: x
                    )
                    # 4. Compute MPD F1
                    # pdb.set_trace()
                    self.mpd_f1_metrics.append(
                        ids=ids,
                        predict=[hyp.split() for hyp in hyps],
                        canonical=canonical,
                        perceived=perceived,
                        predict_len=None,
                        canonical_len=None,
                        perceived_len=None,
                        ind2lab=lambda x: x
                    )
                    # pdb.set_trace()

        # ===== Combined Loss =====
        ctc_weight = getattr(self.hparams, "ctc_weight", 0.3)
        loss = ctc_weight * loss_ctc + (1.0 - ctc_weight) * loss_ce
        
        # ===== Logging =====
        if stage == sb.Stage.TRAIN:
            self.train_stats.setdefault("ctc_loss", []).append(float(loss_ctc.detach().cpu()))
            self.train_stats.setdefault("ce_loss", []).append(float(loss_ce.detach().cpu()))
            self.train_stats.setdefault("total_loss", []).append(float(loss.detach().cpu()))
        else:
            self.valid_stats.setdefault("ctc_loss", []).append(float(loss_ctc.detach().cpu()))
            self.valid_stats.setdefault("ce_loss", []).append(float(loss_ce.detach().cpu()))
            self.valid_stats.setdefault("total_loss", []).append(float(loss.detach().cpu()))
        
        return loss
    
    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        predictions = self.compute_forward(batch, stage=stage)
        loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()

    @torch.no_grad()
    def inference_batch(self, batch, max_new_tokens=100, do_sample=False, temperature=1.0, top_k=50, top_p=0.9, num_beams=1, inference_prompt_mode="with_alternatives"):
        """
        Pure inference when batch only contains id and sig (no target phonemes).
        
        Args:
            batch: A batch containing:
                - batch.id: list of utterance IDs
                - batch.sig: tuple of (wavs [B, T], wav_lens [B])
            max_new_tokens: Maximum number of tokens to generate
            do_sample: Whether to use sampling (True) or greedy decoding (False)
            temperature: Sampling temperature (only used if do_sample=True)
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            num_beams: Number of beams for beam search (if applicable)
            inference_prompt_mode: "with_alternatives" (add potential phonemes) or "canonical_only" (no alternatives)
            
        Returns:
            dict: {
                "ids": list of utterance IDs,
                "generated_tokens": tensor of generated token IDs [B, gen_len],
                "generated_text": list of decoded phoneme strings,
                "ctc_predictions": list of CTC decoded phoneme sequences (if available)
            }
        """
        # Ensure critical components are initialized
        self._ensure_initialized()
        
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        ids = batch.id
        
        # AudioEncoder + Projector
        wav_feats = self.modules.perceived_ssl(wavs)  # [B, T, 1024]
        
        # AudioEncoder
        try:
            Z, _ = self.hparams.audio_encoder_modules(wavs)
        except:
            Z = self.hparams.audio_encoder_modules(wavs)
        
        # CTC branch (optional, for comparison)
        ctc_predictions = None
        if hasattr(self.modules, "ctc_lin"):
            ctc_logits = self.modules.ctc_lin(Z)
            p_ctc = self.hparams.log_softmax(ctc_logits)
            ctc_sequence = sb.decoders.ctc_greedy_decode(
                p_ctc, wav_lens, blank_id=self.hparams.blank_index
            )
            # Decode CTC predictions to phoneme strings
            ctc_predictions = []
            for seq in ctc_sequence:
                # seq = [s for s in seq if s != self.hparams.blank_index]
                # seq = [s for s in seq if s != 70]
                phn_list = self.label_encoder.decode_ndim(seq)
                ctc_predictions.append(" ".join(phn_list))
        
        # Projector
        if hasattr(self.modules, "projector") and self.modules.projector is not None:
            Z = self.modules.projector(Z)
        
        B, Ts, H = Z.shape
        device = self.device
        tok = self.hparams.LLM_tokenizer
        embed_fn = self.modules.LLM.get_input_embeddings()
        
        # Get LLM dtype
        llm_dtype = embed_fn.weight.dtype
        
        # Prepare special tokens
        BOS_ID = tok.bos_token_id
        EOS_ID = tok.eos_token_id
        PAD_ID = tok.pad_token_id if tok.pad_token_id is not None else 0
        
        if tok.sep_token is None or tok.sep_token_id is None:
            tok.sep_token = "<|reserved_special_token_0|>"
        SEP_ID = tok.sep_token_id
        
        def col_tokens(tok_id):
            return torch.full((B, 1), tok_id, dtype=torch.long, device=device)
        
        SEP = col_tokens(SEP_ID)
        BOS = col_tokens(BOS_ID)
        
        # Embed special tokens
        SEP_embed = embed_fn(SEP)
        BOS_embed = embed_fn(BOS)
        
        # Check prompt type
        use_prompt = getattr(self.hparams, "use_prompt", False)
        has_split_prompt = use_prompt and hasattr(self, "prompt_prefix_embed") and hasattr(self, "prompt_suffix_embed")
        has_prompt = use_prompt and hasattr(self, "prompt_embed") and self.prompt_embed is not None
        
        # Build input embeddings for inference
        if has_split_prompt:
            # Build dynamic chat prompt with PP context for each sample (same as training)
            PLACEHOLDER = "<<<SPEECH_EMBEDDING_HERE>>>"
            
            prefix_ids_list = []
            suffix_ids_list = []
            prefix_lens = []
            suffix_lens = []
            
            for b in range(B):
                # Build user content for this sample
                user_content = f"{PLACEHOLDER}"
                pp_context_lines = []
                
                # 获取 prompt 格式配置
                prompt_format = getattr(self.hparams, "prompt_format", "statllm")
                
                if prompt_format != "slam":
                    # 添加word context（可选，基于hparams配置）
                    include_wrd = getattr(self.hparams, "include_wrd_in_prompt", True)
                    if include_wrd and hasattr(batch, "wrd") and batch.wrd is not None and len(batch.wrd) > b:
                        if isinstance(batch.wrd[b], list):
                            words_str = " ".join(batch.wrd[b])
                        else:
                            words_str = str(batch.wrd[b])
                        pp_context_lines.append(f"Word: {words_str}")
                    
                    # 根据 prompt_format 选择格式
                    if hasattr(batch, "phn_list_canonical") and batch.phn_list_canonical is not None and len(batch.phn_list_canonical) > b:
                        can_phn_list = batch.phn_list_canonical[b]
                        
                        if prompt_format == "statllm":
                            # ===== STATLLM 格式 =====
                            if isinstance(can_phn_list, list) and len(can_phn_list) > 0:
                                # 推理时控制是否添加alternatives
                                if inference_prompt_mode == "canonical_only":
                                    # 只显示canonical，不添加alternatives
                                    can_phn_str = " ".join(can_phn_list)
                                    pp_context_lines.append(f"Canonical sequence: {can_phn_str}")
                                    if self.show_canonical_phn_length:
                                        pp_context_lines.append(f"Length: {len(can_phn_list)} phonemes")
                                else:
                                    # "with_alternatives" 模式：添加alternatives
                                    interleaved_phonemes = []
                                    
                                    # 检查是否使用Err_pos_aware模式
                                    statllm_mode = getattr(self.hparams, "statllm_mode", "standard")  # "standard" or "err_pos_aware"
                                    
                                    # 获取perceived phonemes用于比较
                                    perc_phn_list = None
                                    if statllm_mode == "err_pos_aware" and hasattr(batch, "phn_list_perceived") and batch.phn_list_perceived is not None and len(batch.phn_list_perceived) > b:
                                        perc_phn_list = batch.phn_list_perceived[b]
                                        if not isinstance(perc_phn_list, list):
                                            perc_phn_list = [perc_phn_list]
                                    
                                    for idx, phn in enumerate(can_phn_list):
                                        phoneme_str = phn
                                        alternatives = []
                                        
                                        # 判断是否需要添加alternatives
                                        should_add_alternatives = True
                                        
                                        if statllm_mode == "err_pos_aware" and perc_phn_list is not None:
                                            # 只在perceived与canonical不同的位置添加alternatives
                                            if idx < len(perc_phn_list) and perc_phn_list[idx] == phn:
                                                # 该位置相同，不添加alternatives
                                                should_add_alternatives = False
                                        
                                        if should_add_alternatives:
                                            if self.confusion_matrix and phn in self.confusion_matrix:
                                                confusions = self.confusion_matrix[phn].get("confusions", [])
                                                for item in confusions[:3]:
                                                    confused_phn = item.get("phoneme")
                                                    if confused_phn and confused_phn != phn:
                                                        alternatives.append(confused_phn)
                                        
                                        if alternatives:
                                            phoneme_str += f" ({', '.join(alternatives)})"
                                        interleaved_phonemes.append(phoneme_str)
                                    
                                    canonical_str = " ".join(interleaved_phonemes)
                                    pp_context_lines.append(f"Canonical sequence: {canonical_str}")
                                    if self.show_canonical_phn_length:
                                        pp_context_lines.append(f"Length: {len(can_phn_list)} phonemes")
                            else:
                                can_phn_str = " ".join(can_phn_list) if isinstance(can_phn_list, list) else str(can_phn_list)
                                pp_context_lines.append(f"Canonical sequence: {can_phn_str}")
                        
                        elif prompt_format == "ppatp":
                            # ===== PPATP 格式 =====
                            can_phn_str = " ".join(can_phn_list) if isinstance(can_phn_list, list) else str(can_phn_list)
                            can_phn_len = len(can_phn_list) if isinstance(can_phn_list, list) else 1
                            
                            if self.show_canonical_phn_length:
                                pp_context_lines.append(f"Canonical sequence (length={can_phn_len}): {can_phn_str}")
                            else:
                                pp_context_lines.append(f"Canonical sequence: {can_phn_str}")
                            
                            if isinstance(can_phn_list, list):
                                potential_pron = self.generate_potential_pronunciations(can_phn_list, top_k=5)
                            else:
                                potential_pron = can_phn_str
                            pp_context_lines.append(f"Potential pronunciation: {potential_pron}")
                        
                        elif prompt_format == "atp":
                            # ===== ATP 格式：word + canonical only =====
                            can_phn_str = " ".join(can_phn_list) if isinstance(can_phn_list, list) else str(can_phn_list)
                            can_phn_len = len(can_phn_list) if isinstance(can_phn_list, list) else 1
                            
                            if self.show_canonical_phn_length:
                                pp_context_lines.append(f"Canonical sequence (length={can_phn_len}): {can_phn_str}")
                            else:
                                pp_context_lines.append(f"Canonical sequence: {can_phn_str}")
                
                # 构建完整的user content
                if prompt_format == "slam":
                    user_content += "\nTranscribe the preceding speech into phonemes."
                    user_content += "\nReturn in CMUdict format with spaces between phonemes."
                elif pp_context_lines:
                    if prompt_format == "statllm":
                        user_content += "\nHere is what the speaker read and the canonical phoneme sequence with potential mispronunciation hints:\n"
                        user_content += "\n".join(pp_context_lines)
                        user_content += "\n\nPlease predict the actual phonemes the speaker pronounced."
                        user_content += "\nReturn in CMUdict format with spaces between phonemes."
                    elif prompt_format == "ppatp":
                        user_content += "\nHere I give you what the speaker read and the canonical phonemes, as well each canonical phonemes's potential pronunciations.\n"
                        user_content += "\n".join(pp_context_lines)
                        user_content += "\nYou need to predict what phoneme the speaker is actually said, rather than the canonical ones."
                        user_content += "\nReturn in CMUdict format with spaces between phonemes."
                    elif prompt_format == "atp":
                        user_content += "\nHere is what the speaker read and the canonical phoneme sequence:\n"
                        user_content += "\n".join(pp_context_lines)
                        user_content += "\n\nPlease predict the actual phonemes the speaker pronounced."
                        user_content += "\nReturn in CMUdict format with spaces between phonemes."
                else:
                    user_content += "\nTranscribe the preceding speech into phonemes."
                # Construct chat structure
                chat_structure = [
                    {"role": "system", "content": "You are a arabic phoneme transcriber."},
                    {"role": "user", "content": user_content}
                ]
                
                # Apply chat template
                full_prompt_str = tok.apply_chat_template(
                    chat_structure,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                if PLACEHOLDER not in full_prompt_str:
                    raise ValueError("Chat template processing removed the placeholder!")
                
                prefix_str, suffix_str = full_prompt_str.split(PLACEHOLDER)
                
                # Tokenize
                prefix_tokens = tok(prefix_str, return_tensors="pt", add_special_tokens=False).to(device)
                suffix_tokens = tok(suffix_str, return_tensors="pt", add_special_tokens=False).to(device)
                
                prefix_seq = prefix_tokens["input_ids"].squeeze(0)
                suffix_seq = suffix_tokens["input_ids"].squeeze(0)
                
                prefix_ids_list.append(prefix_seq)
                suffix_ids_list.append(suffix_seq)
                prefix_lens.append(len(prefix_seq))
                suffix_lens.append(len(suffix_seq))
            
            # Pad sequences
            from torch.nn.utils.rnn import pad_sequence
            prefix_ids = pad_sequence(prefix_ids_list, batch_first=True, padding_value=PAD_ID)
            suffix_ids = pad_sequence(suffix_ids_list, batch_first=True, padding_value=PAD_ID)
            
            prefix_lens_tensor = torch.tensor(prefix_lens, dtype=torch.long, device=device)
            suffix_lens_tensor = torch.tensor(suffix_lens, dtype=torch.long, device=device)
            
            # Embed
            prefix_embed = embed_fn(prefix_ids)  # [B, L_prefix, H]
            suffix_embed = embed_fn(suffix_ids)  # [B, L_suffix, H]
            
            inputs_embeds = torch.cat([
                prefix_embed,
                Z,
                suffix_embed
            ], dim=1)
            
            # Build attention mask (mask padding positions)
            attention_mask = torch.ones(B, inputs_embeds.size(1), dtype=torch.long, device=device)
            L_prefix_padded = prefix_ids.size(1)
            L_suffix_padded = suffix_ids.size(1)
            
            for b in range(B):
                # Mask prefix padding
                actual_prefix_len = prefix_lens_tensor[b].item()
                if actual_prefix_len < L_prefix_padded:
                    attention_mask[b, actual_prefix_len:L_prefix_padded] = 0
                
                # Mask suffix padding
                suffix_start = L_prefix_padded + Ts
                actual_suffix_len = suffix_lens_tensor[b].item()
                if actual_suffix_len < L_suffix_padded:
                    attention_mask[b, suffix_start + actual_suffix_len:suffix_start + L_suffix_padded] = 0
        
        elif has_prompt:
            prompt_embed = self.prompt_embed
            prompt_embed_batch = prompt_embed.unsqueeze(0).expand(B, -1, -1)
            inputs_embeds = torch.cat([
                prompt_embed_batch,
                SEP_embed,
                Z,
                BOS_embed
            ], dim=1)
            attention_mask = torch.ones(B, inputs_embeds.size(1), dtype=torch.long, device=device)
        else:
            inputs_embeds = torch.cat([
                SEP_embed,
                Z,
                BOS_embed
            ], dim=1)
            attention_mask = torch.ones(B, inputs_embeds.size(1), dtype=torch.long, device=device)
        
        # Match LLM dtype
        if inputs_embeds.dtype != llm_dtype:
            inputs_embeds = inputs_embeds.to(llm_dtype)
        
        # Generate phoneme tokens
        gen_kwargs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "num_return_sequences": 1,
            "pad_token_id": PAD_ID,
            "eos_token_id": EOS_ID,
            "bos_token_id": BOS_ID,
            "do_sample": do_sample,
            "use_cache": True,
        }
        
        if do_sample:
            gen_kwargs.update({
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
            })
        
        gen_out = self.modules.LLM.generate(**gen_kwargs)
        
        # Decode generated tokens to text
        generated_text = tok.batch_decode(gen_out, skip_special_tokens=True)
        
        return {
            "ids": ids,
            "generated_tokens": gen_out,
            "generated_text": generated_text,
            "ctc_predictions": ctc_predictions,
        }

    def inference(
            self,
            test_set,
            max_key=None,
            min_key=None,
            progressbar=None,
            test_loader_kwargs={},
            max_new_tokens=100,
            do_sample=False,
            temperature=1.0,
            top_k=50,
            top_p=0.9,
            output_file=None,
        ):
            """
            Iterate test_set and perform inference (no loss computation).
            Only requires batch.id and batch.sig.

            Arguments
            ---------
            test_set : Dataset, DataLoader
                If a DataLoader is given, it is iterated directly. Otherwise passed
                to ``self.make_dataloader()``.
            max_key : str
                Key to use for finding best checkpoint.
            min_key : str
                Key to use for finding best checkpoint.
            progressbar : bool
                Whether to display the progress in a progressbar.
            test_loader_kwargs : dict
                Kwargs passed to ``make_dataloader()`` if ``test_set`` is not a
                DataLoader.
            max_new_tokens : int
                Maximum number of tokens to generate.
            do_sample : bool
                Whether to use sampling or greedy decoding.
            temperature : float
                Sampling temperature.
            top_k : int
                Top-k sampling parameter.
            top_p : float
                Top-p (nucleus) sampling parameter.
            output_file : str
                Optional file path to save results.

            Returns
            -------
            list of dict: All inference results
            """
            from torch.utils.data import DataLoader
            from speechbrain.dataio.dataloader import LoopedLoader
            
            if progressbar is None:
                progressbar = not self.noprogressbar

            enable = progressbar and sb.utils.distributed.if_main_process()

            if not (
                isinstance(test_set, DataLoader)
                or isinstance(test_set, LoopedLoader)
            ):
                test_loader_kwargs["ckpt_prefix"] = None
                test_set = self.make_dataloader(
                    test_set, sb.Stage.TEST, **test_loader_kwargs
                )
            
            # Load best checkpoint if available
            self.on_evaluate_start(max_key=max_key, min_key=min_key)
            self.modules.eval()
            
            all_results = []
            
            with torch.no_grad():
                for batch in tqdm(
                    test_set,
                    dynamic_ncols=True,
                    disable=not enable,
                    colour=self.tqdm_barcolor.get("test", "green"),
                ):
                    # Run inference
                    result = self.inference_batch(
                        batch,
                        max_new_tokens=max_new_tokens,
                        do_sample=do_sample,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        num_beams=3,
                        inference_prompt_mode="with_alternatives",
                    )
                    
                    # Collect results
                    for i, utt_id in enumerate(result["ids"]):
                        item = {
                            "id": utt_id,
                            "llm_prediction": result["generated_text"][i],
                        }
                        if result["ctc_predictions"] is not None:
                            item["ctc_prediction"] = result["ctc_predictions"][i]
                        all_results.append(item)
            # sort results by ID
            all_results = sorted(all_results, key=lambda x: x["id"])
            # Optionally save to file
            if output_file is not None:
                import json
                import csv
                import os
                
                # Determine output directory and base name
                output_dir = os.path.dirname(output_file) if os.path.dirname(output_file) else "."
                base_name = os.path.splitext(os.path.basename(output_file))[0]
                
                # Save LLM predictions to CSV
                llm_csv_path = os.path.join(output_dir, f"{base_name}_LLM.csv")
                with open(llm_csv_path, "w", encoding="utf-8", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["ID", "Labels"])
                    for item in all_results:
                        # Get file stem (filename without path and extension)
                        file_stem = os.path.splitext(os.path.basename(str(item["id"])))[0]
                        writer.writerow([file_stem, item["llm_prediction"]])
                print(f"LLM predictions saved to {llm_csv_path}")
                
                # Save CTC predictions to CSV (if available)
                if all_results and "ctc_prediction" in all_results[0]:
                    ctc_csv_path = os.path.join(output_dir, f"{base_name}_CTC.csv")
                    with open(ctc_csv_path, "w", encoding="utf-8", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(["ID", "Labels"])
                        for item in all_results:
                            file_stem = os.path.splitext(os.path.basename(str(item["id"])))[0]
                            writer.writerow([file_stem, item["ctc_prediction"]])
                    print(f"CTC predictions saved to {ctc_csv_path}")
                
                # Also save JSONL for full details
                jsonl_path = os.path.join(output_dir, f"{base_name}.jsonl")
                with open(jsonl_path, "w", encoding="utf-8") as f:
                    for item in all_results:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                print(f"Full results saved to {jsonl_path}")
            
            return all_results
    
    def on_stage_start(self, stage, epoch):
        "Gets called when a stage (either training, validation, test) starts."
        self.ctc_metrics = self.hparams.ctc_stats()
        self.llm_metrics = self.hparams.llm_stats()  # 添加LLM损失统计
        self.mpd_f1_metrics = MpdStats()  # LLM MPD F1统计
        self.ctc_mpd_f1_metrics = MpdStats()  # CTC MPD F1统计
        
        if hasattr(self.modules, "ctc_lin"):
            self.ctc_metrics = self.hparams.ctc_stats()

        if stage != sb.Stage.TRAIN:
            self.per_metrics = self.hparams.per_stats()
            self.llm_per_metrics = self.hparams.per_stats()# 添加LLM PER统计
            self.mpd_f1_metrics = MpdStats()  # LLM MPD F1统计
            self.ctc_mpd_f1_metrics = MpdStats()  # CTC MPD F1统计
            
    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a stage to summarize and log."""
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        elif stage == sb.Stage.VALID:
            stage_stats["epoch"] = epoch
            per_ctc = self.per_metrics.summarize("error_rate")
            per_llm = self.llm_per_metrics.summarize("error_rate")  # 添加LLM PER计算
            stage_stats["ctc_per"] = per_ctc
            stage_stats["llm_per"] = per_llm  # 添加LLM PER到统计
            llm_loss = self.llm_metrics.summarize("average")
            # Summarize and log metrics
            stage_stats["llm_loss"] = llm_loss
            
            # LLM MPD F1
            mpd_f1 = self.mpd_f1_metrics.summarize("mpd_f1")
            stage_stats["llm_mpd_f1"] = mpd_f1
            
            # CTC MPD F1
            ctc_mpd_f1 = self.ctc_mpd_f1_metrics.summarize("mpd_f1")
            stage_stats["ctc_mpd_f1"] = ctc_mpd_f1
        
            if hasattr(self.modules, "ctc_lin"):
                ctc_loss = self.ctc_metrics.summarize("average")
                stage_stats["ctc_loss"] = ctc_loss
                
        if stage == sb.Stage.VALID:
            self.hparams.train_logger.log_stats(
                stats_meta=stage_stats,
            )
            # You can add your custom checkpointing logic here
                # e.g., self.checkpointer.save_and_keep_only(meta={"PER": stage_stats['ctc_per']}, min_keys=["PER"])
            if epoch % self.hparams.valid_search_interval == 0:

                improved = False
                # ckpt_name = f"{epoch:03d}_CTC_PER_{per_ctc:.4f}_LLM_PER_{per_llm:.4f}.ckpt"
                ckpt_name = f"epoch{epoch:03d}_CTC_PER{per_ctc:.4f}_LLM_PER{per_llm:.4f}_LLM_MPD_F1_{mpd_f1:.4f}.ckpt"
                self.checkpointer.save_and_keep_only(meta={"CTC_PER": per_ctc, "LLM_PER": per_llm, "LLM_MPD_F1": mpd_f1},
                                                    name=ckpt_name,
                                                    num_to_keep=2,
                                                    importance_keys=[
                                                        lambda ckpt: (
                                                            -ckpt.meta["LLM_PER"],  
                                                            -ckpt.meta["CTC_PER"],  
                                                            -ckpt.meta["LLM_MPD_F1"]
                                                        )
                                                    ]
                                                )
                if stage_loss < self.best_valid_loss or len(self.best_valid_loss_list) < 10:
                    ckpt_name = f"best_valid_loss_{epoch:03d}_{stage_loss:.4f}.ckpt"
                    # Do NOT save checkpoint for valid loss (just update stats)
                    self.best_valid_loss_list.append((stage_loss, epoch, ckpt_name))
                    self.best_valid_loss_list = sorted(self.best_valid_loss_list, key=lambda x: x[0])[:10]
                    self.best_valid_loss = self.best_valid_loss_list[0][0]
                    improved = True

                if improved:
                    self.no_improve_epochs = 0
                    self.last_improved_epoch = epoch
                else:
                    self.no_improve_epochs += 1
            
            wandb.log({
                f"{stage.name.lower()}_loss": stage_loss,
                f"{stage.name.lower()}_ctc_per": per_ctc,
                f"{stage.name.lower()}_llm_per": per_llm,
                f"{stage.name.lower()}_ctc_loss": ctc_loss if hasattr(self.modules, "ctc_lin") else None,
                f"{stage.name.lower()}_llm_loss": llm_loss,
                f"{stage.name.lower()}_ctc_mpd_f1": ctc_mpd_f1 if hasattr(self, "ctc_mpd_f1_metrics") else None,
                f"{stage.name.lower()}_llm_mpd_f1": mpd_f1 if hasattr(self, "mpd_f1_metrics") else None,
            }, step=epoch)
            if self.no_improve_epochs >= self.patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {self.patience} epochs)")
                raise StopIteration
        
        if stage == sb.Stage.TEST:
            # import pdb; pdb.set_trace()
            self.hparams.train_logger.log_stats(
                stats_meta=stage_stats,
            )
            # import pdb; pdb.set_trace()
            per_ctc = self.per_metrics.summarize("error_rate")
            mpd_f1 = self.mpd_f1_metrics.summarize("mpd_f1")
            ctc_mpd_f1 = self.ctc_mpd_f1_metrics.summarize("mpd_f1")
            
            
            # Check if LLM metrics have data before summarizing (避免空数据错误)
            per_llm = None
            llm_loss = None
            
            try:
                per_llm = self.llm_per_metrics.summarize("error_rate")
            except (ZeroDivisionError, ValueError, IndexError, RuntimeError) as e:
                print(f"[Warning] LLM PER metrics are empty: {type(e).__name__}: {e}")
            
            try:
                llm_loss = self.llm_metrics.summarize("average")
            except (ZeroDivisionError, ValueError, IndexError, RuntimeError) as e:
                print(f"[Warning] LLM loss metrics are empty: {type(e).__name__}: {e}")
            
            ctc_loss = self.ctc_metrics.summarize("average")
            
            test_stats = {"loss": stage_loss, "PER": per_ctc}
            if per_llm is not None:
                test_stats["PER_seq"] = per_llm
            if llm_loss is not None:
                test_stats["llm_loss"] = llm_loss
            if hasattr(self.modules, "ctc_lin"):
                test_stats["ctc_loss"] = ctc_loss
            if hasattr(self, "mpd_f1_metrics"):
                test_stats["llm_mpd_f1"] = mpd_f1
            if hasattr(self, "ctc_mpd_f1_metrics"):
                test_stats["ctc_mpd_f1"] = ctc_mpd_f1
            
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=test_stats,
            )
            with open(self.hparams.per_file, "w") as w:
                w.write("CTC loss stats:\n")
                self.ctc_metrics.write_stats(w)
                w.write("\nCTC PER stats:\n")
                self.per_metrics.write_stats(w)
            
            # Only write LLM stats if data exists
            if llm_loss is not None or per_llm is not None:
                if not hasattr(self.hparams, 'per_seq_file'):
                    self.hparams.per_seq_file = self.hparams.per_file.replace('.txt', '_llm.txt')
                with open(self.hparams.per_seq_file, "w") as w:
                    if llm_loss is not None:
                        w.write("LLM loss stats:\n")
                        self.llm_metrics.write_stats(w)
                    if per_llm is not None:
                        w.write("\nLLM PER stats:\n")
                        self.llm_per_metrics.write_stats(w)
            
                if not hasattr(self.hparams, 'mpd_seq_file'):
                    self.hparams.mpd_seq_file = self.hparams.mpd_file.replace('.txt', '_llm.txt')

                with open(self.hparams.mpd_seq_file, "w") as m:
                    m.write("LLM MPD results and stats:\n")
                    self.mpd_f1_metrics.write_stats(m)
                    print(
                        "LLM MPD results and stats written to file",
                        self.hparams.mpd_seq_file,
                    )
                
                # 也保存CTC的MPD统计
                if not hasattr(self.hparams, 'ctc_mpd_seq_file'):
                    self.hparams.ctc_mpd_seq_file = self.hparams.mpd_file.replace('.txt', '_ctc.txt')
                
                with open(self.hparams.ctc_mpd_seq_file, "w") as m:
                    m.write("CTC MPD results and stats:\n")
                    self.ctc_mpd_f1_metrics.write_stats(m)
                    print(
                        "CTC MPD results and stats written to file",
                        self.hparams.ctc_mpd_seq_file,
                    )
                    
    def check_gradients(self, loss):
        """Check if gradients are finite"""
        if not torch.isfinite(loss):
            print("Warning: loss is not finite, skipping step")
            return False
        return True

    def fit_batch(self, batch):
        """Fit one batch, override to do multiple updates.

        The default implementation depends on a few methods being defined
        with a particular behavior:

        * ``compute_forward()``
        * ``compute_objectives()``

        Also depends on having optimizers passed at initialization.

        Arguments
        ---------
        batch : list of torch.Tensors
            Batch of data to use for training. Default implementation assumes
            this batch has two elements: inputs and targets.

        Returns
        -------
        detached loss
        """

        if self.hparams.auto_mix_prec:
            # self.pretrained_opt_class.zero_grad()
            self.adam_optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)

            # normalize the loss by gradient_accumulation and scale for mixed precision
            self.scaler.scale(loss / self.hparams.gradient_accumulation).backward()
            # self.scaler.unscale_(self.pretrained_opt_class)
            self.scaler.unscale_(self.adam_optimizer)

            if self.check_gradients(loss):
                # if any(p.requires_grad for p in self.pretrained_opt_class.param_groups[0]['params']):
                #     self.scaler.step(self.pretrained_opt_class)
                if any(p.requires_grad for p in self.adam_optimizer.param_groups[0]['params']):
                    self.scaler.step(self.adam_optimizer)

            self.scaler.update()

        else:
            outputs = self.compute_forward(batch, sb.Stage.TRAIN)
            
            loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            # normalize the loss by gradient_accumulation step
            (loss / self.hparams.gradient_accumulation).backward()

            if self.step % self.hparams.gradient_accumulation == 0:
                # gradient clipping & early stop if loss is not fini
                
                if self.check_gradients(loss):
                    self.pretrained_opt_class.step()
                    self.adam_optimizer.step()

                self.pretrained_opt_class.zero_grad()
                self.adam_optimizer.zero_grad()    

        return loss.detach().cpu()
    
    def load_pretrained_components(self, checkpoint_path, components_to_load=None, freeze_loaded=True):
        """
        Load specific components from a pretrained model checkpoint
        
        Args:
            checkpoint_path (str): Path to the checkpoint directory or file
            components_to_load (list): List of components to load. 
                                     Options: ['ssl'] for SSL model only,
                                              ['ssl', 'post_ssl_encoder'] for SSL + CTC branch, mainly for CTC per
                                     If None, loads ['ssl'] by default
            freeze_loaded (bool): Whether to freeze the loaded components
        """
        if components_to_load is None:
            components_to_load = ['ssl']  # Default: load SSL 
            logger.info("No components specified for loading, defaulting to ['ssl']")
            logger.info("For stable LLM pretraining, consider load a full Speech Encoder")
        
        logger.info(f"\n🔄 Loading pretrained components from: {checkpoint_path}")
        logger.info(f"   Components to load: {components_to_load}")
        
        from speechbrain.utils.parameter_transfer import Pretrainer

        if "post_ssl_encoder" in components_to_load:
            logger.info("⏳ Loading pretrained SSL model and post_ssl_encoder from %s", checkpoint_path)
            pretrainer = Pretrainer(
                collect_in=self.hparams.pretrained_model_path, 
                loadables={
                    "perceived_ssl":     self.hparams.perceived_ssl,
                    "post_encoder_modules":     self.hparams.post_encoder_modules,
                },
                paths={
                    "perceived_ssl":     "perceived_ssl.ckpt",
                    "post_encoder_modules":   "model.ckpt",
                },
                custom_hooks={}
            )
        # else:
        #     logger.info("⏳ Loading pretrained SSL model only from %s", checkpoint_path)
        #     pretrainer = Pretrainer(
        #         collect_in=self.hparams.pretrained_model_path, 
        #         loadables={
        #             "perceived_ssl":     self.modules.perceived_ssl,
        #         },
        #         paths={
        #             "perceived_ssl":     "perceived_ssl.ckpt",
        #         },
        #         custom_hooks={}
        #     )

        # DONE
        paths = pretrainer.collect_files(default_source=self.hparams.pretrained_model_path)

        # Check SSL
        # before = self.modules.perceived_ssl.state_dict()["model.encoder.layers.23.final_layer_norm.weight"]
        # Check CTC head
        # before = self.hparams.ctc_branch[1].state_dict()["w.weight"]
        # before = self.hparams.ssl_proj[1].state_dict()["w.weight"]
        # self.modules.ctc_lin.state_dict()['w.weight']
        
        # import pdb; pdb.set_trace()
        pretrainer.load_collected()
        # import pdb; pdb.set_trace()
        
        # Freeze loaded components if requested
        if freeze_loaded:
            for component in components_to_load:
                if component == 'ssl':
                    for param in self.modules.perceived_ssl.parameters():
                        param.requires_grad = False
                    self.ssl_frozen = True
                    logger.info("   🔒 SSL model frozen")
                    
                elif component == 'post_ssl_encoder':
                    for param in self.hparams.post_ssl_encoder.parameters():
                        param.requires_grad = False
                    self.post_ssl_encoder_frozen = True
                    logger.info("   🔒 Post SSL Encoder frozen")
                    
                elif component == 'ctc_head':
                    for param in self.modules.ctc_lin.parameters():
                        param.requires_grad = False
                    logger.info("   🔒 CTC head frozen")
    
        # print(f"   ✅ Successfully loaded components: {loaded_components}")
        # return loaded_components

    def on_fit_start(self):
        """Gets called at the beginning of ``fit()``, on multiple processes
        if ``distributed_count > 0`` and backend is ddp.

        Default implementation compiles the jit modules, initializes
        optimizers, and loads the latest checkpoint to resume training.
        """
        # Run this *after* starting all processes since jit modules cannot be
        # pickled.
        self._compile()

        # Wrap modules with parallel backend after jit
        self._wrap_distributed()

        self.init_optimizers()
        
        if self.checkpointer is not None:
            self.checkpointer.recover_if_possible(min_key="LLM_PER")
        
        # Only Init the AudioEncoderPretrainer with pretrained components during training
        # Essential for loading pretrained
        if getattr(self.hparams, 'load_pretrained_components', False):
            pretrained_path = getattr(self.hparams, 'pretrained_model_path', '')
            components = getattr(self.hparams, 'components_to_load', ['ssl'])
            freeze_loaded = getattr(self.hparams, 'freeze_loaded_components', True)
        
        # for AudioEncoderPretrainer loading
        import os
        if pretrained_path and os.path.exists(pretrained_path):
            try:
                self.hparams.AudioEncoderPretrainer.collect_files(default_source=self.hparams.pretrained_model_path)
                self.hparams.AudioEncoderPretrainer.load_collected()
                print("✅ Successfully loaded pretrained components.")
                # pdb.set_trace()
            except Exception as e:
                print(f"❌ Failed to load pretrained components: {e}")
                print("   Continuing with random initialization...")
        else:
            print(f"⚠️  Pretrained model path not found: {pretrained_path}")
            print("   Continuing with random initialization...")
   
    def init_optimizers(self):
        # Collect all trainable parameters
        model_params = list(self.hparams.model.parameters())
        trainable_model_params = list(self.hparams.trainable_model.parameters())
        # model_params = list(self.hparams.trainable_model.parameters())
        # audio_encoder_params = list(self.hparams.audio_encoder_modules.parameters())
        
        # Add prompt embeddings if they exist AND are learnable parameters
        # (Only soft prompts are nn.Parameter, text prompts are plain tensors)
        if hasattr(self, "prompt_embed") and self.prompt_embed is not None:
            if isinstance(self.prompt_embed, torch.nn.Parameter):
                model_params.append(self.prompt_embed)
                print(f"[Optimizer] Added soft prompt embeddings to optimizer")
            else:
                print(f"[Optimizer] Text prompt embeddings are frozen (not added to optimizer)")
        
        # Add LayerNorm parameters
        # if hasattr(self, "llm_norm") and self.llm_norm is not None:
        #     model_params.extend(self.llm_norm.parameters())
        
        # import pdb; pdb.set_trace()
        # Create optimizer with all model parameters

        self.adam_optimizer = self.hparams.adam_opt_class(
            trainable_model_params, 
        )
        
        
        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("adam_opt", self.adam_optimizer)
            # self.checkpointer.add_recoverable("pretrained_opt", self.pretrained_opt_class)
            self.checkpointer.add_recoverable("tokenizer", self.label_encoder)

