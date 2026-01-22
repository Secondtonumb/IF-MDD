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
    
class SSL_LLM_MultiTarget_ver1(sb.Brain):
    """Multi-target variant: jointly predicts words, canonical phonemes, and target phonemes.
    
    Sequence structure (TRAIN/VALID):
        [SEP(1)] [speech(Ts)] [BOS(1)] [words(L_wrd)] [canonical(L_can)] [target(L_tgt)] [EOS(1)]
    
    Loss computation:
        - Three independent CE losses: loss_wrd, loss_can, loss_tgt
        - Combined LLM loss: loss_llm = (loss_wrd + loss_can + loss_tgt) / 3
        - Total loss: 0.3 * loss_ctc + 0.7 * loss_llm
    
    Metrics computation (VALID/TEST):
        - Four PER values: per_wrd, per_can, per_tgt, per_avg
        - Four loss values: loss_wrd, loss_can, loss_tgt, loss_llm
    
    TEST inference:
        - Supports selective generation via target_to_generate parameter
        - Default: generates all three targets ["word", "canonical", "target"]
    """
    
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
        
        # 训练追踪
        self.best_valid_loss = float('inf')
        self.best_valid_loss_list = []  # List of (valid_loss, epoch, ckpt_name)
        self.train_stats = {"ctc_loss": [], "ce_loss": [], "total_loss": []}
        self.valid_stats = {"ctc_loss": [], "ce_loss": [], "total_loss": [], "per": []}
        
        # Multi-target specific stats
        self.multi_target_loss_stats = {
            "loss_wrd": [], "loss_can": [], "loss_tgt": [], "loss_llm": []
        }
        
        # 创建phoneme token掩码
        self.phoneme_bias = None
        self.setup_phoneme_mask()

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
        
    def _ensure_initialized(self):
        """Lazy initialization for components that need LLM to be loaded.
        This ensures inference works even without calling on_fit_start."""
        llm_handle = self.modules.LLM

        # 自动判断是否被 DDP 包裹
        import torch
        if isinstance(llm_handle, torch.nn.parallel.DistributedDataParallel):
            llm_handle = llm_handle.module
        
        embed_fn = llm_handle.get_input_embeddings()

        if not hasattr(self, "llm_norm") or self.llm_norm is None:
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

                chat_structure = [
                    {
                        "role": "system", 
                        "content": "You are a phoneme transcriber."
                    },
                    {
                        "role": "user", 
                        "content": f"{PLACEHOLDER}"
                    }
                ]

                full_prompt_str = tok.apply_chat_template(
                    chat_structure, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                
                if PLACEHOLDER not in full_prompt_str:
                    raise ValueError("Chat template processing removed the placeholder!")
                
                prefix_str, suffix_str = full_prompt_str.split(PLACEHOLDER)
                
                prefix_tokens = tok(prefix_str, return_tensors="pt", add_special_tokens=False).to(self.device)
                suffix_tokens = tok(suffix_str, return_tensors="pt", add_special_tokens=False).to(self.device)
                
                prefix_ids = prefix_tokens["input_ids"].squeeze(0)
                suffix_ids = suffix_tokens["input_ids"].squeeze(0)

                with torch.no_grad():
                    self.prompt_prefix_embed = embed_fn(prefix_ids)
                    self.prompt_suffix_embed = embed_fn(suffix_ids)
                
                self.prompt_embed = torch.cat([self.prompt_prefix_embed, self.prompt_suffix_embed], dim=0)
                print(f"[Lazy Init] Generated Llama 3 Prompt via Template.")

    def _build_multi_target_sequence(self, B, Z, Ts, prompt_len, has_split_prompt, prompt_embed_batch,
                                     wrd_embed, phn_can_embed, phn_tgt_embed,
                                     SEP_embed, BOS_embed, EOS_embed, SEP_TGT_embed):
        """Build multi-target embedding sequence: [SEP][speech][BOS][words][SEP_TGT][canonical][SEP_TGT][target][EOS]
        
        Args:
            All embedding tensors [B, *, H]
            
        Returns:
            inputs_embeds: [B, seq_len, H] - concatenated sequence
        """
        if has_split_prompt:
            inputs_embeds = torch.cat([
                self.prompt_prefix_embed.unsqueeze(0).expand(B, -1, -1),
                Z,
                self.prompt_suffix_embed.unsqueeze(0).expand(B, -1, -1),
                wrd_embed,
                SEP_TGT_embed,
                phn_can_embed,
                SEP_TGT_embed,
                phn_tgt_embed,
                EOS_embed
            ], dim=1)
        elif prompt_embed_batch is not None:
            inputs_embeds = torch.cat([
                prompt_embed_batch,
                SEP_embed,
                Z,
                BOS_embed,
                wrd_embed,
                SEP_TGT_embed,
                phn_can_embed,
                SEP_TGT_embed,
                phn_tgt_embed,
                EOS_embed
            ], dim=1)
        else:
            inputs_embeds = torch.cat([
                SEP_embed,
                Z,
                BOS_embed,
                wrd_embed,
                SEP_TGT_embed,
                phn_can_embed,
                SEP_TGT_embed,
                phn_tgt_embed,
                EOS_embed
            ], dim=1)
        
        return inputs_embeds

    def _build_inference_sequence(self, B, Z, has_split_prompt, prompt_embed_batch,
                                  SEP_embed, BOS_embed):
        """Build inference-only embedding sequence for generation: [SEP][speech][BOS]
        
        Used in TEST stage for autoregressive generation without target prefilling.
        
        Args:
            B: Batch size
            Z: Speech embeddings [B, Ts, H]
            has_split_prompt: Whether using split prompt tuning
            prompt_embed_batch: Soft prompt batch embeddings [B, P, H]
            SEP_embed: SEP token embedding [B, 1, H]
            BOS_embed: BOS token embedding [B, 1, H]
            
        Returns:
            inputs_embeds: [B, seq_len, H] - concatenated sequence
        """
        if has_split_prompt:
            inputs_embeds = torch.cat([
                self.prompt_prefix_embed.unsqueeze(0).expand(B, -1, -1),
                Z,
                self.prompt_suffix_embed.unsqueeze(0).expand(B, -1, -1)
            ], dim=1)
        elif prompt_embed_batch is not None:
            inputs_embeds = torch.cat([
                prompt_embed_batch,
                SEP_embed,
                Z,
                BOS_embed
            ], dim=1)
        else:
            inputs_embeds = torch.cat([
                SEP_embed,
                Z,
                BOS_embed
            ], dim=1)
        
        return inputs_embeds

    def _build_input_embeddings(self, B, Z, phn_embed, SEP_embed, BOS_embed, EOS_embed, 
                                has_split_prompt=False, prompt_embed_batch=None):
        """Build input embedding sequence for LLM forward pass.
        
        For multi-target: [SEP] [speech] [BOS] [words] [canonical] [target] [EOS]
        """
        if has_split_prompt:
            prefix_embed = self.prompt_prefix_embed
            suffix_embed = self.prompt_suffix_embed
            inputs_embeds = torch.cat([
                prefix_embed.unsqueeze(0).expand(B, -1, -1),
                Z,
                suffix_embed.unsqueeze(0).expand(B, -1, -1),
                phn_embed,
                EOS_embed
            ], dim=1)
        elif prompt_embed_batch is not None:
            inputs_embeds = torch.cat([
                prompt_embed_batch,
                SEP_embed,
                Z,
                BOS_embed,
                phn_embed,
                EOS_embed
            ], dim=1)
        else:
            inputs_embeds = torch.cat([
                SEP_embed,
                Z,
                BOS_embed,
                phn_embed,
                EOS_embed
            ], dim=1)
        
        return inputs_embeds

    def compute_forward(self, batch, stage):
        """Given an input batch it computes the model forward pass.
        
        Multi-target: predicts words, canonical phonemes, and target phonemes.
        
        Returns:
            - p_ctc: [B, T, 41] - CTC logits 
            - ce_logits: [B, L, 128000] or None - LLM logits over big vocab
            - ce_targets: dict with target token IDs and position tracking, or None
            - wav_lens: [B] - sequence lengths
        """
        # Ensure critical components are initialized
        self._ensure_initialized()
        
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "augmentation"):
            wavs = self.hparams.augmentation(wavs)

        # AudioEncoder + Projector
        wav_feats = self.modules.perceived_ssl(wavs)
        
        # AudioEncoder
        try:
            Z, _ = self.hparams.audio_encoder_modules(wavs)
        except:
            Z = self.hparams.audio_encoder_modules(wavs)
        
        # CTC branch
        ctc_logits = self.modules.ctc_lin(Z)
        p_ctc = self.hparams.log_softmax(ctc_logits)

        if hasattr(self.modules, "projector") and self.modules.projector is not None:
            Z = self.modules.projector(Z)
        
        B, Ts, H = Z.shape
        device = self.device
        tok = self.hparams.LLM_tokenizer
        embed_fn = self.modules.LLM.get_input_embeddings()
        
        # ===== Tokenize multi-target sequences =====
        # Target phonemes (required)
        phn_tgt_seq = phn_list_to_seq(batch.phn_list_target)
        phn_tgt_tokens = tok(phn_tgt_seq, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        phn_tgt_ids = phn_tgt_tokens["input_ids"]
        phn_tgt_mask = phn_tgt_tokens["attention_mask"]
        L_tgt = phn_tgt_ids.size(1)
        
        # Canonical phonemes (required)
        phn_can_seq = phn_list_to_seq(batch.phn_list_canonical)
        phn_can_tokens = tok(phn_can_seq, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        phn_can_ids = phn_can_tokens["input_ids"]
        phn_can_mask = phn_can_tokens["attention_mask"]
        L_can = phn_can_ids.size(1)
        
        # Words (required)
        wrd_seq = phn_list_to_seq(batch.wrd)
        wrd_tokens = tok(wrd_seq, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        wrd_ids = wrd_tokens["input_ids"]
        wrd_mask = wrd_tokens["attention_mask"]
        L_wrd = wrd_ids.size(1)
        
        # ===== Prepare special tokens =====
        BOS_ID = tok.bos_token_id
        EOS_ID = tok.eos_token_id
        PAD_ID = tok.pad_token_id if tok.pad_token_id is not None else 0
        
        # Use newline as separator token (simpler than special tokens)
        newline_tokens = tok("\n", return_tensors="pt", add_special_tokens=False).to(device)
        SEP_ID = newline_tokens["input_ids"][0, 0].item()
        
        # SEP_TGT token for separating multi-target outputs (words | canonical | target)
        # Use the same separator token for consistency
        SEP_TGT_ID = SEP_ID
        
        # Store as instance attributes for access in compute_objectives
        self.BOS_ID = BOS_ID
        self.EOS_ID = EOS_ID
        self.PAD_ID = PAD_ID
        self.SEP_ID = SEP_ID
        self.SEP_TGT_ID = SEP_TGT_ID
        
        def col_tokens(tok_id):
            return torch.full((B, 1), tok_id, dtype=torch.long, device=device)
        
        SEP = col_tokens(SEP_ID)
        BOS = col_tokens(BOS_ID)
        EOS = col_tokens(EOS_ID)
        SEP_TGT = col_tokens(SEP_TGT_ID)
        
        # ===== Optional prompt tuning =====
        use_prompt = getattr(self.hparams, "use_prompt", False)
        prompt_len = 0
        
        has_prompt = use_prompt and hasattr(self, "prompt_embed") and self.prompt_embed is not None
        
        if has_prompt:
            prompt_embed = self.prompt_embed
            prompt_len = prompt_embed.size(0)
            prompt_embed_batch = prompt_embed.unsqueeze(0).expand(B, -1, -1)
        else:
            prompt_embed = None
            prompt_embed_batch = None
        
        # ===== Build multi-target embedding sequence =====
        # Sequence: [SEP] [speech] [BOS] [words] [canonical] [target] [EOS]
        
        # Embed all token sequences
        wrd_embed = embed_fn(wrd_ids)  # [B, L_wrd, H]
        phn_can_embed = embed_fn(phn_can_ids)  # [B, L_can, H]
        phn_tgt_embed = embed_fn(phn_tgt_ids)  # [B, L_tgt, H]
        # Embed special tokens
        SEP_embed = embed_fn(SEP)
        BOS_embed = embed_fn(BOS)
        EOS_embed = embed_fn(EOS)
        SEP_TGT_embed = embed_fn(SEP_TGT)  # [B, 1, H]
        
        # Use helper function to build embedding sequence
        has_split_prompt = use_prompt and hasattr(self, "prompt_prefix_embed") and hasattr(self, "prompt_suffix_embed")
        
        inputs_embeds = self._build_multi_target_sequence(
            B=B,
            Z=Z,
            Ts=Ts,
            prompt_len=prompt_len,
            has_split_prompt=has_split_prompt,
            prompt_embed_batch=prompt_embed_batch,
            wrd_embed=wrd_embed,
            phn_can_embed=phn_can_embed,
            phn_tgt_embed=phn_tgt_embed,
            SEP_embed=SEP_embed,
            BOS_embed=BOS_embed,
            EOS_embed=EOS_embed,
            SEP_TGT_embed=SEP_TGT_embed,
        )
        
        # Align dtype with LLM weights
        llm_dtype = embed_fn.weight.dtype
        if inputs_embeds.dtype != llm_dtype:
            inputs_embeds = inputs_embeds.to(llm_dtype)
        
        # ===== Build attention mask =====
        seq_len = inputs_embeds.size(1)
        
        if stage == sb.Stage.TRAIN:
            attention_mask = torch.ones(B, seq_len, dtype=torch.long, device=device)
            
            if has_split_prompt:
                text_start = self.prompt_prefix_embed.size(0) + Ts + self.prompt_suffix_embed.size(0)
                bos_pos = text_start - 1
            else:
                sep_pos = prompt_len
                speech_start = sep_pos + 1
                speech_end = speech_start + Ts
                bos_pos = speech_end
                text_start = bos_pos + 1
            
            # Build position ranges for each target (accounting for SEP_TGT separators)
            # Sequence: [SEP] [speech] [BOS] [words] [SEP_TGT] [canonical] [SEP_TGT] [target] [EOS]
            wrd_start = text_start
            wrd_end = wrd_start + L_wrd
            sep_tgt_1_pos = wrd_end  # First SEP_TGT after words
            can_start = wrd_end + 1  # Canonical starts after first SEP_TGT
            can_end = can_start + L_can
            sep_tgt_2_pos = can_end  # Second SEP_TGT after canonical
            tgt_start = can_end + 1  # Target starts after second SEP_TGT
            tgt_end = tgt_start + L_tgt
            eos_pos = tgt_end
            
            # Mask out padding in each sequence segment
            # Compact format: [BOS] [words L_wrd] [SEP_TGT] [canonical L_can] [SEP_TGT] [target L_tgt] [padding] [EOS]
            for b in range(B):
                num_wrd = int(wrd_mask[b].sum().item())
                num_can = int(phn_can_mask[b].sum().item())
                num_tgt = int(phn_tgt_mask[b].sum().item())
                
                # Mask padding in target sequence (after actual tokens, before EOS)
                if num_tgt < L_tgt:
                    # Target padding: [tgt_start + num_tgt, tgt_end)
                    attention_mask[b, tgt_start + num_tgt:tgt_end] = 0
                
                # Mask padding in canonical sequence (after actual tokens, before SEP_TGT_2)
                if num_can < L_can:
                    # Canonical padding: [can_start + num_can, can_end)
                    attention_mask[b, can_start + num_can:can_end] = 0
                
                # Mask padding in word sequence (after actual tokens, before SEP_TGT_1)
                if num_wrd < L_wrd:
                    # Word padding: [wrd_start + num_wrd, wrd_end)
                    attention_mask[b, wrd_start + num_wrd:wrd_end] = 0
            
            # Build labels for multi-target (left-shifted for causal LM)
            # Sequence in embeddings: [SEP] [speech] [BOS] [words L_wrd] [SEP_TGT] [canonical L_can] [SEP_TGT] [target L_tgt] [EOS]
            # Position i predicts token at position i+1
            # Labels mapping: position i → token that should appear at position i+1
            ignore_idx = -100
            labels = torch.full((B, seq_len), ignore_idx, dtype=torch.long, device=device)
            
            for b in range(B):
                num_wrd = int(wrd_mask[b].sum().item())
                num_can = int(phn_can_mask[b].sum().item())
                num_tgt = int(phn_tgt_mask[b].sum().item())
                
                if num_wrd > 0:
                    # BOS → word[0], word[0] → word[1], ..., word[num_wrd-2] → word[num_wrd-1], word[num_wrd-1] → SEP_TGT
                    labels[b, bos_pos:bos_pos + num_wrd] = wrd_ids[b, :num_wrd]
                    
                    # Last word position predicts SEP_TGT (if more targets follow)
                    if num_can > 0 or num_tgt > 0:
                        labels[b, bos_pos + num_wrd - 1] = SEP_TGT_ID
                
                if num_can > 0:
                    # Canonical chain
                    if num_wrd == 0:
                        # If no words, BOS directly predicts canonical
                        labels[b, bos_pos:bos_pos + num_can] = phn_can_ids[b, :num_can]
                    else:
                        # SEP_TGT position → can[0], can[0] → can[1], ..., can[num_can-1] → SEP_TGT (or EOS if no target)
                        labels[b, sep_tgt_1_pos:sep_tgt_1_pos + num_can] = phn_can_ids[b, :num_can]
                    
                    # Last canonical predicts second SEP_TGT (if target exists) or EOS (if no target)
                    if num_tgt > 0:
                        if num_wrd == 0:
                            labels[b, bos_pos + num_can - 1] = SEP_TGT_ID
                        else:
                            labels[b, sep_tgt_1_pos + num_can - 1] = SEP_TGT_ID
                    else:
                        # No target: last canonical predicts EOS
                        if num_wrd == 0:
                            labels[b, bos_pos + num_can - 1] = EOS_ID
                        else:
                            labels[b, sep_tgt_1_pos + num_can - 1] = EOS_ID
                
                if num_tgt > 0:
                    # Target chain
                    if num_can == 0 and num_wrd == 0:
                        # If no words and no canonical, BOS directly predicts target
                        labels[b, bos_pos:bos_pos + num_tgt] = phn_tgt_ids[b, :num_tgt]
                    elif num_can == 0:
                        # If no canonical, words chain directly to target
                        labels[b, sep_tgt_1_pos:sep_tgt_1_pos + num_tgt] = phn_tgt_ids[b, :num_tgt]
                    else:
                        # SEP_TGT[2] position → tgt[0], tgt[i] → tgt[i+1], tgt[num_tgt-1] → EOS
                        labels[b, sep_tgt_2_pos:sep_tgt_2_pos + num_tgt] = phn_tgt_ids[b, :num_tgt]
                    
                    # Last target position predicts EOS
                    if num_tgt > 0:
                        labels[b, sep_tgt_2_pos + num_tgt] = EOS_ID
            
            # Run forward pass
            llm_out = self.modules.LLM(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=False,
                return_dict=True,
            )
            
            ce_logits = llm_out.logits
            
            # Debug: verify labels and positions (only run once)
            if not hasattr(self, '_label_check_done'):
                b_debug = 0  # First batch
                num_wrd_debug = int(wrd_mask[b_debug].sum().item())
                num_can_debug = int(phn_can_mask[b_debug].sum().item())
                num_tgt_debug = int(phn_tgt_mask[b_debug].sum().item())
                
                print(f"[DEBUG] TRAIN Label Structure:")
                print(f"  Positions: bos={bos_pos}, wrd_end={wrd_end}, sep_tgt_1={sep_tgt_1_pos}, can_end={can_end}, sep_tgt_2={sep_tgt_2_pos}, tgt_end={tgt_end}")
                print(f"  Counts: num_wrd={num_wrd_debug}, num_can={num_can_debug}, num_tgt={num_tgt_debug}")
                
                # Check key label positions
                if num_wrd_debug > 0:
                    wrd_labels = labels[b_debug, bos_pos:bos_pos + num_wrd_debug]
                    wrd_label_values = wrd_labels.tolist()
                    print(f"  Word labels (bos_pos:{bos_pos}): {wrd_label_values}")
                    print(f"    → Last word label (should predict SEP_TGT={SEP_TGT_ID}): {labels[b_debug, bos_pos + num_wrd_debug - 1].item()}")
                
                if num_can_debug > 0 and num_wrd_debug > 0:
                    can_labels = labels[b_debug, sep_tgt_1_pos:sep_tgt_1_pos + num_can_debug]
                    print(f"  Can labels (sep_tgt_1_pos:{sep_tgt_1_pos}): first few = {can_labels[:min(3, len(can_labels))].tolist()}")
                    print(f"    → Last can label (should predict SEP_TGT={SEP_TGT_ID}): {labels[b_debug, sep_tgt_1_pos + num_can_debug - 1].item()}")
                
                if num_tgt_debug > 0 and num_can_debug > 0:
                    tgt_labels = labels[b_debug, sep_tgt_2_pos:sep_tgt_2_pos + num_tgt_debug]
                    print(f"  Tgt labels (sep_tgt_2_pos:{sep_tgt_2_pos}): first few = {tgt_labels[:min(3, len(tgt_labels))].tolist()}")
                    print(f"    → Last tgt label (should predict EOS={EOS_ID}): {labels[b_debug, sep_tgt_2_pos + num_tgt_debug].item()}")
                
                self._label_check_done = True
            
            # Debug: verify causal masking (only run once)
            if not hasattr(self, '_causal_check_done'):
                print(f"[INFO] MultiTarget Model - LLM type: {type(self.modules.LLM).__name__}")
                print(f"[INFO] Input shape: {inputs_embeds.shape}, Logits shape: {ce_logits.shape}")
                print(f"[INFO] Position ranges (with SEP_TGT) - WRD:[{wrd_start},{wrd_end}), SEP_TGT1:{sep_tgt_1_pos}, CAN:[{can_start},{can_end}), SEP_TGT2:{sep_tgt_2_pos}, TGT:[{tgt_start},{tgt_end})")
                self._causal_check_done = True

            return p_ctc, ce_logits, {
                "labels": labels,
                "position_ranges": {
                    "wrd": (wrd_start, wrd_end, wrd_mask),
                    "can": (can_start, can_end, phn_can_mask),
                    "tgt": (tgt_start, tgt_end, phn_tgt_mask),
                    "bos": bos_pos
                }
            }, wav_lens
        
        else:
            # ===== Validation/Test Stage =====
            if stage == sb.Stage.VALID:
                attention_mask = torch.ones(B, seq_len, dtype=torch.long, device=device)
                
                if use_prompt and hasattr(self, "prompt_prefix_embed") and hasattr(self, "prompt_suffix_embed"):
                    text_start = self.prompt_prefix_embed.size(0) + Ts + self.prompt_suffix_embed.size(0)
                else:
                    sep_pos = prompt_len
                    speech_start = sep_pos + 1
                    speech_end = speech_start + Ts
                    bos_pos = speech_end
                    text_start = bos_pos + 1
                
                # Calculate position ranges (with SEP_TGT separators)
                wrd_start = text_start
                wrd_end = wrd_start + L_wrd
                sep_tgt_1_pos = wrd_end
                can_start = wrd_end + 1
                can_end = can_start + L_can
                sep_tgt_2_pos = can_end
                tgt_start = can_end + 1
                tgt_end = tgt_start + L_tgt
                
                for b in range(B):
                    num_tgt = int(phn_tgt_mask[b].sum().item())
                    if num_tgt < L_tgt:
                        attention_mask[b, tgt_start + num_tgt + 1:] = 0
                
                llm_out = self.modules.LLM(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    output_hidden_states=False,
                    return_dict=True,
                )
                ce_logits = llm_out.logits
                
                # Build labels same as training (compact format with chain structure)
                ignore_idx = -100
                labels = torch.full((B, seq_len), ignore_idx, dtype=torch.long, device=device)
                
                if use_prompt and hasattr(self, "prompt_prefix_embed") and hasattr(self, "prompt_suffix_embed"):
                    text_start = self.prompt_prefix_embed.size(0) + Ts + self.prompt_suffix_embed.size(0)
                    bos_pos = text_start - 1
                else:
                    sep_pos = prompt_len
                    bos_pos = sep_pos + 1 + Ts
                    text_start = bos_pos + 1
                
                wrd_start = text_start
                wrd_end = wrd_start + L_wrd
                sep_tgt_1_pos = wrd_end
                can_start = wrd_end + 1
                can_end = can_start + L_can
                sep_tgt_2_pos = can_end
                tgt_start = can_end + 1
                tgt_end = tgt_start + L_tgt
                
                # Build labels with supervised SEP_TGT learning (same as TRAIN stage)
                for b in range(B):
                    num_wrd = int(wrd_mask[b].sum().item())
                    num_can = int(phn_can_mask[b].sum().item())
                    num_tgt = int(phn_tgt_mask[b].sum().item())
                    
                    if num_wrd > 0:
                        # Left-shifted: BOS predicts word[0], word[i] predicts word[i+1]
                        labels[b, bos_pos:bos_pos + num_wrd] = wrd_ids[b, :num_wrd]
                        
                        # Last word predicts first SEP_TGT (if more targets follow)
                        if num_can > 0 or num_tgt > 0:
                            labels[b, bos_pos + num_wrd - 1] = SEP_TGT_ID
                    
                    if num_can > 0:
                        if num_wrd == 0:
                            # No words: BOS directly predicts canonical
                            labels[b, bos_pos:bos_pos + num_can] = phn_can_ids[b, :num_can]
                        else:
                            # SEP_TGT[1] predicts can[0], can[i] predicts can[i+1]
                            labels[b, sep_tgt_1_pos:sep_tgt_1_pos + num_can] = phn_can_ids[b, :num_can]
                        
                        # Last canonical predicts second SEP_TGT (if target exists) or EOS
                        if num_tgt > 0:
                            if num_wrd == 0:
                                labels[b, bos_pos + num_can - 1] = SEP_TGT_ID
                            else:
                                labels[b, sep_tgt_1_pos + num_can - 1] = SEP_TGT_ID
                        else:
                            # No target: last canonical predicts EOS
                            if num_wrd == 0:
                                labels[b, bos_pos + num_can - 1] = self.EOS_ID
                            else:
                                labels[b, sep_tgt_1_pos + num_can - 1] = self.EOS_ID
                    
                    if num_tgt > 0:
                        if num_can == 0 and num_wrd == 0:
                            labels[b, bos_pos:bos_pos + num_tgt] = phn_tgt_ids[b, :num_tgt]
                        elif num_can == 0:
                            labels[b, sep_tgt_1_pos:sep_tgt_1_pos + num_tgt] = phn_tgt_ids[b, :num_tgt]
                        else:
                            # SEP_TGT[2] predicts tgt[0], tgt[i] predicts tgt[i+1]
                            labels[b, sep_tgt_2_pos:sep_tgt_2_pos + num_tgt] = phn_tgt_ids[b, :num_tgt]
                        
                        # Last target predicts EOS
                        labels[b, sep_tgt_2_pos + num_tgt] = self.EOS_ID
                
                return p_ctc, ce_logits, {
                    "labels": labels,
                    "position_ranges": {
                        "wrd": (wrd_start, wrd_end, wrd_mask),
                        "can": (can_start, can_end, phn_can_mask),
                        "tgt": (tgt_start, tgt_end, phn_tgt_mask),
                        "bos": bos_pos
                    }
                }, wav_lens
            
            elif stage == sb.Stage.TEST:
                # ===== Autoregressive generation (no teacher forcing) =====
                # Only provide: [SEP] [speech] [BOS]
                # Model generates: selected targets + EOS
                
                has_split_prompt = use_prompt and hasattr(self, "prompt_prefix_embed") and hasattr(self, "prompt_suffix_embed")
                
                inputs_embeds_inference = self._build_inference_sequence(
                    B=B,
                    Z=Z,
                    has_split_prompt=has_split_prompt,
                    prompt_embed_batch=prompt_embed_batch,
                    SEP_embed=SEP_embed,
                    BOS_embed=BOS_embed
                )
                
                if inputs_embeds_inference.dtype != llm_dtype:
                    inputs_embeds_inference = inputs_embeds_inference.to(llm_dtype)
                
                attention_mask_inference = torch.ones(B, inputs_embeds_inference.size(1), dtype=torch.long, device=device)
                
                # Get target_to_generate from hparams (default: all three)
                target_to_generate = getattr(self.hparams, "target_to_generate", ["word", "canonical", "target"])
                
                # Calculate max_new_tokens accounting for full chain structure:
                # [word_tokens] [SEP_TGT] [can_tokens] [SEP_TGT] [tgt_tokens] [EOS]
                # Need: L_wrd + 1 + L_can + 1 + L_tgt + 1 = total tokens to generate
                num_targets_active = sum([1 for t in ["word", "canonical", "target"] if t in target_to_generate])
                num_sep_tgt = max(0, num_targets_active - 1)  # Separators between targets
                num_eos = 1  # Final EOS token
                
                max_gen_len = (
                    (L_wrd if "word" in target_to_generate else 0) +
                    (L_can if "canonical" in target_to_generate else 0) +
                    (L_tgt if "target" in target_to_generate else 0) +
                    num_sep_tgt +
                    num_eos +
                    5  # Small safety margin for variations
                )
                
                gen_out = self.modules.LLM.generate(
                    inputs_embeds=inputs_embeds_inference,
                    attention_mask=attention_mask_inference,
                    num_return_sequences=1,
                    pad_token_id=PAD_ID,
                    eos_token_id=EOS_ID,
                    bos_token_id=BOS_ID,
                    do_sample=False,
                    use_cache=True,
                    num_beams=1,
                    max_new_tokens=max_gen_len,
                    temperature=0.1,
                    repetition_penalty=1.0,  
                    length_penalty=1.0,      
                    early_stopping=True,     
                )

                gen_tokens = gen_out
                
                return p_ctc, None, {
                    "generated_ids": gen_tokens,
                    "target_phonemes": batch.phn_list_target,
                    "canonical_phonemes": batch.phn_list_canonical,
                    "words": batch.wrd,
                    "target_to_generate": target_to_generate
                }, wav_lens

    def compute_objectives(self, predictions, batch, stage):
        """Compute multi-target training objectives with separate losses for each target."""
        ids = batch.id
        wavs, wav_lens = batch.sig
        targets, target_lens = batch.phn_encoded_target
        
        # Unpack predictions from compute_forward
        if len(predictions) == 4:
            p_ctc, ce_logits, ce_targets, lens_for_ctc = predictions
        else:
            raise ValueError(f"Expected 4 values from compute_forward, got {len(predictions)}")
        
        # ===== CTC Loss =====
        T = p_ctc.size(1)
        clipped_target_lens = torch.minimum(target_lens, torch.full_like(target_lens, T))
        loss_ctc = self.hparams.ctc_cost(p_ctc.float(), targets, lens_for_ctc, clipped_target_lens)
        
        # ===== Multi-Target LLM Loss =====
        loss_wrd = torch.tensor(0.0, device=self.device)
        loss_can = torch.tensor(0.0, device=self.device)
        loss_tgt = torch.tensor(0.0, device=self.device)
        loss_ce = torch.tensor(0.0, device=self.device)
        
        if stage != sb.Stage.TEST:
            if ce_logits is not None and isinstance(ce_targets, dict) and "labels" in ce_targets:
                labels = ce_targets["labels"]
                pos_ranges = ce_targets.get("position_ranges", {})
                
                if labels is not None and pos_ranges:
                    B, seq_len, vocab_size = ce_logits.shape
                    ignore_idx = -100
                    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_idx)
                    
                    # Extract position ranges
                    wrd_start, wrd_end, wrd_mask = pos_ranges.get("wrd", (0, 0, None))
                    can_start, can_end, can_mask = pos_ranges.get("can", (0, 0, None))
                    tgt_start, tgt_end, tgt_mask = pos_ranges.get("tgt", (0, 0, None))
                    
                    # Compute loss for each target
                    if wrd_start < wrd_end:
                        wrd_logits = ce_logits[:, wrd_start:wrd_end, :].reshape(-1, vocab_size)
                        wrd_labels = labels[:, wrd_start:wrd_end].reshape(-1)
                        loss_wrd = ce_loss_fn(wrd_logits, wrd_labels)
                    
                    if can_start < can_end:
                        can_logits = ce_logits[:, can_start:can_end, :].reshape(-1, vocab_size)
                        can_labels = labels[:, can_start:can_end].reshape(-1)
                        loss_can = ce_loss_fn(can_logits, can_labels)
                    
                    if tgt_start < tgt_end:
                        tgt_logits = ce_logits[:, tgt_start:tgt_end, :].reshape(-1, vocab_size)
                        tgt_labels = labels[:, tgt_start:tgt_end].reshape(-1)
                        loss_tgt = ce_loss_fn(tgt_logits, tgt_labels)
                    
                    # Combined loss (equal weights: 1:1:1)
                    loss_ce = (loss_wrd + loss_can + loss_tgt) / 3.0

        # --- 2. Compute Metrics (VALID & TEST) ---
        if stage != sb.Stage.TRAIN:
            # 2.1 CTC Metrics
            ctc_sequence = sb.decoders.ctc_greedy_decode(
                p_ctc, wav_lens, blank_id=self.hparams.blank_index
            )
            self.ctc_metrics.append(ids, p_ctc, targets, lens_for_ctc, clipped_target_lens)
            try:
                # pop id==70 from ctc
                ctc_sequence = [[tok for tok in seq if tok != 70] for seq in ctc_sequence]
                self.per_metrics.append(
                    ids=ids,
                    predict=ctc_sequence,
                    target=targets,
                    predict_len=None,
                    target_len=target_lens,
                    ind2lab=self.label_encoder.decode_ndim,
                )
            except Exception as e:
                logger.error(f"Error in CTC metrics: {type(e).__name__}: {e}")

            # 2.2 Multi-Target LLM Metrics
            if stage == sb.Stage.VALID:
                # Teacher-Forcing Evaluation
                try:
                    canonical = batch.phn_list_canonical
                    perceived = batch.phn_list_perceived
                except:
                    canonical = None
                    perceived = None

                if ce_logits is not None and isinstance(ce_targets, dict) and "labels" in ce_targets:
                    labels = ce_targets["labels"]
                    pos_ranges = ce_targets.get("position_ranges", {})
                    
                    llm_predictions = ce_logits.argmax(dim=-1)
                    p_llm = F.log_softmax(ce_logits, dim=-1)
                    
                    valid_mask = (labels != -100)
                    B = labels.size(0)
                    
                    wrd_start, wrd_end, _ = pos_ranges.get("wrd", (0, 0, None))
                    can_start, can_end, _ = pos_ranges.get("can", (0, 0, None))
                    tgt_start, tgt_end, _ = pos_ranges.get("tgt", (0, 0, None))
                    
                    for b in range(B):
                        valid_pos = valid_mask[b]
                        num_valid = valid_pos.sum().item()
                        if num_valid > 0:
                            self.llm_metrics.append(
                                ids=[ids[b]],
                                log_probabilities=p_llm[b, valid_pos, :].unsqueeze(0),
                                targets=labels[b, valid_pos].unsqueeze(0),
                                length=torch.tensor([num_valid], device=labels.device)
                            )
                            
                            # PER for each target
                            pred_tokens = llm_predictions[b, valid_pos]
                            valid_labels = labels[b, valid_pos]
                            
                            pred_text = self.hparams.LLM_tokenizer.decode(pred_tokens, skip_special_tokens=True)
                            target_text = self.hparams.LLM_tokenizer.decode(valid_labels, skip_special_tokens=True)
                            
                            # Store per-target metrics
                            if wrd_start < wrd_end:
                                wrd_pred = self.hparams.LLM_tokenizer.decode(
                                    llm_predictions[b, wrd_start:min(wrd_end, seq_len)], skip_special_tokens=True
                                )
                                self.wrd_per_metrics.append(
                                    ids=[ids[b]],
                                    predict=[wrd_pred.split()],
                                    target=[batch.wrd[b]],
                                    predict_len=None,
                                    target_len=None,
                                    ind2lab=lambda x: x
                                )
                            
                            if can_start < can_end:
                                can_pred = self.hparams.LLM_tokenizer.decode(
                                    llm_predictions[b, can_start:min(can_end, seq_len)], skip_special_tokens=True
                                )
                                self.can_per_metrics.append(
                                    ids=[ids[b]],
                                    predict=[can_pred.split()],
                                    target=[batch.phn_list_canonical[b]],
                                    predict_len=None,
                                    target_len=None,
                                    ind2lab=lambda x: x
                                )
                            
                            if tgt_start < tgt_end:
                                tgt_pred = self.hparams.LLM_tokenizer.decode(
                                    llm_predictions[b, tgt_start:min(tgt_end, seq_len)], skip_special_tokens=True
                                )
                                self.tgt_per_metrics.append(
                                    ids=[ids[b]],
                                    predict=[tgt_pred.split()],
                                    target=[batch.phn_list_target[b]],
                                    predict_len=None,
                                    target_len=None,
                                    ind2lab=lambda x: x
                                )
                            
                            # Overall PER
                            self.llm_per_metrics.append(
                                ids=[ids[b]],
                                predict=[pred_text.split()],
                                target=[target_text.split()],
                                predict_len=None,
                                target_len=None,
                                ind2lab=lambda x: x
                            )
                            
                            # MPD F1
                            self.mpd_f1_metrics.append(
                                ids=[ids[b]],
                                predict=[pred_text.split()],
                                canonical=[canonical[b]] if canonical else [""],
                                perceived=[perceived[b]] if perceived else [""],
                                predict_len=None,
                                canonical_len=None,
                                perceived_len=None,
                                ind2lab=lambda x: x
                            )

            elif stage == sb.Stage.TEST:
                # Autoregressive Generation Evaluation
                canonical = batch.phn_list_canonical
                perceived = batch.phn_list_perceived
                
                if isinstance(ce_targets, dict) and "generated_ids" in ce_targets:
                    gen_ids = ce_targets["generated_ids"]
                    target_to_generate = ce_targets.get("target_to_generate", ["word", "canonical", "target"])
                    
                    # Decode Hypotheses (Generated) using token ID level splitting for reliability
                    # Split by SEP_TGT_ID and EOS_ID positions to extract: words | canonical | target
                    
                    hyps_wrd = []
                    hyps_can = []
                    hyps_tgt = []
                    
                    for b_idx in range(gen_ids.shape[0]):
                        gen_token_seq = gen_ids[b_idx]  # [seq_len]
                        
                        # Find positions of SEP_TGT_ID and EOS_ID
                        sep_tgt_positions = (gen_token_seq == self.SEP_TGT_ID).nonzero(as_tuple=True)[0].tolist()
                        eos_positions = (gen_token_seq == self.EOS_ID).nonzero(as_tuple=True)[0].tolist()
                        
                        # Extract segments based on separator positions
                        # Expected structure: [word_tokens] [SEP_TGT] [can_tokens] [SEP_TGT] [tgt_tokens] [EOS]
                        
                        if len(sep_tgt_positions) >= 2 and len(eos_positions) > 0:
                            # Valid chain with both SEP_TGT separators and EOS
                            sep_tgt_1_idx = sep_tgt_positions[0]
                            sep_tgt_2_idx = sep_tgt_positions[1]
                            eos_idx = eos_positions[0]
                            
                            # Extract word tokens: [0, sep_tgt_1_idx)
                            wrd_tokens = gen_token_seq[0:sep_tgt_1_idx]
                            # Extract canonical tokens: [sep_tgt_1_idx+1, sep_tgt_2_idx)
                            can_tokens = gen_token_seq[sep_tgt_1_idx + 1:sep_tgt_2_idx]
                            # Extract target tokens: [sep_tgt_2_idx+1, eos_idx)
                            tgt_tokens = gen_token_seq[sep_tgt_2_idx + 1:eos_idx]
                            
                            # Decode and clean up
                            hyps_wrd.append(self.hparams.LLM_tokenizer.decode(wrd_tokens, skip_special_tokens=True).strip())
                            hyps_can.append(self.hparams.LLM_tokenizer.decode(can_tokens, skip_special_tokens=True).strip())
                            hyps_tgt.append(self.hparams.LLM_tokenizer.decode(tgt_tokens, skip_special_tokens=True).strip())
                        else:
                            # Chain incomplete: missing SEP_TGT or EOS
                            # Log warning and use placeholder
                            logger.warning(
                                f"[TEST] Incomplete chain at batch {b_idx}: "
                                f"SEP_TGT_count={len(sep_tgt_positions)}, EOS_count={len(eos_positions)}"
                            )
                            hyps_wrd.append("")
                            hyps_can.append("")
                            hyps_tgt.append("")
                    
                    # Decode References
                    refs_tgt = phn_list_to_seq(batch.phn_list_target)
                    refs_can = phn_list_to_seq(batch.phn_list_canonical)
                    refs_wrd = phn_list_to_seq(batch.wrd)
                    
                    # Compute per-target PER scores
                    if "word" in target_to_generate:
                        self.wrd_per_metrics.append(
                            ids=ids,
                            predict=[hyp.split() for hyp in hyps_wrd],
                            target=[ref.split() for ref in refs_wrd],
                            predict_len=None,
                            target_len=None,
                            ind2lab=lambda x: x
                        )
                    
                    if "canonical" in target_to_generate:
                        self.can_per_metrics.append(
                            ids=ids,
                            predict=[hyp.split() for hyp in hyps_can],
                            target=[ref.split() for ref in refs_can],
                            predict_len=None,
                            target_len=None,
                            ind2lab=lambda x: x
                        )
                    
                    if "target" in target_to_generate:
                        self.tgt_per_metrics.append(
                            ids=ids,
                            predict=[hyp.split() for hyp in hyps_tgt],
                            target=[ref.split() for ref in refs_tgt],
                            predict_len=None,
                            target_len=None,
                            ind2lab=lambda x: x
                        )
                    
                    # Overall PER (on target phonemes)
                    self.llm_per_metrics.append(
                        ids=ids,
                        predict=[hyp.split() for hyp in hyps_tgt],
                        target=[ref.split() for ref in refs_tgt],
                        predict_len=None,
                        target_len=None,
                        ind2lab=lambda x: x
                    )
                    
                    # MPD F1 (on target phonemes)
                    self.mpd_f1_metrics.append(
                        ids=ids,
                        predict=[hyp.split() for hyp in hyps_tgt],
                        canonical=canonical,
                        perceived=perceived,
                        predict_len=None,
                        canonical_len=None,
                        perceived_len=None,
                        ind2lab=lambda x: x
                    )

        # ===== Combined Loss =====
        ctc_weight = getattr(self.hparams, "ctc_weight", 0.3)
        loss = ctc_weight * loss_ctc + (1.0 - ctc_weight) * loss_ce
        
        # ===== Logging =====
        if stage == sb.Stage.TRAIN:
            self.train_stats.setdefault("ctc_loss", []).append(float(loss_ctc.detach().cpu()))
            self.train_stats.setdefault("ce_loss", []).append(float(loss_ce.detach().cpu()))
            self.train_stats.setdefault("total_loss", []).append(float(loss.detach().cpu()))
            
            self.multi_target_loss_stats.setdefault("loss_wrd", []).append(float(loss_wrd.detach().cpu()))
            self.multi_target_loss_stats.setdefault("loss_can", []).append(float(loss_can.detach().cpu()))
            self.multi_target_loss_stats.setdefault("loss_tgt", []).append(float(loss_tgt.detach().cpu()))
            self.multi_target_loss_stats.setdefault("loss_llm", []).append(float(loss_ce.detach().cpu()))
        
        return loss
    
    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        predictions = self.compute_forward(batch, stage=stage)
        loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch):
        "Gets called when a stage (either training, validation, test) starts."
        self.ctc_metrics = self.hparams.ctc_stats()
        self.llm_metrics = self.hparams.llm_stats()
        self.mpd_f1_metrics = MpdStats()
        
        if hasattr(self.modules, "ctc_lin"):
            self.ctc_metrics = self.hparams.ctc_stats()

        if stage != sb.Stage.TRAIN:
            self.per_metrics = self.hparams.per_stats()
            self.llm_per_metrics = self.hparams.per_stats()
            self.wrd_per_metrics = self.hparams.per_stats()
            self.can_per_metrics = self.hparams.per_stats()
            self.tgt_per_metrics = self.hparams.per_stats()
            self.mpd_f1_metrics = MpdStats()
            
    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a stage to summarize and log."""
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        elif stage == sb.Stage.VALID:
            stage_stats["epoch"] = epoch
            per_ctc = self.per_metrics.summarize("error_rate")
            per_llm = self.llm_per_metrics.summarize("error_rate")
            per_wrd = self.wrd_per_metrics.summarize("error_rate")
            per_can = self.can_per_metrics.summarize("error_rate")
            per_tgt = self.tgt_per_metrics.summarize("error_rate")
            per_avg = (per_wrd + per_can + per_tgt) / 3.0
            
            stage_stats["ctc_per"] = per_ctc
            stage_stats["llm_per"] = per_llm
            stage_stats["wrd_per"] = per_wrd
            stage_stats["can_per"] = per_can
            stage_stats["tgt_per"] = per_tgt
            stage_stats["per_avg"] = per_avg
            
            llm_loss = self.llm_metrics.summarize("average")
            stage_stats["llm_loss"] = llm_loss
            
            mpd_f1 = self.mpd_f1_metrics.summarize("mpd_f1")
            stage_stats["llm_mpd_f1"] = mpd_f1
        
            if hasattr(self.modules, "ctc_lin"):
                ctc_loss = self.ctc_metrics.summarize("average")
                stage_stats["ctc_loss"] = ctc_loss
                
        if stage == sb.Stage.VALID:
            self.hparams.train_logger.log_stats(
                stats_meta=stage_stats,
            )
            
            if epoch % self.hparams.valid_search_interval == 0:
                improved = False
                ckpt_name = f"epoch{epoch:03d}_CTC_PER{per_ctc:.4f}_WRD_PER{per_wrd:.4f}_CAN_PER{per_can:.4f}_TGT_PER{per_tgt:.4f}_LLM_MPD_F1_{mpd_f1:.4f}.ckpt"
                self.checkpointer.save_and_keep_only(
                    meta={
                        "CTC_PER": per_ctc,
                        "LLM_PER": per_llm,
                        "WRD_PER": per_wrd,
                        "CAN_PER": per_can,
                        "TGT_PER": per_tgt,
                        "PER_AVG": per_avg,
                        "LLM_MPD_F1": mpd_f1
                    },
                    name=ckpt_name,
                    num_to_keep=2,
                    importance_keys=[
                        lambda ckpt: (
                            -ckpt.meta["PER_AVG"],
                            -ckpt.meta["TGT_PER"],
                            -ckpt.meta["LLM_MPD_F1"]
                        )
                    ]
                )
                
                if stage_loss < self.best_valid_loss or len(self.best_valid_loss_list) < 10:
                    ckpt_name = f"best_valid_loss_{epoch:03d}_{stage_loss:.4f}.ckpt"
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
                f"{stage.name.lower()}_wrd_per": per_wrd,
                f"{stage.name.lower()}_can_per": per_can,
                f"{stage.name.lower()}_tgt_per": per_tgt,
                f"{stage.name.lower()}_per_avg": per_avg,
                f"{stage.name.lower()}_llm_loss": llm_loss,
                f"{stage.name.lower()}_ctc_loss": ctc_loss if hasattr(self.modules, "ctc_lin") else None,
                f"{stage.name.lower()}_llm_mpd_f1": mpd_f1 if hasattr(self, "mpd_f1_metrics") else None,
            }, step=epoch)
            
            if self.no_improve_epochs >= self.patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {self.patience} epochs)")
                raise StopIteration
        
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta=stage_stats,
            )
            per_ctc = self.per_metrics.summarize("error_rate")
            mpd_f1 = self.mpd_f1_metrics.summarize("mpd_f1")
            
            per_llm = None
            per_wrd = None
            per_can = None
            per_tgt = None
            per_avg = None
            llm_loss = None
            
            try:
                per_llm = self.llm_per_metrics.summarize("error_rate")
                per_wrd = self.wrd_per_metrics.summarize("error_rate")
                per_can = self.can_per_metrics.summarize("error_rate")
                per_tgt = self.tgt_per_metrics.summarize("error_rate")
                per_avg = (per_wrd + per_can + per_tgt) / 3.0
            except (ZeroDivisionError, ValueError, IndexError, RuntimeError) as e:
                print(f"[Warning] Multi-target PER metrics error: {type(e).__name__}: {e}")
            
            try:
                llm_loss = self.llm_metrics.summarize("average")
            except (ZeroDivisionError, ValueError, IndexError, RuntimeError) as e:
                print(f"[Warning] LLM loss metrics are empty: {type(e).__name__}: {e}")
            
            ctc_loss = self.ctc_metrics.summarize("average")
            
            test_stats = {"loss": stage_loss, "PER_CTC": per_ctc}
            if per_llm is not None:
                test_stats["PER_LLM"] = per_llm
            if per_wrd is not None:
                test_stats["PER_WRD"] = per_wrd
            if per_can is not None:
                test_stats["PER_CAN"] = per_can
            if per_tgt is not None:
                test_stats["PER_TGT"] = per_tgt
            if per_avg is not None:
                test_stats["PER_AVG"] = per_avg
            if llm_loss is not None:
                test_stats["llm_loss"] = llm_loss
            if hasattr(self.modules, "ctc_lin"):
                test_stats["ctc_loss"] = ctc_loss
            if hasattr(self, "mpd_f1_metrics"):
                test_stats["llm_mpd_f1"] = mpd_f1
            
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=test_stats,
            )
            
            with open(self.hparams.per_file, "w") as w:
                w.write("CTC loss stats:\n")
                self.ctc_metrics.write_stats(w)
                w.write("\nCTC PER stats:\n")
                self.per_metrics.write_stats(w)
            
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
                    if per_wrd is not None:
                        w.write("\nWRD PER stats:\n")
                        self.wrd_per_metrics.write_stats(w)
                    if per_can is not None:
                        w.write("\nCAN PER stats:\n")
                        self.can_per_metrics.write_stats(w)
                    if per_tgt is not None:
                        w.write("\nTGT PER stats:\n")
                        self.tgt_per_metrics.write_stats(w)
            
                if not hasattr(self.hparams, 'mpd_seq_file'):
                    self.hparams.mpd_seq_file = self.hparams.mpd_file.replace('.txt', '_llm.txt')

                with open(self.hparams.mpd_seq_file, "w") as m:
                    m.write("MPD results and stats:\n")
                    self.mpd_f1_metrics.write_stats(m)
                    print("MPD results and stats written to file", self.hparams.mpd_seq_file)
                    
    def check_gradients(self, loss):
        """Check if gradients are finite"""
        if not torch.isfinite(loss):
            print("Warning: loss is not finite, skipping step")
            return False
        return True

    def fit_batch(self, batch):
        """Fit one batch, override to do multiple updates."""
        if self.hparams.auto_mix_prec:
            self.adam_optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)

            self.scaler.scale(loss / self.hparams.gradient_accumulation).backward()
            self.scaler.unscale_(self.adam_optimizer)

            if self.check_gradients(loss):
                if any(p.requires_grad for p in self.adam_optimizer.param_groups[0]['params']):
                    self.scaler.step(self.adam_optimizer)

            self.scaler.update()

        else:
            outputs = self.compute_forward(batch, sb.Stage.TRAIN)
            loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            (loss / self.hparams.gradient_accumulation).backward()

            if self.step % self.hparams.gradient_accumulation == 0:
                if self.check_gradients(loss):
                    self.pretrained_opt_class.step()
                    self.adam_optimizer.step()

                self.pretrained_opt_class.zero_grad()
                self.adam_optimizer.zero_grad()    

        return loss.detach().cpu()

    @torch.no_grad()
    def inference_batch(self, batch, max_new_tokens=100, do_sample=False, temperature=1.0, 
                        top_k=50, top_p=0.9, num_beams=1):
        """
        Pure inference when batch only contains id and sig (no target phonemes).
        Generates words, canonical, and target phonemes separately.
        
        Args:
            batch: A batch containing:
                - batch.id: list of utterance IDs
                - batch.sig: tuple of (wavs [B, T], wav_lens [B])
            max_new_tokens: Maximum number of tokens to generate
            do_sample: Whether to use sampling (True) or greedy decoding (False)
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            num_beams: Number of beams for beam search
            
        Returns:
            dict: {
                "ids": list of utterance IDs,
                "generated_tokens": tensor of generated token IDs [B, gen_len],
                "words": list of generated word strings,
                "canonical": list of generated canonical phoneme strings,
                "target": list of generated target phoneme strings,
                "ctc_predictions": list of CTC decoded phoneme sequences (if available)
            }
        """
        # Ensure critical components are initialized
        self._ensure_initialized()
        
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        ids = batch.id
        tok = self.hparams.LLM_tokenizer
        
        # AudioEncoder + Projector
        wav_feats = self.modules.perceived_ssl(wavs)
        
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
            ctc_predictions = []
            for seq in ctc_sequence:
                phn_list = self.label_encoder.decode_ndim(seq)
                ctc_predictions.append(" ".join(phn_list))
        
        # Projector
        if hasattr(self.modules, "projector") and self.modules.projector is not None:
            Z = self.modules.projector(Z)
        
        B, Ts, H = Z.shape
        device = self.device
        embed_fn = self.modules.LLM.get_input_embeddings()
        
        # Get LLM dtype
        llm_dtype = embed_fn.weight.dtype
        
        # Prepare special tokens
        BOS_ID = tok.bos_token_id
        EOS_ID = tok.eos_token_id
        PAD_ID = tok.pad_token_id if tok.pad_token_id is not None else 0
        
        # Use newline as separator token (simpler than special tokens)
        newline_tokens = tok("\n", return_tensors="pt", add_special_tokens=False).to(device)
        SEP_ID = newline_tokens["input_ids"][0, 0].item()
        SEP_TGT_ID = SEP_ID
        
        # Store for TEST parsing
        self.SEP_TGT_ID = SEP_TGT_ID
        self.EOS_ID = EOS_ID
        self.BOS_ID = BOS_ID
        self.PAD_ID = PAD_ID
        
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
        
        if has_prompt:
            prompt_embed = self.prompt_embed
            prompt_len = prompt_embed.size(0)
            prompt_embed_batch = prompt_embed.unsqueeze(0).expand(B, -1, -1)
        else:
            prompt_embed_batch = None
        
        # Build input embeddings for inference: [SEP][speech][BOS]
        if has_split_prompt:
            inputs_embeds = torch.cat([
                self.prompt_prefix_embed.unsqueeze(0).expand(B, -1, -1),
                Z,
                self.prompt_suffix_embed.unsqueeze(0).expand(B, -1, -1)
            ], dim=1)
        elif prompt_embed_batch is not None:
            inputs_embeds = torch.cat([
                prompt_embed_batch,
                SEP_embed,
                Z,
                BOS_embed
            ], dim=1)
        else:
            inputs_embeds = torch.cat([
                SEP_embed,
                Z,
                BOS_embed
            ], dim=1)
        
        # Match LLM dtype
        if inputs_embeds.dtype != llm_dtype:
            inputs_embeds = inputs_embeds.to(llm_dtype)
        
        attention_mask = torch.ones(B, inputs_embeds.size(1), dtype=torch.long, device=device)
        
        # Get target_to_generate
        target_to_generate = getattr(self.hparams, "target_to_generate", ["word", "canonical", "target"])
        
        # Calculate max_new_tokens (conservative estimate)
        max_gen_len = max_new_tokens
        
        # Generate tokens
        gen_kwargs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "max_new_tokens": max_gen_len,
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
        
        # Parse generated tokens by SEP_TGT and EOS markers
        # Expected structure: [words][SEP_TGT][canonical][SEP_TGT][target][EOS]
        words_list = []
        canonical_list = []
        target_list = []
        
        for b_idx in range(gen_out.shape[0]):
            gen_token_seq = gen_out[b_idx]
            
            # Find positions of SEP_TGT_ID and EOS_ID
            sep_tgt_positions = (gen_token_seq == SEP_TGT_ID).nonzero(as_tuple=True)[0].tolist()
            eos_positions = (gen_token_seq == EOS_ID).nonzero(as_tuple=True)[0].tolist()
            
            # Extract segments based on separator positions
            if len(sep_tgt_positions) >= 2 and len(eos_positions) > 0:
                sep_tgt_1_idx = sep_tgt_positions[0]
                sep_tgt_2_idx = sep_tgt_positions[1]
                eos_idx = eos_positions[0]
                
                # Extract words, canonical, target
                wrd_tokens = gen_token_seq[0:sep_tgt_1_idx]
                can_tokens = gen_token_seq[sep_tgt_1_idx + 1:sep_tgt_2_idx]
                tgt_tokens = gen_token_seq[sep_tgt_2_idx + 1:eos_idx]
                
                words_list.append(tok.decode(wrd_tokens, skip_special_tokens=True).strip())
                canonical_list.append(tok.decode(can_tokens, skip_special_tokens=True).strip())
                target_list.append(tok.decode(tgt_tokens, skip_special_tokens=True).strip())
            else:
                # Chain incomplete
                logger.warning(f"[Inference] Incomplete chain at batch {b_idx}")
                words_list.append("")
                canonical_list.append("")
                target_list.append("")
        
        return {
            "ids": ids,
            "generated_tokens": gen_out,
            "words": words_list,
            "canonical": canonical_list,
            "target": target_list,
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
        Iterate test_set and perform multi-target inference (no loss computation).
        Generates words, canonical phonemes, and target phonemes separately.

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
            Kwargs passed to ``make_dataloader()`` if ``test_set`` is not a DataLoader.
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
            Optional file path to save results (base name; will generate separate files 
            for words, canonical, target, and CTC predictions).

        Returns
        -------
        list of dict: All inference results with 'id', 'words', 'canonical', 'target', 'ctc_predictions'
        """
        import csv
        import json
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
                    num_beams=1,
                )
                
                # Collect results
                for i, utt_id in enumerate(result["ids"]):
                    item = {
                        "id": utt_id,
                        "words": result["words"][i],
                        "canonical": result["canonical"][i],
                        "target": result["target"][i],
                    }
                    if result["ctc_predictions"] is not None:
                        item["ctc_prediction"] = result["ctc_predictions"][i]
                    all_results.append(item)
        
        # Sort results by ID
        all_results = sorted(all_results, key=lambda x: str(x["id"]))
        
        # Optionally save to file
        if output_file is not None:
            import os
            
            output_dir = os.path.dirname(output_file) if os.path.dirname(output_file) else "."
            base_name = os.path.splitext(os.path.basename(output_file))[0]
            
            # Save Words predictions to CSV
            words_csv_path = os.path.join(output_dir, f"{base_name}_words.csv")
            with open(words_csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["ID", "Words"])
                for item in all_results:
                    file_stem = os.path.splitext(os.path.basename(str(item["id"])))[0]
                    writer.writerow([file_stem, item["words"]])
            print(f"Words predictions saved to {words_csv_path}")
            
            # Save Canonical predictions to CSV
            can_csv_path = os.path.join(output_dir, f"{base_name}_canonical.csv")
            with open(can_csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["ID", "Canonical"])
                for item in all_results:
                    file_stem = os.path.splitext(os.path.basename(str(item["id"])))[0]
                    writer.writerow([file_stem, item["canonical"]])
            print(f"Canonical predictions saved to {can_csv_path}")
            
            # Save Target predictions to CSV
            tgt_csv_path = os.path.join(output_dir, f"{base_name}_target.csv")
            with open(tgt_csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["ID", "Target"])
                for item in all_results:
                    file_stem = os.path.splitext(os.path.basename(str(item["id"])))[0]
                    writer.writerow([file_stem, item["target"]])
            print(f"Target predictions saved to {tgt_csv_path}")
            
            # Save CTC predictions to CSV (if available)
            if all_results and "ctc_prediction" in all_results[0]:
                ctc_csv_path = os.path.join(output_dir, f"{base_name}_ctc.csv")
                with open(ctc_csv_path, "w", encoding="utf-8", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["ID", "CTC"])
                    for item in all_results:
                        file_stem = os.path.splitext(os.path.basename(str(item["id"])))[0]
                        writer.writerow([file_stem, item["ctc_prediction"]])
                print(f"CTC predictions saved to {ctc_csv_path}")
            
            # Also save JSONL for full details
            jsonl_path = os.path.join(output_dir, f"{base_name}_multitarget.jsonl")
            with open(jsonl_path, "w", encoding="utf-8") as f:
                for item in all_results:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"Full results saved to {jsonl_path}")
        
        return all_results

    def on_fit_start(self):
        """Gets called at the beginning of ``fit()``."""
        self._compile()
        self._wrap_distributed()
        self.init_optimizers()

        if self.checkpointer is not None:
            self.checkpointer.recover_if_possible(min_key="PER_AVG")
            
        if getattr(self.hparams, 'load_pretrained_components', False):
            pretrained_path = getattr(self.hparams, 'pretrained_model_path', '')
            components = getattr(self.hparams, 'components_to_load', ['ssl'])
            freeze_loaded = getattr(self.hparams, 'freeze_loaded_components', True)
        
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
        model_params = list(self.hparams.model.parameters())
        trainable_model_params = list(self.hparams.trainable_model.parameters())
        
        if hasattr(self, "prompt_embed") and self.prompt_embed is not None:
            if isinstance(self.prompt_embed, torch.nn.Parameter):
                model_params.append(self.prompt_embed)
                print(f"[Optimizer] Added soft prompt embeddings to optimizer")

        self.adam_optimizer = self.hparams.adam_opt_class(
            trainable_model_params, 
        )
        
        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("adam_opt", self.adam_optimizer)
            self.checkpointer.add_recoverable("tokenizer", self.label_encoder)
