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

import sys
sys.path.append("/work/gm64/m64000/IF-MDD")
from debug_input_embedding import InputEmbeddingDebugger

logger = sb.utils.logger.get_logger(__name__)


class PeftAdapterRecoverable:
    """Wrapper to save ONLY the adapter weights of a PeftModel in SpeechBrain checkpoints.
    
    This wrapper handles loading state_dicts with extra quantization keys gracefully
    by using set_peft_model_state_dict instead of strict load_state_dict.
    """
    def __init__(self, model):
        self.model = model
    
    def state_dict(self):
        return get_peft_model_state_dict(self.model)
    
    def load_state_dict(self, state_dict):
        # Use PEFT's set_peft_model_state_dict which handles mismatches gracefully
        set_peft_model_state_dict(self.model, state_dict)
    
    def to(self, device):
        self.model.to(device)


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


class JSONPromptBuilder:
    """
    构建JSON格式的Prompt，用于结构化输出多个目标的预测.
    
    使用方式:
        builder = JSONPromptBuilder(
            use_json=True,
            placeholder="[AUDIO]",
            targets=["transcription", "canonical_phonemes", "target_phonemes"]
        )
        prompt = builder.build_prompt()
        # 输出:
        # [AUDIO]
        # Analyze the audio and output the result in JSON format with the following keys:
        # {
        #     "transcription": "arabic words",
        #     "canonical_phonemes": "phoneme sequence",
        #     "target_phonemes": "phoneme sequence"
        # }
        # Output JSON only.
    """
    
    def __init__(self, use_json=True, placeholder="[AUDIO]", targets=None, language="Arabic"):
        self.use_json = use_json
        self.placeholder = placeholder
        self.targets = targets or ["transcription", "canonical_phonemes", "target_phonemes"]
        self.language = language
    
    def build_prompt(self):
        """构建完整的Prompt"""
        if not self.use_json:
            return self.placeholder
        
        json_template = self._build_json_template()
        
        prompt = f"""{self.placeholder}
Analyze the audio and output the result in JSON format with the following keys:
{json_template}
Output JSON only. Do not include any explanation or additional text."""
        
        return prompt
    
    def _build_json_template(self):
        """构建JSON模板"""
        json_dict = {}
        descriptions = {
            "transcription": f"{self.language} words or letters",
            "canonical_phonemes": "canonical phoneme sequence (space-separated)",
            "target_phonemes": "target phoneme sequence (space-separated)",
            "word": f"{self.language} words or letters",
            "canonical": "canonical phoneme sequence (space-separated)",
            "target": "target phoneme sequence (space-separated)",
        }
        
        for target in self.targets:
            desc = descriptions.get(target, "output sequence")
            json_dict[target] = desc
        
        # 格式化为JSON字符串，带缩进
        import json
        json_str = json.dumps(json_dict, indent=4, ensure_ascii=False)
        return json_str
    
    def parse_json_output(self, text):
        """
        解析LLM生成的JSON文本，提取各个目标的预测.
        
        Args:
            text: LLM生成的文本
            
        Returns:
            dict: 解析后的预测结果，如 {
                "transcription": "...",
                "canonical_phonemes": "...",
                "target_phonemes": "..."
            }
        """
        import json
        import re
        
        text = text.strip()
        
        # 尝试找到JSON部分
        try:
            # 方法1：直接解析整个文本
            result = json.loads(text)
            return self._normalize_keys(result)
        except json.JSONDecodeError:
            pass
        
        # 方法2：查找 { 和 } 之间的内容
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group())
                return self._normalize_keys(result)
            except json.JSONDecodeError:
                pass
        
        # 方法3：尝试修复常见的JSON错误
        try:
            # 删除末尾可能的非JSON文本
            text = re.sub(r'\}[\s\S]*$', '}', text)
            result = json.loads(text)
            return self._normalize_keys(result)
        except json.JSONDecodeError:
            pass
        
        # 失败时返回空结果
        logger.warning(f"Failed to parse JSON from LLM output: {text[:100]}")
        return {target: "" for target in self.targets}
    
    def _normalize_keys(self, json_dict):
        """
        规范化JSON键名，处理不同的命名方式.
        例: "canonical_phonemes" -> 也接受 "canonical", "canonical phonemes"
        """
        result = {}
        
        # 键名映射表
        key_mapping = {
            "transcription": ["transcription", "word", "words", "arabic", "text"],
            "canonical_phonemes": ["canonical_phonemes", "canonical", "canonical phonemes", "canonical_phoneme"],
            "target_phonemes": ["target_phonemes", "target", "target phonemes", "target_phoneme"],
        }
        
        # 生成反向映射
        reverse_mapping = {}
        for standard_key, variants in key_mapping.items():
            for variant in variants:
                reverse_mapping[variant.lower()] = standard_key
        
        # 规范化输入的键
        for key, value in json_dict.items():
            normalized_key = reverse_mapping.get(key.lower(), key)
            result[normalized_key] = value if isinstance(value, str) else str(value)
        
        return result
    
    def clean_prediction(self, prediction_text):
        """清理预测文本，移除不需要的前缀和后缀"""
        prediction_text = prediction_text.strip()
        
        # 移除常见的前缀
        unwanted_prefixes = [
            "transcription:",
            "canonical_phonemes:",
            "canonical phonemes:",
            "target_phonemes:",
            "target phonemes:",
            "phoneme sequence:",
            "sequence:",
            "output:",
        ]
        
        for prefix in unwanted_prefixes:
            if prediction_text.lower().startswith(prefix):
                prediction_text = prediction_text[len(prefix):].strip()
        
        return prediction_text


class ChatPromptBuilder:
    """
    构建Chat格式的Prompt，用于Chat模型（如ChatGPT, Llama-chat等）。
    输出格式为三行（用换行符分隔）：
    1. Arabic transcription
    2. Canonical phoneme sequence
    3. Target phoneme sequence
    
    使用方式:
        builder = ChatPromptBuilder(
            placeholder="[AUDIO]",
            format_type="newline"  # "newline" 或 "structured"
        )
        chat_structure = builder.build_chat_messages()
        # 输出: [{role: "system", content: "..."}, {role: "user", content: "..."}]
    """
    
    def __init__(self, placeholder="[AUDIO]", format_type="newline", language="Arabic"):
        """
        Args:
            placeholder: 代表音频的占位符
            format_type: 输出格式类型
                - "newline": 三行格式（用换行符分隔）
                - "structured": 结构化格式（带标签）
                - "json": JSON格式（改用JSONPromptBuilder）
            language: 目标语言
        """
        self.placeholder = placeholder
        self.format_type = format_type
        self.language = language
    
    def build_chat_messages(self):
        """
        构建Chat格式的消息列表，用于Chat模型。
        
        Returns:
            list: [{role: "system", content: "..."}, {role: "user", content: "..."}]
        """
        system_message = self._build_system_message()
        user_message = self._build_user_message()
        
        return [
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user",
                "content": user_message
            }
        ]
    
    def _build_system_message(self):
        """构建系统消息"""
        if self.format_type == "newline":
            return f"""You are an {self.language} language expert. You are strict about output format.
You will receive an audio embedding sequence.

You must output EXACTLY three lines with NO EXTRA TEXT:
1. The {self.language} transcription
2. The canonical phoneme sequence (space-separated)
3. The target phoneme sequence (space-separated)

Example output format:
السلام عليكم
s a l a a m a a l i k u m
s a l a a m a a l i k u m

CONSTRAINTS:
- NO labels (don't write "Transcription:", "Canonical:", etc.)
- NO explanations
- NO additional text
- ONLY the three sequences separated by newlines
"""
        
        elif self.format_type == "structured":
            return f"""You are an {self.language} language expert and phonetic annotation specialist.
You will analyze audio embeddings and provide three specific outputs.

Output format:
Line 1: Transcription: <{self.language} text>
Line 2: Canonical: <space-separated phonemes>
Line 3: Target: <space-separated phonemes>

Be concise and follow the exact format above."""
        
        else:
            raise ValueError(f"Unknown format_type: {self.format_type}")
    
    def _build_user_message(self):
        """构建用户消息"""
        if self.format_type == "newline":
            return f"""{self.placeholder}

Transcribe this audio to {self.language} text and provide both canonical and target phoneme sequences.
Output only the three sequences as specified, no explanations."""
        
        elif self.format_type == "structured":
            return f"""{self.placeholder}

Please provide the transcription, canonical phoneme sequence, and target phoneme sequence for this audio."""
        
        else:
            raise ValueError(f"Unknown format_type: {self.format_type}")
    
    def parse_chat_output(self, text):
        """
        解析Chat格式的LLM输出（三行格式）。
        
        Args:
            text: LLM生成的文本
            
        Returns:
            dict: {
                "transcription": "...",
                "canonical_phonemes": "...",
                "target_phonemes": "..."
            }
        """
        lines = text.strip().split('\n')
        
        # 移除空行
        lines = [line.strip() for line in lines if line.strip()]
        
        result = {
            "transcription": "",
            "canonical_phonemes": "",
            "target_phonemes": ""
        }
        
        if len(lines) >= 3:
            # 三行格式
            result["transcription"] = self._clean_line(lines[0])
            result["canonical_phonemes"] = self._clean_line(lines[1])
            result["target_phonemes"] = self._clean_line(lines[2])
        elif len(lines) == 2:
            # 两行（可能缺少一个）
            result["transcription"] = self._clean_line(lines[0])
            result["canonical_phonemes"] = self._clean_line(lines[1])
            logger.warning(f"Only 2 lines parsed, target_phonemes is empty")
        elif len(lines) == 1:
            # 一行（只有transcription）
            result["transcription"] = self._clean_line(lines[0])
            logger.warning(f"Only 1 line parsed, phoneme sequences are empty")
        else:
            logger.warning(f"No valid lines found in output: {text[:100]}")
        
        return result
    
    def _clean_line(self, line):
        """清理单行文本"""
        line = line.strip()
        
        # 移除标签（如果有结构化格式）
        labels = [
            "Transcription:",
            "transcription:",
            "Canonical:",
            "canonical:",
            "Target:",
            "target:",
            "Canonical phoneme:",
            "Target phoneme:",
        ]
        
        for label in labels:
            if line.lower().startswith(label.lower()):
                line = line[len(label):].strip()
        
        # 移除其他前缀
        if line.startswith("-"):
            line = line[1:].strip()
        if line.startswith(":"):
            line = line[1:].strip()
        
        return line


class SSL_LLM_MultiTarget_ver2(sb.Brain):
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
        self.debugger = InputEmbeddingDebugger(
            brain=self,
            tokenizer=self.hparams.LLM_tokenizer,
            output_dir=self.hparams.save_folder + "/debug_outputs"
        )
        
        
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
                        "content": """You are an arabic expert. You strictly follow the output format constraints.
                                    You will receive an audio embedding sequence.
                                    You must output EXACTLY three lines:
                                    1. The Arabic transcription.
                                    2. The canonical phoneme sequence.
                                    3. The target phoneme sequence.
                                    NO LABELS, NO EXTRA TEXT."""
                    },
                    {
                        "role": "user", 
                        "content": f"""{PLACEHOLDER}
                                    Transcribe this audio... Output raw text only.
                                    """
                        # "content": f"""{PLACEHOLDER} Transcribe this into arabic words and the provide it's canonical phoneme and target phoneme sequence.
                        #             Split the word, canonical phoneme and target phoneme sequences with newlines only.
                        #             Don't add any extra text other than the three sequences.
                        #             """
                    }
                ]

            #     chat_structure = [
            #     {
            #         "role": "system", 
            #         "content": """You are an arabic expert. You strictly follow the output format constraints.
            #                     You will receive an audio embedding sequence.
            #                     You must output EXACTLY three lines:
            #                     1. The Arabic transcription.
            #                     2. The canonical phoneme sequence.
            #                     3. The target phoneme sequence.
            #                     NO LABELS, NO EXTRA TEXT."""
            #     },
            #     # --- 添加 One-Shot 示例 ---
            #     {
            #         "role": "user", 
            #         "content": """(Audio Placeholder)
            #                     Transcribe this audio into words, provide its canonical phoneme sequence, and target phoneme sequence."""
            #     },
            #     {
            #         "role": "assistant",
            #         "content": """بسم الله
            #                     b i s m i l l a h
            #                     b i s m i l l a""" 
            #     },
            #     # --- 真实的输入 ---
            #     {
            #         "role": "user", 
            #         "content": f"""{PLACEHOLDER}
            #                     Transcribe this audio into words, provide its canonical phoneme sequence, and target phoneme sequence.
            #                     Output raw text only, separated by newlines.
            #                     """
            #     }
            # ]

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
                
                self.prefix_ids = prefix_tokens["input_ids"].squeeze(0)
                self.suffix_ids = suffix_tokens["input_ids"].squeeze(0)

                with torch.no_grad():
                    self.prompt_prefix_embed = embed_fn(self.prefix_ids)
                    self.prompt_suffix_embed = embed_fn(self.suffix_ids)
                
                self.prompt_embed = torch.cat([self.prompt_prefix_embed, self.prompt_suffix_embed], dim=0)
                print(f"[Lazy Init] Generated Llama 3 Prompt via Template.")

    def _ensure_initialized_infer(self):
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

                # chat_structure = [
                #     {
                #         "role": "system", 
                #         "content": """You are an arabic expert. You strictly follow the output format constraints.
                #                     You will receive an audio embedding sequence.
                #                     You must output EXACTLY three lines:
                #                     1. The Arabic transcription.
                #                     2. The canonical phoneme sequence.
                #                     3. The target phoneme sequence.
                #                     NO LABELS, NO EXTRA TEXT."""
                #     },
                #     {
                #         "role": "user", 
                #         "content": f"""{PLACEHOLDER}
                #                     Transcribe this audio... Output raw text only.
                #                     """
                #         # "content": f"""{PLACEHOLDER} Transcribe this into arabic words and the provide it's canonical phoneme and target phoneme sequence.
                #         #             Split the word, canonical phoneme and target phoneme sequences with newlines only.
                #         #             Don't add any extra text other than the three sequences.
                #         #             """
                #     }
                # ]

                chat_structure = [
                {
                    "role": "system", 
                    "content": """You are an arabic expert. You strictly follow the output format constraints.
                                You will receive an audio embedding sequence.
                                You must output EXACTLY three lines:
                                1. The Arabic transcription.
                                2. The canonical phoneme sequence.
                                3. The target phoneme sequence.
                                NO LABELS, NO EXTRA TEXT."""
                },
                # --- 添加 One-Shot 示例 ---
                {
                    "role": "user", 
                    "content": """(Audio Placeholder)
                                Transcribe this audio into words, provide its canonical phoneme sequence, and target phoneme sequence."""
                },
                {
                    "role": "assistant",
                    "content": """بسم الله
                                b i s m i l l a h
                                b i s m i l l a""" 
                },
                # --- 真实的输入 ---
                {
                    "role": "user", 
                    "content": f"""{PLACEHOLDER}
                                Transcribe this audio into words, provide its canonical phoneme sequence, and target phoneme sequence.
                                Output raw text only, separated by newlines.
                                """
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
                
                self.prefix_ids = prefix_tokens["input_ids"].squeeze(0)
                self.suffix_ids = suffix_tokens["input_ids"].squeeze(0)

                with torch.no_grad():
                    self.prompt_prefix_embed = embed_fn(self.prefix_ids)
                    self.prompt_suffix_embed = embed_fn(self.suffix_ids)
                
                self.prompt_embed = torch.cat([self.prompt_prefix_embed, self.prompt_suffix_embed], dim=0)
                print(f"[Lazy Init] Generated Llama 3 Prompt via Template.")

    def _ensure_initialized_infer_target(self):
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
                    "content": """You are an arabic expert. You strictly follow the output format constraints.
                                You will receive an audio embedding sequence.
                                You must output EXACTLY one lines
                                The target phoneme sequence.
                                NO LABELS, NO EXTRA TEXT."""
                },

                {
                    "role": "user", 
                    "content": f"""{PLACEHOLDER}
                                Transcribe this audio into phoneme
                                """
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
                
                self.prefix_ids = prefix_tokens["input_ids"].squeeze(0)
                self.suffix_ids = suffix_tokens["input_ids"].squeeze(0)

                with torch.no_grad():
                    self.prompt_prefix_embed = embed_fn(self.prefix_ids)
                    self.prompt_suffix_embed = embed_fn(self.suffix_ids)
                
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

    def _build_compact_token_ids_flexible(self, tokenized_data, training_targets, 
                                          SEP_TGT_ID, EOS_ID, PAD_ID, device):
        """Build compact multi-target token sequence with flexible target selection.
        
        Args:
            tokenized_data: dict with keys 'wrd', 'can', 'tgt' containing 'ids' and 'mask'
                           (only keys in training_targets will be present)
            training_targets: list of target names, e.g. ["word", "canonical", "target"]
            SEP_TGT_ID, EOS_ID, PAD_ID: special token IDs
            device: torch device
            
        Returns:
            compact_ids: [B, max_text_len] - compact token IDs
            compact_pos_ranges: dict - position ranges for each included target
        """
        if not training_targets:
            raise ValueError("training_targets cannot be empty")
        
        # Get batch size from first available tokenized data
        first_key = list(tokenized_data.keys())[0]
        B = tokenized_data[first_key]["ids"].size(0)
        
        # Map full names to short names
        target_map = {
            "word": "wrd",
            "canonical": "can",
            "target": "tgt"
        }
        
        # Compute actual lengths (without padding) for each target
        actual_lens = {}
        for target_name in training_targets:
            short_name = target_map[target_name]
            if short_name in tokenized_data:
                actual_lens[target_name] = tokenized_data[short_name]["mask"].sum(dim=1).long()  # [B]
        
        # Calculate compact text length per sample
        # Format: [target1_actual][SEP][target2_actual][SEP]...[targetN_actual][EOS]
        num_seps = len(training_targets) - 1  # SEP between targets
        text_lens = sum(actual_lens[t] for t in training_targets) + num_seps + 1  # 1 for EOS
        max_text_len = text_lens.max().item()
        
        # Initialize compact sequence
        compact_ids = torch.full((B, max_text_len), PAD_ID, dtype=torch.long, device=device)
        
        # Build position ranges dict
        compact_pos_ranges = {}
        
        # Build compact sequences per sample
        for b in range(B):
            pos = 0
            
            for i, target_name in enumerate(training_targets):
                short_name = target_map[target_name]
                token_ids = tokenized_data[short_name]["ids"]
                token_mask = tokenized_data[short_name]["mask"]
                n_tokens = token_mask[b].sum().item()
                
                # Record position
                start_pos = pos
                
                if n_tokens > 0:
                    compact_ids[b, pos:pos + n_tokens] = token_ids[b, :n_tokens]
                    pos += n_tokens
                
                end_pos = pos
                
                # Store position info for this target (use short name as key for consistency)
                if short_name not in compact_pos_ranges:
                    compact_pos_ranges[short_name] = {
                        'starts': [],
                        'ends': [],
                        'nums': []
                    }
                
                compact_pos_ranges[short_name]['starts'].append(start_pos)
                compact_pos_ranges[short_name]['ends'].append(end_pos)
                compact_pos_ranges[short_name]['nums'].append(n_tokens)
                
                # Add separator if not last target
                if i < len(training_targets) - 1:
                    compact_ids[b, pos] = SEP_TGT_ID
                    pos += 1
            
            # Add EOS at end
            compact_ids[b, pos] = EOS_ID
        
        # Convert lists to tensors in compact_pos_ranges (use short names)
        target_map_reverse = {"wrd": "word", "can": "canonical", "tgt": "target"}
        for short_name in ["wrd", "can", "tgt"]:
            if short_name in compact_pos_ranges:
                compact_pos_ranges[short_name]['nums'] = torch.tensor(
                    compact_pos_ranges[short_name]['nums'], device=device, dtype=torch.long
                )
        
        # Add metadata
        compact_pos_ranges['text_lens'] = text_lens
        compact_pos_ranges['training_targets'] = training_targets
        
        return compact_ids, compact_pos_ranges

    def _build_compact_token_ids(self, wrd_ids, wrd_mask, phn_can_ids, phn_can_mask, 
                                  phn_tgt_ids, phn_tgt_mask, SEP_TGT_ID, EOS_ID, PAD_ID, device):
        """Build compact multi-target token sequence without internal padding.
        
        Transforms padded sequences into compact format: [wrd_actual][SEP_TGT][can_actual][SEP_TGT][tgt_actual][EOS]
        
        Args:
            wrd_ids: [B, L_wrd] - word token IDs (with padding)
            wrd_mask: [B, L_wrd] - word attention mask
            phn_can_ids: [B, L_can] - canonical phoneme token IDs (with padding)
            phn_can_mask: [B, L_can] - canonical attention mask
            phn_tgt_ids: [B, L_tgt] - target phoneme token IDs (with padding)
            phn_tgt_mask: [B, L_tgt] - target attention mask
            SEP_TGT_ID: int - separator token ID between targets
            EOS_ID: int - end of sequence token ID
            PAD_ID: int - padding token ID
            device: torch.device
            
        Returns:
            compact_ids: [B, max_text_len] - compact token IDs with padding only at end
            compact_pos_ranges: dict - position ranges for each target in compact sequence
                {
                    'wrd': (wrd_start, wrd_end, actual_lens),  # [B] actual lengths
                    'can': (can_start, can_end, actual_lens),
                    'tgt': (tgt_start, tgt_end, actual_lens),
                    'text_lens': [B] - actual text length per sample (excluding padding)
                }
        """
        B = wrd_ids.size(0)
        
        # Extract actual token counts per sample
        num_wrd = wrd_mask.sum(dim=1).long()  # [B]
        num_can = phn_can_mask.sum(dim=1).long()  # [B]
        num_tgt = phn_tgt_mask.sum(dim=1).long()  # [B]
        
        # Calculate compact text length per sample
        # Format: [wrd_actual][SEP_TGT][can_actual][SEP_TGT][tgt_actual][EOS]
        # Length: num_wrd + 1 + num_can + 1 + num_tgt + 1
        text_lens = num_wrd + num_can + num_tgt + 3  # [B]
        max_text_len = text_lens.max().item()
        
        # Initialize compact sequence with padding
        compact_ids = torch.full((B, max_text_len), PAD_ID, dtype=torch.long, device=device)
        
        # Build compact sequences per sample
        wrd_start_list = []
        wrd_end_list = []
        can_start_list = []
        can_end_list = []
        tgt_start_list = []
        tgt_end_list = []
        
        for b in range(B):
            n_wrd = num_wrd[b].item()
            n_can = num_can[b].item()
            n_tgt = num_tgt[b].item()
            
            pos = 0
            
            # Word tokens
            wrd_start = pos
            if n_wrd > 0:
                compact_ids[b, pos:pos + n_wrd] = wrd_ids[b, :n_wrd]
                pos += n_wrd
            wrd_end = pos
            wrd_start_list.append(wrd_start)
            wrd_end_list.append(wrd_end)
            
            # SEP_TGT separator
            compact_ids[b, pos] = SEP_TGT_ID
            pos += 1
            
            # Canonical phoneme tokens
            can_start = pos
            if n_can > 0:
                compact_ids[b, pos:pos + n_can] = phn_can_ids[b, :n_can]
                pos += n_can
            can_end = pos
            can_start_list.append(can_start)
            can_end_list.append(can_end)
            
            # SEP_TGT separator
            compact_ids[b, pos] = SEP_TGT_ID
            pos += 1
            
            # Target phoneme tokens
            tgt_start = pos
            if n_tgt > 0:
                compact_ids[b, pos:pos + n_tgt] = phn_tgt_ids[b, :n_tgt]
                pos += n_tgt
            tgt_end = pos
            tgt_start_list.append(tgt_start)
            tgt_end_list.append(tgt_end)
            
            # EOS token
            compact_ids[b, pos] = EOS_ID
            pos += 1
            
            # Remaining positions are already PAD_ID
        
        # Build position ranges dict
        compact_pos_ranges = {
            'wrd': (wrd_start_list, wrd_end_list, num_wrd),
            'can': (can_start_list, can_end_list, num_can),
            'tgt': (tgt_start_list, tgt_end_list, num_tgt),
            'text_lens': text_lens
        }
        
        return compact_ids, compact_pos_ranges

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
        if stage != sb.Stage.TEST:
            self._ensure_initialized()
        else:
            # self._ensure_initialized_infer()
            self._ensure_initialized_infer_target()
        
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        
        # Decode References
        if getattr(self.hparams, "target_type")  == "target":
            phn_list_training_target = batch.phn_list_target
        elif getattr(self.hparams, "target_type")  == "perceived":
            phn_list_training_target = batch.phn_list_perceived
        else:
            logger.warning(f"Unknown target_type: {getattr(self.hparams, 'target_type')}, defaulting to 'target'")
            phn_list_training_target = batch.phn_list_target
            
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "augmentation"):
            wavs = self.hparams.augmentation(wavs)
        
        # Audio Encoder
        try:
            Z, _ = self.hparams.audio_encoder_modules(wavs)
        except:
            Z = self.hparams.audio_encoder_modules(wavs)
        
        # CTC branch
        ctc_logits = self.modules.ctc_lin(Z)
        p_ctc = self.hparams.log_softmax(ctc_logits)
        
        # LLM embeddings
        if hasattr(self.modules, "projector") and self.modules.projector is not None:
            Z = self.modules.projector(Z)
        
        B, Ts, H = Z.shape
        device = self.device
        tok = self.hparams.LLM_tokenizer
        embed_fn = self.modules.LLM.get_input_embeddings()
        
        # ===== Select targets to predict (flexible) =====
        # Read training targets from hparams, default: all three
        training_targets = getattr(self.hparams, "training_targets", ["word", "canonical", "target"])
        # logger.info(f"Training targets: {training_targets}")
        
        # Tokenize only selected targets
        tokenized_data = {}
        
        if "word" in training_targets:
            wrd_tokens = tok(batch.wrd, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            tokenized_data["wrd"] = {
                "ids": wrd_tokens["input_ids"],
                "mask": wrd_tokens["attention_mask"]
            }
        
        if "canonical" in training_targets:
            phn_can_seq = phn_list_to_seq(batch.phn_list_canonical)
            phn_can_tokens = tok(phn_can_seq, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            tokenized_data["can"] = {
                "ids": phn_can_tokens["input_ids"],
                "mask": phn_can_tokens["attention_mask"]
            }
        
        if "target" in training_targets:
            phn_tgt_seq = phn_list_to_seq(phn_list_training_target)
            phn_tgt_tokens = tok(phn_tgt_seq, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            tokenized_data["tgt"] = {
                "ids": phn_tgt_tokens["input_ids"],
                "mask": phn_tgt_tokens["attention_mask"]
            }
        
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
        self.training_targets = training_targets  # Save for use in compute_objectives
        
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
        
        # ===== Build compact multi-target token sequence (flexible) =====
        # Build compact sequence with dynamic target order
        # e.g., ["word", "canonical", "target"] → [wrd][SEP][can][SEP][tgt][EOS]
        # e.g., ["target"] → [tgt][EOS]
        # e.g., ["canonical", "target"] → [can][SEP][tgt][EOS]
        compact_ids, compact_pos_ranges = self._build_compact_token_ids_flexible(
            tokenized_data=tokenized_data,
            training_targets=training_targets,
            SEP_TGT_ID=SEP_TGT_ID,
            EOS_ID=EOS_ID,
            PAD_ID=PAD_ID,
            device=device
        )
        # Embed compact text sequence
        compact_text_embed = embed_fn(compact_ids)  # [B, max_text_len, H]
        
        # Embed special tokens for speech boundary
        SEP_embed = embed_fn(SEP)
        BOS_embed = embed_fn(BOS)
        
        # Build final input embeddings: [SEP/Prompt] [speech] [BOS] [compact_text]
        has_split_prompt = use_prompt and hasattr(self, "prompt_prefix_embed") and hasattr(self, "prompt_suffix_embed")
        
        if has_split_prompt:
            inputs_embeds = torch.cat([
                self.prompt_prefix_embed.unsqueeze(0).expand(B, -1, -1),
                Z,
                self.prompt_suffix_embed.unsqueeze(0).expand(B, -1, -1),
                compact_text_embed
            ], dim=1)
        elif prompt_embed_batch is not None:
            inputs_embeds = torch.cat([
                prompt_embed_batch,
                SEP_embed,
                Z,
                BOS_embed,
                compact_text_embed
            ], dim=1)
        else:
            inputs_embeds = torch.cat([
                SEP_embed,
                Z,
                BOS_embed,
                compact_text_embed
            ], dim=1)
        
        # Align dtype with LLM weights
        llm_dtype = embed_fn.weight.dtype
        if inputs_embeds.dtype != llm_dtype:
            inputs_embeds = inputs_embeds.to(llm_dtype)
        
        # ===== Build attention mask =====
        seq_len = inputs_embeds.size(1)
        
        if stage == sb.Stage.TRAIN:
            # Calculate text start position in full sequence
            if has_split_prompt:
                text_start = self.prompt_prefix_embed.size(0) + Ts + self.prompt_suffix_embed.size(0)
            else:
                sep_pos = prompt_len
                speech_start = sep_pos + 1
                speech_end = speech_start + Ts
                bos_pos = speech_end
                text_start = bos_pos + 1
            
            # Build attention mask: mask only end padding
            attention_mask = torch.ones(B, seq_len, dtype=torch.long, device=device)
            text_lens = compact_pos_ranges['text_lens']  # [B]
            
            for b in range(B):
                text_len = text_lens[b].item()
                # Mask padding region: [text_start + text_len, seq_len)
                if text_start + text_len < seq_len:
                    attention_mask[b, text_start + text_len:] = 0
            
            # Build labels for compact multi-target sequence (left-shifted for causal LM)
            # Compact sequence: [wrd][SEP_TGT][can][SEP_TGT][tgt][EOS][PAD...]
            # Position i in embeddings predicts token at position i+1 in sequence
            # BOS predicts compact_ids[0], compact_ids[i] → compact_ids[i+1]
            ignore_idx = -100
            labels = torch.full((B, seq_len), ignore_idx, dtype=torch.long, device=device)
            
            # Extract position info from compact_pos_ranges
            text_lens = compact_pos_ranges['text_lens']  # [B]
            training_targets_used = compact_pos_ranges.get('training_targets', training_targets)
            
            for b in range(B):
                text_len = text_lens[b].item()
                
                # BOS position predicts first token in compact sequence
                if has_split_prompt:
                    bos_offset = self.prompt_prefix_embed.size(0) + Ts + self.prompt_suffix_embed.size(0) - 1
                else:
                    bos_offset = prompt_len + 1 + Ts
                
                # Left-shift labels: position i predicts compact_ids[b, i-text_start+1]
                labels[b, bos_offset:bos_offset + text_len] = compact_ids[b, :text_len]
            
            # import pdb; pdb.set_trace()
            labels_noshift = torch.full((B, seq_len), ignore_idx, dtype=torch.long, device=device)
            for b in range(B):
                # Count total tokens to predict
                num_predict = text_lens[b].item() - 1  # Exclude EOS at the end
                bos_pos = bos_offset
                if num_predict > 0:
                    labels_noshift[b, bos_pos] = BOS_ID
                    labels_noshift[b, text_start : text_start + num_predict] = compact_ids[b, :num_predict]
                    labels_noshift[b, text_start + num_predict] = EOS_ID
            
            # Run forward pass
            llm_out = self.modules.LLM(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels_noshift,
                output_hidden_states=False,
                return_dict=True,
            )
            
            ce_logits = llm_out.logits
            loss = llm_out.loss
            # batch count %100 ==0, get debug loss
            # logger.info(f"LLM Loss: {loss.item():.4f}")
            
            # self.debugger.full_diagnosis(batch, stage)
            # import pdb; pdb.set_trace()
            
            # Debug: verify labels and positions (only run once)
            if not hasattr(self, '_label_check_done'):
                b_debug = 0  # First batch
                text_len_debug = text_lens[b_debug].item()
                
                print(f"[DEBUG] TRAIN Compact Sequence Structure:")
                print(f"  Text length: {text_len_debug}")
                print(f"  Training targets: {training_targets_used}")
                
                # Print position info for each target
                for target_name in training_targets_used:
                    if target_name in compact_pos_ranges:
                        starts = compact_pos_ranges[target_name]['starts']
                        ends = compact_pos_ranges[target_name]['ends']
                        print(f"  {target_name}: [{starts[b_debug]},{ends[b_debug]})")
                
                print(f"  Compact IDs sample: {compact_ids[b_debug, :min(20, text_len_debug)].tolist()}")
                
                # Verify no internal padding
                compact_text = self.hparams.LLM_tokenizer.decode(compact_ids[b_debug, :text_len_debug], skip_special_tokens=False)
                print(f"  Decoded compact text: {compact_text[:200]}...")
                
                self._label_check_done = True
            
            # Debug: verify causal masking (only run once)
            if not hasattr(self, '_causal_check_done'):
                print(f"[INFO] MultiTarget Model - LLM type: {type(self.modules.LLM).__name__}")
                print(f"[INFO] Input shape: {inputs_embeds.shape}, Logits shape: {ce_logits.shape}")
                print(f"[INFO] Compact format eliminates internal padding")
                self._causal_check_done = True

            return p_ctc, ce_logits, {
                "labels": labels,
                "compact_pos_ranges": compact_pos_ranges,
                "text_start": text_start,
                "compact_ids": compact_ids
            }, wav_lens
        
        else:
            # ===== Validation/Test Stage =====
            if stage == sb.Stage.VALID:
                # Calculate text start position in full sequence
                if has_split_prompt:
                    text_start = self.prompt_prefix_embed.size(0) + Ts + self.prompt_suffix_embed.size(0)
                else:
                    sep_pos = prompt_len
                    speech_start = sep_pos + 1
                    speech_end = speech_start + Ts
                    bos_pos = speech_end
                    text_start = bos_pos + 1
                
                # Build attention mask: mask only end padding
                attention_mask = torch.ones(B, seq_len, dtype=torch.long, device=device)
                text_lens = compact_pos_ranges['text_lens']  # [B]
                
                for b in range(B):
                    text_len = text_lens[b].item()
                    # Mask padding region: [text_start + text_len, seq_len)
                    if text_start + text_len < seq_len:
                        attention_mask[b, text_start + text_len:] = 0
                
                # Run forward pass
                llm_out = self.modules.LLM(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    output_hidden_states=False,
                    return_dict=True,
                )
                ce_logits = llm_out.logits
                
                # Build labels same as TRAIN (left-shifted for compact sequence)
                ignore_idx = -100
                labels = torch.full((B, seq_len), ignore_idx, dtype=torch.long, device=device)
                
                for b in range(B):
                    text_len = text_lens[b].item()
                    
                    # BOS position predicts first token in compact sequence
                    if has_split_prompt:
                        bos_offset = self.prompt_prefix_embed.size(0) + Ts + self.prompt_suffix_embed.size(0) - 1
                    else:
                        bos_offset = prompt_len + 1 + Ts
                    
                    # Left-shift labels
                    labels[b, bos_offset:bos_offset + text_len] = compact_ids[b, :text_len]
                
                return p_ctc, ce_logits, {
                    "labels": labels,
                    "compact_pos_ranges": compact_pos_ranges,
                    "text_start": text_start,
                    "compact_ids": compact_ids
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
                
                # Calculate max_new_tokens using realistic estimates (not padded batch max)
                # CRITICAL FIX: L_wrd, L_can, L_tgt are BATCH MAX with padding, not actual lengths!
                # Use conservative estimates based on typical sequence lengths
                
                # Estimate actual lengths from batch data
                if hasattr(batch, 'phn_list_target') and len(phn_list_training_target) > 0:
                    # Average phoneme count in batch
                    avg_phn_len = int(sum(len(p) for p in phn_list_training_target) / len(phn_list_training_target))
                else:
                    avg_phn_len = 30  # Conservative default
                
                # Conservative estimate for Arabic words (typically 3-8 tokens per word)
                avg_wrd_len = 25  # Allows for ~3-5 Arabic words
                
                num_targets_active = sum([1 for t in ["word", "canonical", "target"] if t in target_to_generate])
                num_sep_tgt = max(0, num_targets_active - 1)  # Separators between targets
                num_eos = 1  # Final EOS token
                
                max_gen_len = (
                    (avg_wrd_len if "word" in target_to_generate else 0) +
                    (avg_phn_len if "canonical" in target_to_generate else 0) +
                    (avg_phn_len if "target" in target_to_generate else 0) +
                    num_sep_tgt +
                    num_eos +
                    15  # Safety margin
                )
                
                # Cap at reasonable limit to prevent runaway generation
                max_gen_len = min(max_gen_len, 150)
                
                # Add stopping criteria: stop when we see EOS or enough SEP_TGT tokens
                from transformers import StoppingCriteria, StoppingCriteriaList
                
                class MultiTargetStoppingCriteria(StoppingCriteria):
                    """Stop generation when EOS or sufficient SEP_TGT tokens are generated."""
                    def __init__(self, sep_tgt_id, eos_id, max_sep_count=2):
                        self.sep_tgt_id = sep_tgt_id
                        self.eos_id = eos_id
                        self.max_sep_count = max_sep_count
                    
                    def __call__(self, input_ids, scores, **kwargs):
                        # Stop if EOS is generated in any sequence
                        if (input_ids[:, -1] == self.eos_id).any():
                            return True
                        
                        # Count SEP_TGT occurrences (stop after generating enough separators)
                        for seq in input_ids:
                            sep_count = (seq == self.sep_tgt_id).sum().item()
                            if sep_count >= self.max_sep_count:
                                return True
                        return False
                
                # stopping_criteria = StoppingCriteriaList([
                #     MultiTargetStoppingCriteria(
                #         sep_tgt_id=SEP_TGT_ID,
                #         eos_id=EOS_ID,
                #         max_sep_count=num_sep_tgt + 1  # Allow one extra for safety
                #     )
                # ])
                
                # Generate with improved parameters
                # import pdb; pdb.set_trace()
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
                    
                    early_stopping=True,
                    repetition_penalty=1.0,  
                )
                # pdb.set_trace()
                gen_tokens = gen_out
                # tok.decode(gen_out[0], skip_special_tokens=False)
                
                # Debug: log first generated sequence (only once)
                if not hasattr(self, '_gen_debug_done'):
                    try:
                        decoded = self.hparams.LLM_tokenizer.decode(gen_tokens[0], skip_special_tokens=False)
                        print(f"\n[TEST DEBUG] Generated sequence sample:")
                        print(f"  Length: {len(gen_tokens[0])} tokens")
                        print(f"  First 50 token IDs: {gen_tokens[0, :50].tolist()}")
                        print(f"  Decoded (first 200 chars): {decoded[:200]}...")
                        print(f"  Max allowed length: {max_gen_len}")
                    except Exception as e:
                        print(f"[TEST DEBUG] Error decoding: {e}")
                    self._gen_debug_done = True
                
                return p_ctc, None, {
                    "generated_ids": gen_tokens,
                    "target_phonemes": phn_list_training_target,
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
                compact_pos_ranges = ce_targets.get("compact_pos_ranges", {})
                text_start = ce_targets.get("text_start", 0)
                compact_ids = ce_targets.get("compact_ids", None)
                
                if labels is not None and compact_pos_ranges:
                    B, seq_len, vocab_size = ce_logits.shape
                    ignore_idx = -100
                    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_idx, reduction='none')
                    
                    # Get training targets used in this batch
                    training_targets_used = compact_pos_ranges.get('training_targets', ['word', 'canonical', 'target'])
                    text_start = ce_targets.get("text_start", 0)
                    
                    # Map full names to short names
                    target_map = {
                        "word": "wrd",
                        "canonical": "can",
                        "target": "tgt"
                    }
                    
                    # Compute loss for each training target
                    loss_dict = {}
                    
                    for target_name in training_targets_used:
                        short_name = target_map.get(target_name)
                        if short_name is None or short_name not in compact_pos_ranges:
                            continue
                        
                        starts = compact_pos_ranges[short_name]['starts']
                        ends = compact_pos_ranges[short_name]['ends']
                        
                        target_losses = []
                        for b in range(B):
                            # Check if this sample has valid indices for this target
                            if b >= len(starts) or b >= len(ends):
                                continue
                            
                            start = text_start + starts[b]
                            end = text_start + ends[b]
                            
                            if start < end and end <= seq_len:
                                logits_b = ce_logits[b, start:end, :]
                                labels_b = labels[b, start:end]
                                loss_b = ce_loss_fn(logits_b, labels_b)
                                
                                # Apply position weights: 1.5x baseline, 5x for last position (SEP or EOS)
                                weights_b = torch.ones_like(loss_b) * 1.5
                                if len(weights_b) > 0:
                                    weights_b[-1] = 5.0
                                loss_b = (loss_b * weights_b).mean()
                                target_losses.append(loss_b)
                        
                        if target_losses:
                            loss_dict[target_name] = torch.stack(target_losses).mean()
                        else:
                            loss_dict[target_name] = torch.tensor(0.0, device=self.device)
                    
                    # Extract individual losses
                    loss_wrd = loss_dict.get('word', torch.tensor(0.0, device=self.device))
                    loss_can = loss_dict.get('canonical', torch.tensor(0.0, device=self.device))
                    loss_tgt = loss_dict.get('target', torch.tensor(0.0, device=self.device))
                    
                    # Combined loss (average of non-zero losses)
                    active_losses = [loss_dict[t] for t in training_targets_used if t in loss_dict]
                    if active_losses:
                        loss_ce = sum(active_losses) / len(active_losses)
                    else:
                        loss_ce = torch.tensor(0.0, device=self.device)

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
                    
                if getattr(self.hparams, "target_type")  == "target":
                    phn_list_training_target = batch.phn_list_target
                elif getattr(self.hparams, "target_type")  == "perceived":
                    phn_list_training_target = batch.phn_list_perceived
                else:
                    logger.warning(f"Unknown target_type: {getattr(self.hparams, 'target_type')}, defaulting to 'target'")
                    phn_list_training_target = batch.phn_list_target

                if ce_logits is not None and isinstance(ce_targets, dict) and "labels" in ce_targets:
                    labels = ce_targets["labels"]
                    compact_pos_ranges = ce_targets.get("compact_pos_ranges", {})
                    text_start = ce_targets.get("text_start", 0)
                    compact_ids = ce_targets.get("compact_ids", None)
                    
                    llm_predictions = ce_logits.argmax(dim=-1)
                    p_llm = F.log_softmax(ce_logits, dim=-1)
                    
                    valid_mask = (labels != -100)
                    B = labels.size(0)
                    
                    # Extract compact position info (new dict-based structure)
                    wrd_info = compact_pos_ranges.get('wrd', None)
                    can_info = compact_pos_ranges.get('can', None)
                    tgt_info = compact_pos_ranges.get('tgt', None)
                    
                    wrd_starts = wrd_info['starts'] if wrd_info else []
                    wrd_ends = wrd_info['ends'] if wrd_info else []
                    can_starts = can_info['starts'] if can_info else []
                    can_ends = can_info['ends'] if can_info else []
                    tgt_starts = tgt_info['starts'] if tgt_info else []
                    tgt_ends = tgt_info['ends'] if tgt_info else []
                    
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
                            
                            # Extract per-target predictions from compact sequence (flexible targets)
                            # ✅ 修复：正确处理字典结构中的位置信息
                            
                            # 从compact_pos_ranges提取每个目标的信息
                            wrd_info = compact_pos_ranges.get('wrd', None)
                            can_info = compact_pos_ranges.get('can', None)
                            tgt_info = compact_pos_ranges.get('tgt', None)
                            
                            seq_len_b = seq_len
                            
                            # Word prediction
                            if wrd_info is not None and b < len(wrd_info['starts']):
                                wrd_start_compact = wrd_info['starts'][b]
                                wrd_end_compact = wrd_info['ends'][b]
                                wrd_start = text_start + wrd_start_compact
                                wrd_end = text_start + wrd_end_compact
                                
                                if wrd_start < wrd_end and wrd_end <= seq_len_b:
                                    wrd_pred_ids = llm_predictions[b, wrd_start:wrd_end]
                                    wrd_pred = self.hparams.LLM_tokenizer.decode(
                                        wrd_pred_ids, skip_special_tokens=True
                                    ).strip()
                                    
                                    self.wrd_per_metrics.append(
                                        ids=[ids[b]],
                                        predict=[wrd_pred],
                                        target=[batch.wrd[b]],
                                        predict_len=None,
                                        target_len=None,
                                        ind2lab=lambda x: x
                                    )
                            
                            # Canonical phoneme prediction
                            if can_info is not None and b < len(can_info['starts']):
                                can_start_compact = can_info['starts'][b]
                                can_end_compact = can_info['ends'][b]
                                can_start = text_start + can_start_compact
                                can_end = text_start + can_end_compact
                                
                                if can_start < can_end and can_end <= seq_len_b:
                                    can_pred_ids = llm_predictions[b, can_start:can_end]
                                    can_pred = self.hparams.LLM_tokenizer.decode(
                                        can_pred_ids, skip_special_tokens=True
                                    )
                                    self.can_per_metrics.append(
                                        ids=[ids[b]],
                                        predict=[can_pred.split()],
                                        target=[batch.phn_list_canonical[b]],
                                        predict_len=None,
                                        target_len=None,
                                        ind2lab=lambda x: x
                                    )
                            
                            # Target phoneme prediction
                            if tgt_info is not None and b < len(tgt_info['starts']):
                                tgt_start_compact = tgt_info['starts'][b]
                                tgt_end_compact = tgt_info['ends'][b]
                                tgt_start = text_start + tgt_start_compact
                                tgt_end = text_start + tgt_end_compact
                                
                                if tgt_start < tgt_end and tgt_end <= seq_len_b:
                                    tgt_pred_ids = llm_predictions[b, tgt_start:tgt_end]
                                    tgt_pred = self.hparams.LLM_tokenizer.decode(
                                        tgt_pred_ids, skip_special_tokens=True
                                    )
                                    
                                    # Target PER metrics
                                    self.tgt_per_metrics.append(
                                        ids=[ids[b]],
                                        predict=[tgt_pred.split()],
                                        target=[phn_list_training_target[b]],
                                        predict_len=None,
                                        target_len=None,
                                        ind2lab=lambda x: x
                                    )
                                    
                                    # Overall PER and MPD F1 (use target phonemes)
                                    self.llm_per_metrics.append(
                                        ids=[ids[b]],
                                        predict=[tgt_pred.split()],
                                        target=[" ".join(phn_list_training_target[b]).split()],
                                        predict_len=None,
                                        target_len=None,
                                        ind2lab=lambda x: x
                                    )
                                    
                                    self.mpd_f1_metrics.append(
                                        ids=[ids[b]],
                                        predict=[tgt_pred.split()],
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
                    if getattr(self.hparams, "target_type")  == "target":
                        phn_list_training_target = batch.phn_list_target
                    elif getattr(self.hparams, "target_type")  == "perceived":
                        phn_list_training_target = batch.phn_list_perceived
                    else:
                        logger.warning(f"Unknown target_type: {getattr(self.hparams, 'target_type')}, defaulting to 'target'")
                        phn_list_training_target = batch.phn_list_target
                        
                    refs_tgt = phn_list_to_seq(phn_list_training_target)
                    refs_can = phn_list_to_seq(batch.phn_list_canonical)
                    # refs_wrd: Keep as Arabic text (from batch.wrd directly)
                    refs_wrd = batch.wrd  # List of Arabic words
                    
                    # Compute per-target PER scores
                    if "word" in target_to_generate:
                        # For words: refs_wrd is list of Arabic texts, convert to list of lists
                        refs_wrd_list = [[w] if isinstance(w, str) else w for w in refs_wrd]
                        # pdb.set_trace()
                        # TODO
                        self.wrd_per_metrics.append(
                            ids=ids,
                            predict=[hyp.split() for hyp in hyps_wrd],
                            target=refs_wrd_list,
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
            per_ctc = self.per_metrics.summarize("error_rate") if len(self.per_metrics.scores) > 0 else 0.0
            per_llm = self.llm_per_metrics.summarize("error_rate") if len(self.llm_per_metrics.scores) > 0 else 0.0
            
            # Handle flexible training targets: only summarize metrics for targets that were trained
            training_targets = getattr(self.hparams, "training_targets", ["word", "canonical", "target"])
            
            # Safely summarize per-target metrics (avoid ZeroDivisionError for empty metrics)
            per_wrd = self.wrd_per_metrics.summarize("error_rate") if len(self.wrd_per_metrics.scores) > 0 else 0.0
            per_can = self.can_per_metrics.summarize("error_rate") if len(self.can_per_metrics.scores) > 0 else 0.0
            per_tgt = self.tgt_per_metrics.summarize("error_rate") if len(self.tgt_per_metrics.scores) > 0 else 0.0
            
            # Compute average only over targets that were actually used
            active_per_values = []
            if "word" in training_targets and len(self.wrd_per_metrics.scores) > 0:
                active_per_values.append(per_wrd)
            if "canonical" in training_targets and len(self.can_per_metrics.scores) > 0:
                active_per_values.append(per_can)
            if "target" in training_targets and len(self.tgt_per_metrics.scores) > 0:
                active_per_values.append(per_tgt)
            
            per_avg = sum(active_per_values) / len(active_per_values) if active_per_values else 0.0
            
            stage_stats["ctc_per"] = per_ctc
            stage_stats["llm_per"] = per_llm
            stage_stats["wrd_per"] = per_wrd
            stage_stats["can_per"] = per_can
            stage_stats["tgt_per"] = per_tgt
            stage_stats["per_avg"] = per_avg
            
            # Safely summarize LLM loss and metrics
            llm_loss = self.llm_metrics.summarize("average") if len(self.llm_metrics.scores) > 0 else 0.0
            stage_stats["llm_loss"] = llm_loss
            
            # Only compute MPD F1 if metrics were collected
            if len(self.mpd_f1_metrics.scores) > 0:
                mpd_f1 = self.mpd_f1_metrics.summarize("mpd_f1")
                stage_stats["llm_mpd_f1"] = mpd_f1
            else:
                mpd_f1 = 0.0
                stage_stats["llm_mpd_f1"] = 0.0
        
            if hasattr(self.modules, "ctc_lin"):
                ctc_loss = self.ctc_metrics.summarize("average") if len(self.ctc_metrics.scores) > 0 else 0.0
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
                            -ckpt.meta["TGT_PER"],
                            ckpt.meta["LLM_MPD_F1"]
                            -ckpt.meta["CAN_PER"],
                            -ckpt.meta["WRD_PER"],
                            -ckpt.meta["PER_AVG"],
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
            per_ctc = self.per_metrics.summarize("error_rate") if len(self.per_metrics.scores) > 0 else 0.0
            mpd_f1 = self.mpd_f1_metrics.summarize("mpd_f1") if len(self.mpd_f1_metrics.scores) > 0 else 0.0
            
            # Safely summarize per-target metrics
            per_llm = self.llm_per_metrics.summarize("error_rate") if len(self.llm_per_metrics.scores) > 0 else None
            per_wrd = self.wrd_per_metrics.summarize("error_rate") if len(self.wrd_per_metrics.scores) > 0 else None
            per_can = self.can_per_metrics.summarize("error_rate") if len(self.can_per_metrics.scores) > 0 else None
            per_tgt = self.tgt_per_metrics.summarize("error_rate") if len(self.tgt_per_metrics.scores) > 0 else None
            
            # Compute average only if all metrics are available
            if per_wrd is not None and per_can is not None and per_tgt is not None:
                per_avg = (per_wrd + per_can + per_tgt) / 3.0
            else:
                per_avg = None
            
            # Safely summarize LLM loss
            llm_loss = self.llm_metrics.summarize("average") if len(self.llm_metrics.scores) > 0 else None
            
            ctc_loss = self.ctc_metrics.summarize("average") if len(self.ctc_metrics.scores) > 0 else 0.0
            
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
                        top_k=50, top_p=0.9, num_beams=1, use_json_prompt=False, use_chat_prompt=False):
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
            use_json_prompt: 是否使用JSON格式的Prompt (default: False)
            use_chat_prompt: 是否使用Chat格式的Prompt，三行输出 (default: False)
            
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
        self._ensure_initialized_infer_target()
        
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        ids = batch.id
        tok = self.hparams.LLM_tokenizer

        # Audio Encoder
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
        
        # 解析生成的输出
        if use_json_prompt:
            # JSON格式输出解析
            words_list, canonical_list, target_list = self._parse_json_output(gen_out, tok)
        elif use_chat_prompt:
            # Chat格式输出解析（三行格式）
            words_list, canonical_list, target_list = self._parse_chat_output(gen_out, tok)
        else:
            # 原始格式输出解析（使用SEP_TGT分隔符）
            words_list, canonical_list, target_list = self._parse_separator_output(gen_out, tok, SEP_TGT_ID, EOS_ID)
        
        return {
            "ids": ids,
            "generated_tokens": gen_out,
            "words": words_list,
            "canonical": canonical_list,
            "target": target_list,
            "ctc_predictions": ctc_predictions,
        }
    
    def _parse_json_output(self, gen_out, tok):
        """
        解析JSON格式的LLM输出.
        
        Args:
            gen_out: 生成的token张量 [B, seq_len]
            tok: tokenizer
            
        Returns:
            (words_list, canonical_list, target_list)
        """
        json_builder = JSONPromptBuilder(use_json=True)
        words_list = []
        canonical_list = []
        target_list = []
        
        for b_idx in range(gen_out.shape[0]):
            gen_token_seq = gen_out[b_idx]
            gen_text = tok.decode(gen_token_seq, skip_special_tokens=True)
            
            # 解析JSON
            parsed = json_builder.parse_json_output(gen_text)
            
            # 提取各个字段
            wrd = parsed.get("transcription", "") or parsed.get("word", "")
            can = parsed.get("canonical_phonemes", "") or parsed.get("canonical", "")
            tgt = parsed.get("target_phonemes", "") or parsed.get("target", "")
            
            # 清理文本
            words_list.append(json_builder.clean_prediction(wrd).strip())
            canonical_list.append(json_builder.clean_prediction(can).strip())
            target_list.append(json_builder.clean_prediction(tgt).strip())
        
        return words_list, canonical_list, target_list
    
    def _parse_separator_output(self, gen_out, tok, sep_tgt_id, eos_id):
        """
        解析使用SEP_TGT分隔符的输出.
        
        Args:
            gen_out: 生成的token张量 [B, seq_len]
            tok: tokenizer
            sep_tgt_id: SEP_TGT token ID
            eos_id: EOS token ID
            
        Returns:
            (words_list, canonical_list, target_list)
        """
        words_list = []
        canonical_list = []
        target_list = []
        
        for b_idx in range(gen_out.shape[0]):
            gen_token_seq = gen_out[b_idx]
            
            # Find positions of SEP_TGT_ID and EOS_ID
            sep_tgt_positions = (gen_token_seq == sep_tgt_id).nonzero(as_tuple=True)[0].tolist()
            eos_positions = (gen_token_seq == eos_id).nonzero(as_tuple=True)[0].tolist()
            
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
        
        return words_list, canonical_list, target_list
    
    def _parse_chat_output(self, gen_out, tok):
        """
        解析Chat格式的LLM输出（三行格式：transcription, canonical, target）.
        
        Args:
            gen_out: 生成的token张量 [B, seq_len]
            tok: tokenizer
            
        Returns:
            (words_list, canonical_list, target_list)
        """
        chat_builder = ChatPromptBuilder(format_type="newline")
        words_list = []
        canonical_list = []
        target_list = []
        
        for b_idx in range(gen_out.shape[0]):
            gen_token_seq = gen_out[b_idx]
            gen_text = tok.decode(gen_token_seq, skip_special_tokens=True)
            
            # 解析三行格式
            parsed = chat_builder.parse_chat_output(gen_text)
            
            # 提取各个字段
            wrd = parsed.get("transcription", "")
            can = parsed.get("canonical_phonemes", "")
            tgt = parsed.get("target_phonemes", "")
            
            words_list.append(wrd.strip())
            canonical_list.append(can.strip())
            target_list.append(tgt.strip())
        
        return words_list, canonical_list, target_list

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
            self.checkpointer.recover_if_possible(min_key="TGT_PER", max_key="LLM_MPD_F1")
            
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
