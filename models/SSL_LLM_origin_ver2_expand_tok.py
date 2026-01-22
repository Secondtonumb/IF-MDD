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
    get_peft_model_state_dict = None
    set_peft_model_state_dict = None

# Import SpeechBrain checkpoint utilities for registering hooks
from speechbrain.utils.checkpoints import register_checkpoint_hooks, mark_as_saver, mark_as_loader


@register_checkpoint_hooks
class PeftAdapterRecoverable:
    """
    Wrapper to save ONLY the adapter weights of a PeftModel in SpeechBrain checkpoints.
    
    This is crucial for efficient checkpointing when fine-tuning large LLMs with LoRA/Adapter:
    - Base LLM weights (~8GB+) are NOT saved each epoch
    - Only adapter weights (~10-100MB) are saved
    - Loading: Base LLM loads once, then adapter weights are merged
    
    Usage in YAML:
        LLM: !new:peft.PeftModel
            base_model: !ref <base_llm>
            peft_config: !ref <lora_config>
    
    The checkpointer will automatically wrap PeftModel objects with this class.
    """
    def __init__(self, model):
        """
        Args:
            model: A PeftModel instance (or DDP-wrapped PeftModel)
        """
        # Handle DDP wrapper
        if hasattr(model, 'module'):
            self.model = model.module
            self._is_ddp = True
        else:
            self.model = model
            self._is_ddp = False
        self._original_wrapper = model  # Keep reference to original (may be DDP)
    
    @mark_as_saver
    def save(self, path):
        """Save ONLY the adapter state dict (not full model)"""
        if get_peft_model_state_dict is None:
            raise ImportError("peft library is required for PeftAdapterRecoverable")
        state_dict = get_peft_model_state_dict(self.model)
        torch.save(state_dict, path)
    
    @mark_as_loader
    def load(self, path, end_of_epoch=False, device=None):
        """Load adapter weights into the PeftModel"""
        if set_peft_model_state_dict is None:
            raise ImportError("peft library is required for PeftAdapterRecoverable")
        del end_of_epoch  # Unused
        state_dict = torch.load(path, map_location=device)
        set_peft_model_state_dict(self.model, state_dict)
    
    def to(self, device):
        """Move model to device"""
        self._original_wrapper.to(device)
        return self
    
    def __getattr__(self, name):
        """Proxy attribute access to underlying model for compatibility"""
        if name in ['model', '_is_ddp', '_original_wrapper']:
            return object.__getattribute__(self, name)
        return getattr(self._original_wrapper, name)


@register_checkpoint_hooks
class LLMAdapterOnlyRecoverable:
    """
    Alternative wrapper for when you want explicit control over what gets saved.
    
    Use this when:
    1. LLM is NOT wrapped with PeftModel but has custom adapter modules
    2. You want to save only specific layers (e.g., projector, adapter layers)
    
    Example usage in init_optimizers:
        self.checkpointer.add_recoverable(
            "llm_adapters",
            LLMAdapterOnlyRecoverable(
                self.modules.LLM,
                adapter_names=["lora", "adapter"]  # Name patterns to match
            )
        )
    """
    def __init__(self, llm_model, adapter_names=None, save_patterns=None):
        """
        Args:
            llm_model: The LLM module
            adapter_names: List of substring patterns to match adapter layer names
                          e.g., ["lora", "adapter", "projector"]
            save_patterns: Alternative regex patterns for matching
        """
        self.llm_model = llm_model
        self.adapter_names = adapter_names or ["lora", "adapter"]
        self.save_patterns = save_patterns
        
    def _get_adapter_params(self):
        """Get state dict containing only adapter parameters"""
        adapter_state = {}
        
        # Handle DDP
        model = self.llm_model.module if hasattr(self.llm_model, 'module') else self.llm_model
        
        for name, param in model.named_parameters():
            # Check if this parameter belongs to an adapter
            is_adapter = any(adapter_name in name.lower() for adapter_name in self.adapter_names)
            if is_adapter:
                adapter_state[name] = param.data.clone()
        
        return adapter_state
    
    @mark_as_saver
    def save(self, path):
        """Save only adapter parameters"""
        state_dict = self._get_adapter_params()
        torch.save(state_dict, path)
    
    @mark_as_loader
    def load(self, path, end_of_epoch=False, device=None):
        """Load adapter parameters"""
        del end_of_epoch  # Unused
        state_dict = torch.load(path, map_location=device)
        
        model = self.llm_model.module if hasattr(self.llm_model, 'module') else self.llm_model
        
        current_state = model.state_dict()
        for name, param in state_dict.items():
            if name in current_state:
                current_state[name] = param
        
        model.load_state_dict(current_state, strict=False)
    
    def to(self, device):
        self.llm_model.to(device)
        return self

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


class PhonemeTokenMapper:
    """
    Maps phoneme IDs (from CTC label_encoder) to LLM reserved special tokens.
    
    This avoids semantic corruption from directly tokenizing phoneme text with LLM tokenizer.
    Instead, we use LLM's reserved special tokens as a dedicated phoneme vocabulary.
    
    Mapping Strategy:
    - Phoneme ID 0 (blank) -> <|reserved_special_token_0|>
    - Phoneme ID 1 -> <|reserved_special_token_1|>
    - ...
    - Phoneme ID N -> <|reserved_special_token_N|>
    
    Smart Initialization:
    - New phoneme tokens are initialized with embeddings from similar English characters
    - e.g., <|p_aa|> inherits from 'a', <|p_sh|> inherits from 's', etc.
    - This gives the model a "semantic hint" to start learning from
    
    Special handling:
    - BOS/EOS use LLM's native BOS/EOS tokens
    - Reserved tokens 0-99 are available in Llama 3
    """
    
    def __init__(self, label_encoder, llm_tokenizer, llm_model=None, device='cuda', 
                 smart_init=True, noise_scale=0.01):
        """
        Initialize the phoneme-to-LLM-token mapper.
        
        Args:
            label_encoder: SpeechBrain label encoder with phoneme vocabulary
            llm_tokenizer: LLM tokenizer (e.g., Llama 3 tokenizer)
            llm_model: LLM model (required for smart initialization of embeddings)
            device: torch device
            smart_init: Whether to use smart embedding initialization
            noise_scale: Scale of noise to add during smart initialization (default: 0.01)
        """
        self.label_encoder = label_encoder
        self.tokenizer = llm_tokenizer
        self.llm_model = llm_model
        self.device = device
        self.smart_init = smart_init
        self.noise_scale = noise_scale
        
        # Build mappings
        self._build_mappings()
        
        # Apply smart initialization if model is provided
        if smart_init and llm_model is not None:
            self._apply_smart_initialization()
    
    def _infer_reference_char(self, phoneme_name: str) -> str:
        """
        Infer a reference English text for a phoneme based on its name.
        
        Strategy: Use the phoneme name itself (e.g., 'aa', 'sh') as the reference text.
        The LLM tokenizer will find the best matching token for this text.
        
        Args:
            phoneme_name: The phoneme name from label_encoder (e.g., 'aa', 'sh', 'ng')
            
        Returns:
            A reference string to use for embedding initialization
        """
        phn = phoneme_name.lower().strip()
        
        # Remove stress markers (0, 1, 2) if present (e.g., 'aa0' -> 'aa')
        phn_base = phn.rstrip('012')
        
        # Special tokens -> space (neutral embedding)
        if phn in ['<blank>', 'sil', 'sp', 'spn', '<unk>', 'SIL', 'SPN', '<pad>']:
            return ' '
        
        # Skip BOS/EOS - they use LLM native tokens, no need for smart init
        if phn in ['<bos>', '<eos>']:
            return None
        
        # Use the phoneme name itself as reference text
        # The tokenizer will find the closest matching embedding
        return phn_base if phn_base else phn
        
    def _build_mappings(self):
        """Build bidirectional mappings between phoneme IDs and LLM token IDs."""
        
        # Get phoneme vocabulary from label_encoder
        # label_encoder.lab2ind: {'aa': 1, 'sil': 2, ...}
        # label_encoder.ind2lab: {1: 'aa', 2: 'sil', ...}
        
        self.num_phonemes = len(self.label_encoder.lab2ind)
        logger.info(f"[PhonemeTokenMapper] Building mappings for {self.num_phonemes} phonemes")
        
        # Phoneme ID -> LLM Token ID
        self.phn_to_llm = {}
        # LLM Token ID -> Phoneme ID  
        self.llm_to_phn = {}
        # LLM Token ID -> Phoneme Name (for readable decoding)
        self.llm_to_phn_name = {}
        # Phoneme Name -> LLM Token ID
        self.phn_name_to_llm = {}
        
        # Reserve token 0-2 for special purposes
        # Token 0: <blank> (CTC blank)
        # Token 1: <bos> equivalent (use LLM's actual BOS)
        # Token 2: <eos> equivalent (use LLM's actual EOS)
        
        # Get reserved token IDs from tokenizer
        # Llama 3 format: <|reserved_special_token_N|>
        reserved_token_offset = 0  # Start from reserved_special_token_0
        
        for phn_id in range(self.num_phonemes):
            # Get phoneme name
            phn_name = self.label_encoder.ind2lab.get(phn_id, f"UNK_{phn_id}")
            
            # Skip BOS/EOS - they use LLM's native tokens
            if phn_name in ['<bos>', '<eos>']:
                if phn_name == '<bos>':
                    llm_token_id = self.tokenizer.bos_token_id
                else:
                    llm_token_id = self.tokenizer.eos_token_id
            else:
                # Map to reserved special token
                reserved_token_name = f"<|reserved_special_token_{reserved_token_offset}|>"
                
                # Get token ID for this reserved token
                llm_token_id = self.tokenizer.convert_tokens_to_ids(reserved_token_name)
                
                if llm_token_id == self.tokenizer.unk_token_id:
                    logger.warning(f"[PhonemeTokenMapper] Reserved token {reserved_token_name} not found!")
                    # Fallback: use offset directly (risky but better than crash)
                    llm_token_id = 128000 + reserved_token_offset  # Llama 3 reserved tokens start around here
                
                reserved_token_offset += 1
            
            # Store mappings
            self.phn_to_llm[phn_id] = llm_token_id
            self.llm_to_phn[llm_token_id] = phn_id
            self.llm_to_phn_name[llm_token_id] = phn_name
            self.phn_name_to_llm[phn_name] = llm_token_id
        
        # Store special token IDs for convenience
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        
        # Get blank token ID (usually phoneme ID 0)
        blank_phn_id = self.label_encoder.lab2ind.get('<blank>', 0)
        self.blank_llm_token_id = self.phn_to_llm.get(blank_phn_id, self.phn_to_llm[0])
        
        logger.info(f"[PhonemeTokenMapper] Mapping complete:")
        logger.info(f"   - Total phonemes: {self.num_phonemes}")
        logger.info(f"   - Reserved tokens used: {reserved_token_offset}")
        logger.info(f"   - BOS token ID: {self.bos_token_id}")
        logger.info(f"   - EOS token ID: {self.eos_token_id}")
        logger.info(f"   - Blank token ID: {self.blank_llm_token_id}")
        
        # Print first few mappings for verification
        logger.info(f"   - Sample mappings:")
        for i, (phn_id, llm_id) in enumerate(list(self.phn_to_llm.items())[:68]):
            phn_name = self.label_encoder.ind2lab.get(phn_id, f"UNK_{phn_id}")
            logger.info(f"     {phn_name} (id={phn_id}) -> LLM token {llm_id}")
    
    def _apply_smart_initialization(self):
        """
        Apply smart initialization to phoneme token embeddings.
        
        For each phoneme, find a reference text (the phoneme name itself),
        get its embedding from the LLM, and use it to initialize the 
        reserved special token's embedding (with small noise added).
        
        This gives the model a "semantic hint" so it doesn't start from random.
        """
        if self.llm_model is None:
            logger.warning("[PhonemeTokenMapper] No LLM model provided, skipping smart initialization")
            return
        
        # Handle DDP wrapper
        model = self.llm_model
        if hasattr(model, 'module'):
            model = model.module
        
        # Get embedding layer
        embeddings = model.get_input_embeddings().weight.data
        embed_dim = embeddings.size(1)
        
        initialized_count = 0
        skipped_count = 0
        
        logger.info(f"[PhonemeTokenMapper] Applying smart initialization...")
        
        with torch.no_grad():
            for phn_id in range(self.num_phonemes):
                phn_name = self.label_encoder.ind2lab.get(phn_id, f"UNK_{phn_id}")
                llm_token_id = self.phn_to_llm.get(phn_id)
                
                if llm_token_id is None:
                    skipped_count += 1
                    continue
                
                # Skip BOS/EOS - they already have meaningful embeddings
                if phn_name in ['<bos>', '<eos>']:
                    skipped_count += 1
                    continue
                
                # Get reference text for this phoneme
                ref_text = self._infer_reference_char(phn_name)
                if ref_text is None:
                    skipped_count += 1
                    continue
                
                # Tokenize reference text to get reference token ID
                ref_tokens = self.tokenizer(ref_text, add_special_tokens=False)['input_ids']
                
                if len(ref_tokens) == 0:
                    # If tokenization fails, use space
                    ref_tokens = self.tokenizer(' ', add_special_tokens=False)['input_ids']
                
                # Use first token's embedding as reference
                ref_token_id = ref_tokens[0]
                
                if ref_token_id >= embeddings.size(0) or llm_token_id >= embeddings.size(0):
                    logger.warning(f"[SmartInit] Token ID out of range: ref={ref_token_id}, phn={llm_token_id}")
                    skipped_count += 1
                    continue
                
                # Get reference embedding
                ref_embed = embeddings[ref_token_id].clone()
                
                # Add small noise to avoid identical embeddings
                noise = torch.randn_like(ref_embed) * self.noise_scale
                
                # Initialize the phoneme token's embedding
                embeddings[llm_token_id] = ref_embed + noise
                
                initialized_count += 1
                
                # Log first few initializations for debugging
                if initialized_count <= 10:
                    logger.info(f"   {phn_name} -> '{ref_text}' (token {ref_token_id}) -> reserved token {llm_token_id}")
        
        logger.info(f"[PhonemeTokenMapper] Smart initialization complete:")
        logger.info(f"   - Initialized: {initialized_count} phoneme tokens")
        logger.info(f"   - Skipped: {skipped_count} tokens (BOS/EOS/special)")
        logger.info(f"   - Noise scale: {self.noise_scale}")
    
    def encode_phoneme_ids(self, phoneme_ids: torch.Tensor) -> torch.Tensor:
        """
        Convert phoneme IDs (from label_encoder) to LLM token IDs.
        
        Args:
            phoneme_ids: Tensor of phoneme IDs [B, L] or [L]
            
        Returns:
            Tensor of LLM token IDs with same shape
        """
        # Handle both batched and unbatched input
        input_shape = phoneme_ids.shape
        flat_ids = phoneme_ids.flatten().tolist()
        
        # Map each phoneme ID to LLM token ID
        llm_ids = [self.phn_to_llm.get(int(pid), self.blank_llm_token_id) for pid in flat_ids]
        
        # Convert back to tensor with original shape
        result = torch.tensor(llm_ids, dtype=torch.long, device=phoneme_ids.device)
        return result.view(input_shape)
    
    def encode_phoneme_list(self, phoneme_list: list) -> torch.Tensor:
        """
        Convert a list of phoneme names to LLM token IDs.
        
        Args:
            phoneme_list: List of phoneme names, e.g., ["sil", "aa", "b", "sil"]
            
        Returns:
            Tensor of LLM token IDs [L]
        """
        llm_ids = []
        for phn_name in phoneme_list:
            if phn_name in self.phn_name_to_llm:
                llm_ids.append(self.phn_name_to_llm[phn_name])
            else:
                # Unknown phoneme - use blank
                logger.warning(f"[PhonemeTokenMapper] Unknown phoneme '{phn_name}', using blank")
                llm_ids.append(self.blank_llm_token_id)
        
        return torch.tensor(llm_ids, dtype=torch.long, device=self.device)
    
    def encode_batch_phoneme_lists(self, batch_phoneme_lists: list, return_mask: bool = True):
        """
        Encode a batch of phoneme lists to padded LLM token IDs.
        
        Args:
            batch_phoneme_lists: List of phoneme lists, e.g., [["sil", "aa"], ["sil", "b", "aa"]]
            return_mask: Whether to return attention mask
            
        Returns:
            token_ids: Padded tensor [B, max_len]
            attention_mask: Mask tensor [B, max_len] (if return_mask=True)
        """
        # Encode each sequence
        encoded_seqs = [self.encode_phoneme_list(phn_list) for phn_list in batch_phoneme_lists]
        
        # Find max length
        max_len = max(len(seq) for seq in encoded_seqs)
        batch_size = len(encoded_seqs)
        
        # Pad sequences
        padded_ids = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long, device=self.device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=self.device)
        
        for i, seq in enumerate(encoded_seqs):
            seq_len = len(seq)
            padded_ids[i, :seq_len] = seq
            attention_mask[i, :seq_len] = 1
        
        if return_mask:
            return padded_ids, attention_mask
        return padded_ids
    
    def decode_llm_ids(self, llm_ids: torch.Tensor, skip_special: bool = True) -> list:
        """
        Convert LLM token IDs back to phoneme names.
        
        Args:
            llm_ids: Tensor of LLM token IDs [B, L] or [L]
            skip_special: Whether to skip BOS/EOS/PAD tokens
            
        Returns:
            List of phoneme name lists (if batched) or single list
        """
        is_batched = llm_ids.dim() == 2
        
        if not is_batched:
            llm_ids = llm_ids.unsqueeze(0)
        
        batch_results = []
        
        for seq in llm_ids:
            phonemes = []
            for token_id in seq.tolist():
                # Skip special tokens if requested
                if skip_special:
                    if token_id in [self.bos_token_id, self.eos_token_id, self.pad_token_id]:
                        continue
                
                # Map back to phoneme name
                if token_id in self.llm_to_phn_name:
                    phn_name = self.llm_to_phn_name[token_id]
                    # Skip blank in output
                    if phn_name != '<blank>':
                        phonemes.append(phn_name)
                else:
                    # Unknown token - might be from LLM generation outside phoneme vocab
                    # Try to decode with tokenizer for debugging
                    decoded = self.tokenizer.decode([token_id])
                    logger.debug(f"[PhonemeTokenMapper] Unknown LLM token {token_id} -> '{decoded}'")
            
            batch_results.append(phonemes)
        
        if not is_batched:
            return batch_results[0]
        return batch_results
    
    def decode_to_string(self, llm_ids: torch.Tensor, skip_special: bool = True) -> list:
        """
        Convert LLM token IDs to space-separated phoneme strings.
        
        Args:
            llm_ids: Tensor of LLM token IDs [B, L] or [L]
            skip_special: Whether to skip BOS/EOS/PAD tokens
            
        Returns:
            List of phoneme strings (if batched) or single string
        """
        phoneme_lists = self.decode_llm_ids(llm_ids, skip_special=skip_special)
        
        if isinstance(phoneme_lists[0], list):
            # Batched
            return [" ".join(phn_list) for phn_list in phoneme_lists]
        else:
            # Single sequence
            return " ".join(phoneme_lists)
    
    def get_valid_token_ids(self) -> list:
        """
        Get list of all valid LLM token IDs for phoneme generation.
        Useful for constrained decoding.
        
        Returns:
            List of valid LLM token IDs
        """
        valid_ids = list(self.llm_to_phn.keys())
        # Also include BOS/EOS
        valid_ids.extend([self.bos_token_id, self.eos_token_id])
        return list(set(valid_ids))
    
    def create_phoneme_bias(self, vocab_size: int = None) -> torch.Tensor:
        """
        Create a bias tensor for constrained generation.
        Valid phoneme tokens get 0 bias, others get -inf.
        
        Args:
            vocab_size: Size of LLM vocabulary (auto-detected if None)
            
        Returns:
            Bias tensor [vocab_size]
        """
        if vocab_size is None:
            vocab_size = len(self.tokenizer)
        
        bias = torch.full((vocab_size,), float('-inf'), device=self.device)
        
        for llm_token_id in self.get_valid_token_ids():
            if 0 <= llm_token_id < vocab_size:
                bias[llm_token_id] = 0.0
        
        return bias


class SSL_LLM_origin_ver2_expand_tok(sb.Brain):
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
            hidden_size = self.hparams.LLM_DIM
            self.llm_norm = nn.LayerNorm(hidden_size).to(self.device)
            print(f"[Lazy Init] Created llm_norm with hidden_size={hidden_size}")
        
        # ===== Initialize PhonemeTokenMapper =====
        # This maps phoneme IDs from label_encoder to LLM reserved tokens
        # to avoid semantic corruption from directly tokenizing phoneme text
        # import pdb; pdb.set_trace()
        if not hasattr(self, "phoneme_mapper") or self.phoneme_mapper is None:
            if hasattr(self, "label_encoder") and self.label_encoder is not None:
                # Get smart_init setting from hparams (default: True)
                smart_init = getattr(self.hparams, "phoneme_smart_init", True)
                noise_scale = getattr(self.hparams, "phoneme_init_noise", 0.01)
                
                self.phoneme_mapper = PhonemeTokenMapper(
                    label_encoder=self.label_encoder,
                    llm_tokenizer=self.hparams.LLM_tokenizer,
                    llm_model=llm_handle,  # Pass LLM for smart initialization
                    device=self.device,
                    smart_init=smart_init,
                    noise_scale=noise_scale
                )
                print(f"[Lazy Init] Created PhonemeTokenMapper with {self.phoneme_mapper.num_phonemes} phonemes")
                print(f"[Lazy Init] Smart init: {smart_init}, noise_scale: {noise_scale}")
                
                # Update phoneme_bias to use mapper's valid tokens
                self.phoneme_bias = self.phoneme_mapper.create_phoneme_bias()
                print(f"[Lazy Init] Updated phoneme_bias with {len(self.phoneme_mapper.get_valid_token_ids())} valid tokens")
            else:
                logger.warning("[Lazy Init] label_encoder not available, PhonemeTokenMapper not initialized")
                self.phoneme_mapper = None
        
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
                        "content": "You are a phoneme transcriber."
                    },
                    {
                        "role": "user", 
                        # 语音在指令之前
                        "content": f"{PLACEHOLDER}\nTranscribe the preceding speech into CMUdict phonemes."
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
        # import pdb; pdb.set_trace()
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "augmentation"):
            wavs = self.hparams.augmentation(wavs)

        # AudioEncoder + Projector
        wav_feats = self.modules.perceived_ssl(wavs)  # [B, T, 1024]
        import pdb
        # AudioEncoder
        try:
            # with Conformer it give extra
            Z, _ = self.hparams.audio_encoder_modules(wavs)
        except:
            # SSL projection only
            Z = self.hparams.audio_encoder_modules(wavs)
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
        
        # ===== Tokenize phoneme sequences using PhonemeTokenMapper =====
        # This maps phoneme names to LLM reserved tokens to avoid semantic corruption
        if hasattr(self, "phoneme_mapper") and self.phoneme_mapper is not None:
            # Use PhonemeTokenMapper for proper phoneme -> reserved token mapping
            # import pdb;  pdb.set_trace()
            phn_ids, phn_mask = self.phoneme_mapper.encode_batch_phoneme_lists(
                batch.phn_list_target, return_mask=True
            )
            # import pdb; pdb.set_trace()
            # phn_ids: [B, L_phn] - LLM reserved token IDs
            # phn_mask: [B, L_phn] - attention mask (1 for valid, 0 for padding)
            
            # Get special token IDs from mapper
            BOS_ID = self.phoneme_mapper.bos_token_id
            EOS_ID = self.phoneme_mapper.eos_token_id
            PAD_ID = self.phoneme_mapper.pad_token_id
        else:
            # Fallback: use original tokenizer (not recommended, will cause semantic issues)
            logger.warning("[compute_forward] PhonemeTokenMapper not available, using raw tokenizer (may cause issues)")
            phn_seq = phn_list_to_seq(batch.phn_list_target)
            phn_tokens = tok(phn_seq, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            phn_ids = phn_tokens["input_ids"]
            phn_mask = phn_tokens["attention_mask"]
            BOS_ID = tok.bos_token_id
            EOS_ID = tok.eos_token_id
            PAD_ID = tok.pad_token_id if tok.pad_token_id is not None else 0
        
        L_phn = phn_ids.size(1)
        
        # ===== Prepare special tokens =====
        # SEP token (use a reserved special token not used for phonemes)
        # We use reserved_special_token_99 to avoid collision with phoneme mappings
        SEP_token_name = "<|reserved_special_token_99|>"
        SEP_ID = tok.convert_tokens_to_ids(SEP_token_name)
        if SEP_ID == tok.unk_token_id:
            # Fallback if token not found
            SEP_ID = 128099  # Llama 3 reserved token approximate location
        # import pdb; pdb.set_trace()
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
        
        # ===== Build input embedding sequence =====
        # Sequence structure: <prompt?> <SEP> <speech> <BOS> <phoneme_tokens> <EOS>
        # Note: we will construct labels such that speech/prompt section is masked (-100)
        
        # Embed phoneme tokens
        phn_embed = embed_fn(phn_ids)  # [B, L_phn, H]
        
        # Embed special tokens
        SEP_embed = embed_fn(SEP)  # [B, 1, H]
        BOS_embed = embed_fn(BOS)  # [B, 1, H]
        EOS_embed = embed_fn(EOS)  # [B, 1, H]
        
        # Concatenate everything
        has_split_prompt = use_prompt and hasattr(self, "prompt_prefix_embed") and hasattr(self, "prompt_suffix_embed")

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
        elif prompt_embed is not None:
            inputs_embeds = torch.cat([
                prompt_embed_batch,  # [B, P, H]
                SEP_embed,          # [B, 1, H]
                Z,                   # [B, Ts, H]
                BOS_embed,          # [B, 1, H]
                phn_embed,          # [B, L_phn, H]
                EOS_embed           # [B, 1, H]
            ], dim=1)  # [B, P+1+Ts+1+L_phn+1, H]
        else:
            inputs_embeds = torch.cat([
                SEP_embed,          # [B, 1, H]
                Z,                   # [B, Ts, H]
                BOS_embed,          # [B, 1, H]
                phn_embed,          # [B, L_phn, H]
                EOS_embed           # [B, 1, H]
            ], dim=1)  # [B, 1+Ts+1+L_phn+1, H]
        
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
                 text_start = self.prompt_prefix_embed.size(0) + Ts + self.prompt_suffix_embed.size(0)
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
                 text_start = self.prompt_prefix_embed.size(0) + Ts + self.prompt_suffix_embed.size(0)
                 bos_pos = text_start - 1
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
            # print(f"[DEBUG] LLM Loss: {loss.item()}")
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
                
                if use_prompt and hasattr(self, "prompt_prefix_embed") and hasattr(self, "prompt_suffix_embed"):
                    text_start = self.prompt_prefix_embed.size(0) + Ts + self.prompt_suffix_embed.size(0)
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
                if use_prompt and hasattr(self, "prompt_prefix_embed") and hasattr(self, "prompt_suffix_embed"):
                    text_start = self.prompt_prefix_embed.size(0) + Ts + self.prompt_suffix_embed.size(0)
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
                has_split_prompt = use_prompt and hasattr(self, "prompt_prefix_embed") and hasattr(self, "prompt_suffix_embed")
                
                if has_split_prompt:
                     prefix_embed = self.prompt_prefix_embed
                     suffix_embed = self.prompt_suffix_embed
                     inputs_embeds_inference = torch.cat([
                        prefix_embed.unsqueeze(0).expand(B, -1, -1),
                        Z,
                        suffix_embed.unsqueeze(0).expand(B, -1, -1)
                    ], dim=1)
                
                elif prompt_embed_batch is not None:
                    inputs_embeds_inference = torch.cat([
                        prompt_embed_batch,  # [B, P, H]
                        SEP_embed,          # [B, 1, H]
                        Z,                   # [B, Ts, H]
                        BOS_embed           # [B, 1, H]
                    ], dim=1)  # [B, P+1+Ts+1, H]
                else:
                    inputs_embeds_inference = torch.cat([
                        SEP_embed,          # [B, 1, H]
                        Z,                   # [B, Ts, H]
                        BOS_embed           # [B, 1, H]
                    ], dim=1)  # [B, 1+Ts+1, H]
                
                # if hasattr(self, "llm_norm") and self.llm_norm is not None:
                #     inputs_embeds_inference = self.llm_norm(inputs_embeds_inference)
                if inputs_embeds_inference.dtype != llm_dtype:
                    inputs_embeds_inference = inputs_embeds_inference.to(llm_dtype)
                
                attention_mask_inference = torch.ones(B, inputs_embeds_inference.size(1), dtype=torch.long, device=device)
                
                # print(f"[DEBUG] Generate input shape: {inputs_embeds_inference.shape}")
                # print(f"[DEBUG] max_new_tokens: {L_phn + 10}, BOS_ID: {BOS_ID}, EOS_ID: {EOS_ID}, PAD_ID: {PAD_ID}")
                # print(f"[DEBUG] Using full 128K vocab (no constraint)")
                
                # Generate phoneme tokens with full vocabulary
                # import pdb; pdb.set_trace()
                # pdb.set_trace()
                gen_out = self.modules.LLM.generate(
                    inputs_embeds=inputs_embeds_inference,
                    attention_mask=attention_mask_inference,
                    num_return_sequences=1,
                    pad_token_id=PAD_ID,
                    eos_token_id=EOS_ID,
                    bos_token_id=BOS_ID,
                    do_sample=True,
                    use_cache=True,
                    # top_k = 100,
                    # top_p = 0.9,
                    max_new_tokens=100, 
                    # repetition_penalty=1.1,
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
                return p_ctc, None, {"generated_ids": gen_tokens, "target_phonemes": batch.phn_list_target}, wav_lens

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
                    # pdb.set_trace()
                    
                    loss_ce = ce_loss_fn(
                        ce_logits.reshape(-1, vocab_size),
                        labels.reshape(-1)
                    )
                    
                    # pdb.set_trace()

        # --- 2. Compute Metrics (VALID & TEST) ---
        if stage != sb.Stage.TRAIN:
            # 2.1 CTC Metrics (Common)
            ctc_sequence = sb.decoders.ctc_greedy_decode(
                p_ctc, wav_lens, blank_id=self.hparams.blank_index
            )
            self.ctc_metrics.append(ids, p_ctc, targets, lens_for_ctc, clipped_target_lens)
            self.per_metrics.append(
                ids=ids,
                predict=ctc_sequence,
                target=targets,
                predict_len=None,
                target_len=target_lens,
                ind2lab=self.label_encoder.decode_ndim,
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
                            
                            # Decode using PhonemeTokenMapper for proper phoneme decoding
                            if hasattr(self, "phoneme_mapper") and self.phoneme_mapper is not None:
                                # import pdb; pdb.set_trace()
                                pred_phonemes = self.phoneme_mapper.decode_llm_ids(pred_tokens, skip_special=True)
                                target_phonemes = self.phoneme_mapper.decode_llm_ids(valid_labels, skip_special=True)
                                pred_text = " ".join(pred_phonemes)
                                target_text = " ".join(target_phonemes)
                            else:
                                # Fallback to raw tokenizer decode
                                pred_text = self.hparams.LLM_tokenizer.decode(pred_tokens, skip_special_tokens=True)
                                target_text = self.hparams.LLM_tokenizer.decode(valid_labels, skip_special_tokens=True)
                            # pdb.set_trace()
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
                    
                    # 1. Decode Hypotheses (Generated) using PhonemeTokenMapper
                    if hasattr(self, "phoneme_mapper") and self.phoneme_mapper is not None:
                        # Proper decoding: LLM reserved tokens -> phoneme names
                        hyps = self.phoneme_mapper.decode_to_string(gen_ids, skip_special=True)
                        if isinstance(hyps, str):
                            hyps = [hyps]  # Ensure list format
                    else:
                        # Fallback to raw tokenizer decode (not recommended)
                        hyps = self.hparams.LLM_tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
                    # 2. Decode References (Targets)
                    refs = phn_list_to_seq(batch.phn_list_target)
                    
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

    def on_stage_start(self, stage, epoch):
        "Gets called when a stage (either training, validation, test) starts."
        self.ctc_metrics = self.hparams.ctc_stats()
        self.llm_metrics = self.hparams.llm_stats()  # 添加LLM损失统计
        self.mpd_f1_metrics = MpdStats()  # 添加MPD F1统计
        
        if hasattr(self.modules, "ctc_lin"):
            self.ctc_metrics = self.hparams.ctc_stats()

        if stage != sb.Stage.TRAIN:
            self.per_metrics = self.hparams.per_stats()
            self.llm_per_metrics = self.hparams.per_stats()# 添加LLM PER统计
            self.mpd_f1_metrics = MpdStats()  # 添加MPD F1统计
            
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
            
            mpd_f1 = self.mpd_f1_metrics.summarize("mpd_f1")
            stage_stats["llm_mpd_f1"] = mpd_f1
        
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
                f"{stage.name.lower()}_llm_loss": llm_loss,
                f"{stage.name.lower()}_ctc_loss": ctc_loss if hasattr(self.modules, "ctc_lin") else None,
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
                    m.write("MPD results and stats:\n")
                    self.mpd_f1_metrics.write_stats(m)
                    print(
                        "MPD results and stats written to file",
                        self.hparams.mpd_seq_file,
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
        
        import pdb; pdb.set_trace()
        pretrainer.load_collected()
        import pdb; pdb.set_trace()
        
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

        # ===== NOTE: Initialization moved to _ensure_initialized() =====
        # This allows the model to work in inference mode without calling fit()
        # The lazy initialization happens on first forward pass
        # 
        # Previously initialized here:
        # - self.llm_norm (LayerNorm for LLM hidden states)
        # - self.prompt_embed (soft or text prompt embeddings)
        # 
        # Now these are created in _ensure_initialized() which is called by compute_forward()
        
        # Initialize optimizers (happens during training only)

        self.init_optimizers()

        # ===== Setup Efficient LLM Checkpointing (Save Adapters Only) =====
        # This prevents saving the entire LLM (~8GB+) each epoch
        # Instead, only adapter weights (~10-100MB) are saved
        self._setup_llm_adapter_checkpointing()

        if self.checkpointer is not None:
            self.checkpointer.recover_if_possible(min_key="LLM_PER")
        
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

    def _setup_llm_adapter_checkpointing(self):
        """
        Setup efficient checkpointing for LLM with adapters.
        
        This method ensures that:
        1. Only adapter weights are saved (not entire LLM)
        2. Base LLM is loaded once from original checkpoint
        3. Each epoch only saves ~10-100MB instead of ~8GB+
        
        Supports:
        - PeftModel (LoRA, QLoRA, etc. from Hugging Face PEFT)
        - Custom adapter patterns
        """
        if self.checkpointer is None:
            logger.warning("[Checkpointing] No checkpointer available, skipping adapter setup")
            return
        
        llm_module = getattr(self.modules, 'LLM', None)
        if llm_module is None:
            logger.info("[Checkpointing] No LLM module found, skipping adapter-only setup")
            return
        
        # Handle DDP wrapper
        real_llm = llm_module.module if hasattr(llm_module, 'module') else llm_module
        
        # ===== Case 1: LLM is a PeftModel (LoRA/QLoRA/etc.) =====
        if PeftModel is not None and isinstance(real_llm, PeftModel):
            logger.info("=" * 60)
            logger.info("[Checkpointing] 🚀 Detected PeftModel (LoRA/Adapter)")
            logger.info("[Checkpointing] Will save ONLY adapter weights, NOT entire LLM!")
            
            # Get adapter parameter count for logging
            adapter_params = sum(p.numel() for p in real_llm.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in real_llm.parameters())
            logger.info(f"[Checkpointing] Trainable adapter params: {adapter_params:,} ({adapter_params/1e6:.2f}M)")
            logger.info(f"[Checkpointing] Total LLM params: {total_params:,} ({total_params/1e9:.2f}B)")
            logger.info(f"[Checkpointing] Checkpoint size reduction: ~{(total_params-adapter_params)/total_params*100:.1f}%")
            logger.info("=" * 60)
            
            # Remove LLM from recoverables if it exists (we'll add wrapped version)
            if 'LLM' in self.checkpointer.recoverables:
                del self.checkpointer.recoverables['LLM']
            
            # Add wrapped version that only saves adapters
            wrapped_llm = PeftAdapterRecoverable(llm_module)
            self.checkpointer.add_recoverable("LLM_adapters", wrapped_llm)
            
            logger.info("[Checkpointing] ✅ Added 'LLM_adapters' to checkpointer (adapter-only)")
            return
        
        # ===== Case 2: Check for custom adapter patterns =====
        # Look for common adapter layer patterns
        adapter_patterns = getattr(self.hparams, 'adapter_patterns', ['lora', 'adapter', 'projector'])
        
        # Count adapter vs total parameters
        adapter_param_count = 0
        total_param_count = 0
        
        for name, param in real_llm.named_parameters():
            total_param_count += param.numel()
            if any(pattern in name.lower() for pattern in adapter_patterns):
                adapter_param_count += param.numel()
        
        # If we found adapter-like parameters, set up custom saving
        if adapter_param_count > 0:
            logger.info("=" * 60)
            logger.info("[Checkpointing] 🔧 Detected custom adapter layers")
            logger.info(f"[Checkpointing] Adapter params: {adapter_param_count:,} ({adapter_param_count/1e6:.2f}M)")
            logger.info(f"[Checkpointing] Total LLM params: {total_param_count:,} ({total_param_count/1e9:.2f}B)")
            logger.info("=" * 60)
            
            # Remove full LLM from recoverables
            if 'LLM' in self.checkpointer.recoverables:
                del self.checkpointer.recoverables['LLM']
            
            # Add custom adapter-only recoverable
            wrapped_llm = LLMAdapterOnlyRecoverable(llm_module, adapter_names=adapter_patterns)
            self.checkpointer.add_recoverable("LLM_adapters", wrapped_llm)
            
            logger.info("[Checkpointing] ✅ Added 'LLM_adapters' with custom patterns to checkpointer")
            return
        
        # ===== Case 3: No adapters detected - warn user =====
        logger.warning("=" * 60)
        logger.warning("[Checkpointing] ⚠️  No adapter/LoRA layers detected in LLM!")
        logger.warning("[Checkpointing] The ENTIRE LLM will be saved each checkpoint (~8GB+)")
        logger.warning("[Checkpointing] Consider using LoRA (peft library) for efficient fine-tuning:")
        logger.warning("[Checkpointing]   from peft import get_peft_model, LoraConfig")
        logger.warning("[Checkpointing]   peft_config = LoraConfig(r=16, lora_alpha=32, ...)")
        logger.warning("[Checkpointing]   model = get_peft_model(model, peft_config)")
        logger.warning("=" * 60)
        
        # Also check all recoverables for PeftModels
        for name, obj in list(self.checkpointer.recoverables.items()):
            real_obj = obj.module if hasattr(obj, 'module') else obj
            if PeftModel is not None and isinstance(real_obj, PeftModel):
                logger.info(f"[Checkpointing] Found PeftModel in '{name}', wrapping with adapter-only saver")
                self.checkpointer.recoverables[name] = PeftAdapterRecoverable(obj)

