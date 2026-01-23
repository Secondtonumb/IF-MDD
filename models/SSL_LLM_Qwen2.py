import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
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
    
class SSL_LLM_Qwen2(sb.Brain):
    """QWEN-specialized LLM for phoneme transcription.
    
    This Brain class handles training and inference with QWEN models.
    Key differences from LLAMA version:
    - Uses <|audio_bos|> and <|audio_eos|> tokens to wrap speech embeddings
    - Direct phoneme prediction (no left-shift)
    - Simplified token handling without SEP or BOS tokens for generation
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
        
        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Training tracking
        self.best_valid_loss = float('inf')
        self.best_valid_loss_list = []  # List of (valid_loss, epoch, ckpt_name)
        self.train_stats = {"ctc_loss": [], "ce_loss": [], "total_loss": []}
        self.valid_stats = {"ctc_loss": [], "ce_loss": [], "total_loss": [], "per": []}

    def _ensure_initialized(self):
        """Lazy initialization for components that need LLM to be loaded.
        This ensures inference works even without calling on_fit_start."""
        # Initialize llm_norm if not exists
        llm_handle = self.modules.LLM

        # Unwrap DDP if needed
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

                # Define conversation structure with placeholder
                chat_structure = [
                    {
                        "role": "system", 
                        "content": "You are a phoneme transcriber."
                    },
                    {
                        "role": "user", 
                        "content": f"{PLACEHOLDER}\nTranscribe the preceding speech into CMUdict phonemes."
                    }
                ]

                # Apply chat template
                full_prompt_str = tok.apply_chat_template(
                    chat_structure, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                
                # Split by placeholder
                if PLACEHOLDER not in full_prompt_str:
                    raise ValueError("Chat template processing removed the placeholder!")
                
                prefix_str, suffix_str = full_prompt_str.split(PLACEHOLDER)
                
                # Tokenize prefix and suffix separately
                prefix_tokens = tok(prefix_str, return_tensors="pt", add_special_tokens=False).to(self.device)
                suffix_tokens = tok(suffix_str, return_tensors="pt", add_special_tokens=False).to(self.device)
                
                prefix_ids = prefix_tokens["input_ids"].squeeze(0)
                suffix_ids = suffix_tokens["input_ids"].squeeze(0)

                with torch.no_grad():
                    self.prompt_prefix_embed = embed_fn(prefix_ids)
                    self.prompt_suffix_embed = embed_fn(suffix_ids)
                
                self.prompt_embed = torch.cat([self.prompt_prefix_embed, self.prompt_suffix_embed], dim=0)
                print(f"[Lazy Init] Generated Qwen Prompt via Template.")

    def compute_forward(self, batch, stage):
        """Given an input batch it computes the model forward pass.
        
        QWEN-specific causal LM approach.
        
        Returns:
            - p_ctc: [B, T, 41] - CTC logits 
            - ce_logits: [B, L, vocab_size] or None - LLM logits over vocab
            - ce_targets: dict with target token IDs, or None
            - wav_lens: [B] - sequence lengths
        """
        # Ensure critical components are initialized
        self._ensure_initialized()
        
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "augmentation"):
            wavs = self.hparams.augmentation(wavs)

        # AudioEncoder + Projector
        wav_feats = self.modules.perceived_ssl(wavs)  # [B, T, 1024]
        
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

        if hasattr(self.modules, "projector") and self.modules.projector is not None:
            Z = self.modules.projector(Z) # [B, T, H]  
        
        B, Ts, H = Z.shape
        device = self.device
        tok = self.hparams.LLM_tokenizer
        embed_fn = self.modules.LLM.get_input_embeddings()
        
        # ===== Tokenize phoneme sequences =====
        phn_seq = phn_list_to_seq(batch.phn_list_target)
        phn_tokens = tok(phn_seq, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        
        phn_ids = phn_tokens["input_ids"]
        phn_mask = phn_tokens["attention_mask"]
        L_phn = phn_ids.size(1)
        
        # ===== QWEN special tokens =====
        EOS_ID = tok.eos_token_id
        PAD_ID = tok.pad_token_id if tok.pad_token_id is not None else 0
        
        # Get audio_bos token: <|audio_bos|>
        audio_bos_str = "<|audio_bos|>"
        audio_bos_tokens = tok(audio_bos_str, return_tensors="pt", add_special_tokens=False)
        AUDIO_BOS_ID = audio_bos_tokens["input_ids"][0, 0].item()
        
        # Get audio_eos token: <|audio_eos|>
        audio_eos_str = "<|audio_eos|>"
        audio_eos_tokens = tok(audio_eos_str, return_tensors="pt", add_special_tokens=False)
        AUDIO_EOS_ID = audio_eos_tokens["input_ids"][0, 0].item()
        
        def col_tokens(tok_id):
            """Create [B, 1] tensor of token ID"""
            return torch.full((B, 1), tok_id, dtype=torch.long, device=device)
        
        AUDIO_BOS = col_tokens(AUDIO_BOS_ID)  # [B, 1]
        AUDIO_EOS = col_tokens(AUDIO_EOS_ID)  # [B, 1]
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
        # QWEN Sequence: [prompt?] [audio_bos] [speech] [audio_eos] [phonemes] [EOS]
        
        # Embed phoneme tokens
        phn_embed = embed_fn(phn_ids)  # [B, L_phn, H]
        
        # Embed special tokens
        AUDIO_BOS_embed = embed_fn(AUDIO_BOS)  # [B, 1, H]
        AUDIO_EOS_embed = embed_fn(AUDIO_EOS)  # [B, 1, H]
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
            # QWEN: [prompt] [audio_bos] [speech] [audio_eos] [phonemes] [EOS]
            inputs_embeds = torch.cat([
                prompt_embed_batch,  # [B, P, H]
                AUDIO_BOS_embed,    # [B, 1, H]
                Z,                   # [B, Ts, H]
                AUDIO_EOS_embed,    # [B, 1, H]
                phn_embed,          # [B, L_phn, H]
                EOS_embed           # [B, 1, H]
            ], dim=1)  # [B, P+1+Ts+1+L_phn+1, H]
        else:
            # QWEN: [audio_bos] [speech] [audio_eos] [phonemes] [EOS]
            inputs_embeds = torch.cat([
                AUDIO_BOS_embed,    # [B, 1, H]
                Z,                   # [B, Ts, H]
                AUDIO_EOS_embed,    # [B, 1, H]
                phn_embed,          # [B, L_phn, H]
                EOS_embed           # [B, 1, H]
            ], dim=1)  # [B, 1+Ts+1+L_phn+1, H]
        
        # Align dtype with LLM weights
        llm_dtype = embed_fn.weight.dtype
        if inputs_embeds.dtype != llm_dtype:
            inputs_embeds = inputs_embeds.to(llm_dtype)
        
        # ===== Build attention mask =====
        seq_len = inputs_embeds.size(1)
        
        if stage == sb.Stage.TRAIN:
            # Build attention mask: 1 for valid positions, 0 for padding
            attention_mask = torch.ones(B, seq_len, dtype=torch.long, device=device)
            
            if has_split_prompt:
                text_start = self.prompt_prefix_embed.size(0) + Ts + self.prompt_suffix_embed.size(0)
            else:
                # QWEN: [prompt(P)] [audio_bos(1)] [speech(Ts)] [audio_eos(1)] [phn(L_phn)] [EOS(1)]
                audio_bos_pos = prompt_len
                speech_start = audio_bos_pos + 1
                speech_end = speech_start + Ts
                audio_eos_pos = speech_end
                text_start = audio_eos_pos + 1
            
            for b in range(B):
                num_phn = int(phn_mask[b].sum().item())
                # Mask positions after actual phonemes + EOS
                if num_phn < L_phn:
                    attention_mask[b, text_start + num_phn + 1:] = 0
            
            ignore_idx = -100
            labels = torch.full((B, seq_len), ignore_idx, dtype=torch.long, device=device)
            
            if has_split_prompt:
                text_start = self.prompt_prefix_embed.size(0) + Ts + self.prompt_suffix_embed.size(0)
            else:
                # QWEN: [prompt(P)] [audio_bos(1)] [speech(Ts)] [audio_eos(1)] [phn(L_phn)] [EOS(1)]
                audio_bos_pos = prompt_len
                speech_start = audio_bos_pos + 1
                speech_end = speech_start + Ts
                audio_eos_pos = speech_end
                text_start = audio_eos_pos + 1
            
            text_end = text_start + L_phn
            eos_pos = text_end  # EOS position

            # ===== QWEN Label Construction =====
            # Each position predicts corresponding phoneme (direct, not left-shifted)
            for b in range(B):
                num_phn = int(phn_mask[b].sum().item())
                if num_phn > 0:
                    # Direct prediction: each text position predicts corresponding phoneme
                    labels[b, text_start : text_start + num_phn] = phn_ids[b, :num_phn]
                    # Last phoneme position predicts EOS
                    labels[b, text_start + num_phn] = EOS_ID
            
            # Run forward pass
            llm_out = self.modules.LLM(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=False,
                return_dict=True,
            )
            
            loss = llm_out.loss
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
                # Teacher-forcing (use full target)
                attention_mask = torch.ones(B, seq_len, dtype=torch.long, device=device)
                
                if use_prompt and hasattr(self, "prompt_prefix_embed") and hasattr(self, "prompt_suffix_embed"):
                    text_start = self.prompt_prefix_embed.size(0) + Ts + self.prompt_suffix_embed.size(0)
                else:
                    # QWEN: [prompt(P)] [audio_bos(1)] [speech(Ts)] [audio_eos(1)] [phn(L_phn)] [EOS(1)]
                    audio_bos_pos = prompt_len
                    speech_start = audio_bos_pos + 1
                    speech_end = speech_start + Ts
                    audio_eos_pos = speech_end
                    text_start = audio_eos_pos + 1
                
                for b in range(B):
                    num_phn = int(phn_mask[b].sum().item())
                    if num_phn < L_phn:
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
                
                # Calculate positions
                if use_prompt and hasattr(self, "prompt_prefix_embed") and hasattr(self, "prompt_suffix_embed"):
                    text_start = self.prompt_prefix_embed.size(0) + Ts + self.prompt_suffix_embed.size(0)
                else:
                    audio_bos_pos = prompt_len
                    speech_start = audio_bos_pos + 1
                    speech_end = speech_start + Ts
                    audio_eos_pos = speech_end
                    text_start = audio_eos_pos + 1
                
                # ===== QWEN Label Construction (VALID) =====
                for b in range(B):
                    num_phn = int(phn_mask[b].sum().item())
                    if num_phn > 0:
                        labels[b, text_start : text_start + num_phn] = phn_ids[b, :num_phn]
                        labels[b, text_start + num_phn] = EOS_ID
                
                return p_ctc, ce_logits, {"labels": labels}, wav_lens
            
            elif stage == sb.Stage.TEST:
                # ===== Autoregressive generation (no teacher forcing) =====
                # Only provide: [prompt?] [audio_bos] [speech] [audio_eos]
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
                        AUDIO_BOS_embed,    # [B, 1, H]
                        Z,                   # [B, Ts, H]
                        AUDIO_EOS_embed     # [B, 1, H]
                    ], dim=1)  # [B, P+1+Ts+1, H]
                else:
                    inputs_embeds_inference = torch.cat([
                        AUDIO_BOS_embed,    # [B, 1, H]
                        Z,                   # [B, Ts, H]
                        AUDIO_EOS_embed     # [B, 1, H]
                    ], dim=1)  # [B, 1+Ts+1, H]
                
                if inputs_embeds_inference.dtype != llm_dtype:
                    inputs_embeds_inference = inputs_embeds_inference.to(llm_dtype)
                
                attention_mask_inference = torch.ones(B, inputs_embeds_inference.size(1), dtype=torch.long, device=device)
                
                # Generate phoneme tokens with full vocabulary
                # QWEN does not use bos_token_id for generation
                gen_out = self.modules.LLM.generate(
                    inputs_embeds=inputs_embeds_inference,
                    attention_mask=attention_mask_inference,
                    num_return_sequences=1,
                    pad_token_id=PAD_ID,
                    eos_token_id=EOS_ID,
                    bos_token_id=None,  # QWEN doesn't use BOS for generation
                    do_sample=True,
                    use_cache=True,
                    num_beams=1,
                    temperature=1.2,
                )
                
                gen_tokens = gen_out  # [B, generated_len]
                
                # Return for metrics calculation
                return p_ctc, None, {"generated_ids": gen_tokens, "target_phonemes": batch.phn_list_target}, wav_lens

    def compute_objectives(self, predictions, batch, stage):
        """Compute training objectives: CTC loss + LLM loss"""
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
        loss_ctc = self.hparams.ctc_cost(p_ctc.float(), targets, lens_for_ctc, clipped_target_lens)
        
        # ===== LLM Loss (CrossEntropyLoss) =====
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
                            # Log Probability Metric
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
                canonical = batch.phn_list_canonical
                _, canonical_lens = batch.phn_encoded_canonical
                perceived = batch.phn_list_perceived
                _, perceived_lens = batch.phn_encoded_perceived
                
                if isinstance(ce_targets, dict) and "generated_ids" in ce_targets:
                    gen_ids = ce_targets["generated_ids"]
                    
                    # 1. Decode Hypotheses (Generated)
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
    def inference_batch(self, batch, max_new_tokens=100, do_sample=False, temperature=1.0, top_k=50, top_p=0.9, num_beams=1):
        """
        Pure inference when batch only contains id and sig (no target phonemes).
        
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
        
        # Prepare special tokens - QWEN audio tokens only
        EOS_ID = tok.eos_token_id
        PAD_ID = tok.pad_token_id if tok.pad_token_id is not None else 0
        
        # Get audio_bos and audio_eos tokens
        audio_bos_str = "<|audio_bos|>"
        audio_bos_tokens = tok(audio_bos_str, return_tensors="pt", add_special_tokens=False)
        AUDIO_BOS_ID = audio_bos_tokens["input_ids"][0, 0].item()
        
        audio_eos_str = "<|audio_eos|>"
        audio_eos_tokens = tok(audio_eos_str, return_tensors="pt", add_special_tokens=False)
        AUDIO_EOS_ID = audio_eos_tokens["input_ids"][0, 0].item()
        
        def col_tokens(tok_id):
            return torch.full((B, 1), tok_id, dtype=torch.long, device=device)
        
        AUDIO_BOS = col_tokens(AUDIO_BOS_ID)
        AUDIO_EOS = col_tokens(AUDIO_EOS_ID)
        
        # Embed special tokens
        AUDIO_BOS_embed = embed_fn(AUDIO_BOS)
        AUDIO_EOS_embed = embed_fn(AUDIO_EOS)
        
        # Check prompt type
        use_prompt = getattr(self.hparams, "use_prompt", False)
        has_split_prompt = use_prompt and hasattr(self, "prompt_prefix_embed") and hasattr(self, "prompt_suffix_embed")
        has_prompt = use_prompt and hasattr(self, "prompt_embed") and self.prompt_embed is not None
        
        # Build input embeddings for inference
        if has_split_prompt:
            prefix_embed = self.prompt_prefix_embed
            suffix_embed = self.prompt_suffix_embed
            inputs_embeds = torch.cat([
                prefix_embed.unsqueeze(0).expand(B, -1, -1),
                Z,
                suffix_embed.unsqueeze(0).expand(B, -1, -1)
            ], dim=1)
        elif has_prompt:
            prompt_embed = self.prompt_embed
            prompt_embed_batch = prompt_embed.unsqueeze(0).expand(B, -1, -1)
            inputs_embeds = torch.cat([
                prompt_embed_batch,
                AUDIO_BOS_embed,
                Z,
                AUDIO_EOS_embed
            ], dim=1)
        else:
            # QWEN: [audio_bos] [speech] [audio_eos]
            inputs_embeds = torch.cat([
                AUDIO_BOS_embed,
                Z,
                AUDIO_EOS_embed
            ], dim=1)
        
        # Match LLM dtype
        if inputs_embeds.dtype != llm_dtype:
            inputs_embeds = inputs_embeds.to(llm_dtype)
        
        attention_mask = torch.ones(B, inputs_embeds.size(1), dtype=torch.long, device=device)
        
        # Generate phoneme tokens
        gen_kwargs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "num_return_sequences": 1,
            "pad_token_id": PAD_ID,
            "eos_token_id": EOS_ID,
            "bos_token_id": None,  # QWEN doesn't use BOS for generation
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
            
            # Sort results by ID
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
        self.llm_metrics = self.hparams.llm_stats()
        self.mpd_f1_metrics = MpdStats()
        
        if hasattr(self.modules, "ctc_lin"):
            self.ctc_metrics = self.hparams.ctc_stats()

        if stage != sb.Stage.TRAIN:
            self.per_metrics = self.hparams.per_stats()
            self.llm_per_metrics = self.hparams.per_stats()
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
            stage_stats["ctc_per"] = per_ctc
            stage_stats["llm_per"] = per_llm
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
            self.hparams.train_logger.log_stats(
                stats_meta=stage_stats,
            )
            per_ctc = self.per_metrics.summarize("error_rate")
            mpd_f1 = self.mpd_f1_metrics.summarize("mpd_f1")
            
            # Check if LLM metrics have data before summarizing
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
        """Fit one batch, override to do multiple updates."""

        if self.hparams.auto_mix_prec:
            self.adam_optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)

            # normalize the loss by gradient_accumulation and scale for mixed precision
            self.scaler.scale(loss / self.hparams.gradient_accumulation).backward()
            self.scaler.unscale_(self.adam_optimizer)

            if self.check_gradients(loss):
                if any(p.requires_grad for p in self.adam_optimizer.param_groups[0]['params']):
                    self.scaler.step(self.adam_optimizer)

            self.scaler.update()

        else:
            outputs = self.compute_forward(batch, sb.Stage.TRAIN)
            
            loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            # normalize the loss by gradient_accumulation step
            (loss / self.hparams.gradient_accumulation).backward()

            if self.step % self.hparams.gradient_accumulation == 0:
                if self.check_gradients(loss):
                    self.pretrained_opt_class.step()
                    self.adam_optimizer.step()

                self.pretrained_opt_class.zero_grad()
                self.adam_optimizer.zero_grad()    

        return loss.detach().cpu()

    def on_fit_start(self):
        """Gets called at the beginning of ``fit()``."""
        self._compile()
        self._wrap_distributed()
        self.init_optimizers()
        
        if self.checkpointer is not None:
            self.checkpointer.recover_if_possible(min_key="LLM_PER")
   
    def init_optimizers(self):
        # Collect all trainable parameters
        model_params = list(self.hparams.model.parameters())
        trainable_model_params = list(self.hparams.trainable_model.parameters())
        
        # Add prompt embeddings if they exist AND are learnable parameters
        if hasattr(self, "prompt_embed") and self.prompt_embed is not None:
            if isinstance(self.prompt_embed, torch.nn.Parameter):
                model_params.append(self.prompt_embed)
                print(f"[Optimizer] Added soft prompt embeddings to optimizer")
            else:
                print(f"[Optimizer] Text prompt embeddings are frozen (not added to optimizer)")

        self.adam_optimizer = self.hparams.adam_opt_class(
            trainable_model_params, 
        )
        
        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("adam_opt", self.adam_optimizer)
            self.checkpointer.add_recoverable("tokenizer", self.label_encoder)
