"""
SSL_LLM Model with Interleaved Sequences for Phoneme Transcription and Error Detection.

Architecture:
- Audio Encoder: WavLM (SSL) + Projector
- CTC Branch: Auxiliary task for phoneme recognition
- LLM Branch: Main task - generates interleaved [canonical, perceived, error_label, ...]

Interleaved Format:
  [can_1] [perc_1] [err_1] [can_2] [perc_2] [err_2] ...
  where err ∈ {=, S, D, I} (correct, substitution, deletion, insertion)
"""

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
import logging

logger = logging.getLogger(__name__)

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
        set_peft_model_state_dict(self.model, state_dict)
    
    def to(self, device):
        self.model.to(device)


def phn_list_to_seq(batch):
    """Convert list of phoneme lists to space-separated strings.
    
    Args:
        batch: List of phoneme lists, e.g., [["sil", "aa", "x"], ["sil", "xa", "th"]]
    
    Returns:
        List of space-separated strings: ["sil aa x", "sil xa th"]
    """
    result = []
    for phn_list in batch:
        result.append(" ".join(x for x in phn_list))
    return result


class SSL_LLM_Interleaved_ver2(sb.Brain):
    """LLM-based phoneme transcriber with interleaved error detection."""
    
    def __init__(self, *args, patience=20, **kwargs):
        super().__init__(*args, **kwargs)
        self.patience = patience
        self.no_improve_epochs = 0
        self.best_per_list = []
        self.best_mpd_f1_list = []
        self.best_per = float('inf')
        self.best_mpd_f1 = float('-inf')
        self.last_improved_epoch = 0
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_valid_loss = float('inf')
        self.best_valid_loss_list = []
        self.train_stats = {"ctc_loss": [], "ce_loss": [], "total_loss": []}
        self.valid_stats = {"ctc_loss": [], "ce_loss": [], "total_loss": [], "per": []}
        
        self.phoneme_bias = None
        self.setup_phoneme_mask()

    def setup_phoneme_mask(self):
        """Create a mask that only allows generating phoneme-related tokens."""
        if getattr(self, "phoneme_bias", None) is not None:
            return

        vocab_size = self.modules.LLM.get_input_embeddings().weight.shape[0]
        self.phoneme_bias = torch.full(
            (vocab_size,), float('-10e9'), device=self.device
        )
        valid_tokens = list(range(44))  # Phoneme tokens
        self.phoneme_bias[valid_tokens] = 0
        
    def _ensure_initialized(self):
        """Lazy initialization for components that need LLM to be loaded."""
        llm_handle = self.modules.LLM

        if isinstance(llm_handle, torch.nn.parallel.DistributedDataParallel):
            llm_handle = llm_handle.module
        
        embed_fn = llm_handle.get_input_embeddings()

        if not hasattr(self, "llm_norm") or self.llm_norm is None:
            hidden_size = self.hparams.LLM_DIM
            self.llm_norm = nn.LayerNorm(hidden_size).to(self.device)
            logger.info(f"[Lazy Init] Created llm_norm with hidden_size={hidden_size}")
        
        # Initialize prompt_embed based on prompt_type if needed
        use_prompt = getattr(self.hparams, "use_prompt", False)
        if use_prompt and (not hasattr(self, "prompt_embed") or self.prompt_embed is None):
            prompt_type = getattr(self.hparams, "prompt_type", "soft")
            prompt_len = getattr(self.hparams, "prompt_len", 10)
            hidden_size = self.hparams.LLM_DIM
            
            if prompt_type == "soft":
                init_method = getattr(self.hparams, "prompt_init", "xavier")
                self.prompt_embed = nn.Parameter(
                    torch.zeros(prompt_len, hidden_size, device=self.device)
                )
                if init_method == "xavier":
                    nn.init.xavier_uniform_(self.prompt_embed)
                elif init_method == "normal":
                    nn.init.normal_(self.prompt_embed, mean=0.0, std=0.02)
                
                logger.info(f"[Lazy Init] Created soft prompt: {self.prompt_embed.shape}, init={init_method}")
            
            elif prompt_type in ["text", "discrete"]:
                tok = self.hparams.LLM_tokenizer
                PLACEHOLDER = "<<<SPEECH_EMBEDDING_HERE>>>"

                chat_structure = [
                    {
                        "role": "system", 
                        "content": "You are a phoneme transcriber. Output interleaved sequences of canonical phonemes, perceived phonemes, and error labels."
                    },
                    {
                        "role": "user", 
                        "content": f"{PLACEHOLDER}\nTranscribe the preceding speech into interleaved [canonical] [perceived] [error] sequences."
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
                logger.info(f"[Lazy Init] Generated LLM Prompt via Template for interleaved sequences.")

    def compute_forward(self, batch, stage):
        """Forward pass with interleaved sequence generation.
        
        Returns:
            Tuple of (p_ctc, ce_logits, ce_targets, wav_lens)
        """
        self._ensure_initialized()
        
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "augmentation"):
            wavs = self.hparams.augmentation(wavs)

        # Audio encoding
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
        
        # ===== Tokenize interleaved phoneme sequences =====
        # batch.phn_list_target_interleaved: list of interleaved token lists
        phn_seq = phn_list_to_seq(batch.phn_list_target_interleaved)
        phn_tokens = tok(phn_seq, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        
        phn_ids = phn_tokens["input_ids"]
        phn_mask = phn_tokens["attention_mask"]
        L_phn = phn_ids.size(1)
        
        # ===== Prepare special tokens =====
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
        EOS = col_tokens(EOS_ID)
        
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
        
        # ===== Build input embedding sequence =====
        phn_embed = embed_fn(phn_ids)
        SEP_embed = embed_fn(SEP)
        BOS_embed = embed_fn(BOS)
        EOS_embed = embed_fn(EOS)
        
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
        
        llm_dtype = embed_fn.weight.dtype
        if inputs_embeds.dtype != llm_dtype:
            inputs_embeds = inputs_embeds.to(llm_dtype)
        
        # ===== Build attention mask =====
        seq_len = inputs_embeds.size(1)
        
        if stage == sb.Stage.TRAIN:
            attention_mask = torch.ones(B, seq_len, dtype=torch.long, device=device)
            
            if has_split_prompt:
                text_start = self.prompt_prefix_embed.size(0) + Ts + self.prompt_suffix_embed.size(0)
            else:
                sep_pos = prompt_len
                speech_start = sep_pos + 1
                speech_end = speech_start + Ts
                bos_pos = speech_end
                text_start = bos_pos + 1
            
            for b in range(B):
                num_phn = int(phn_mask[b].sum().item())
                if num_phn < L_phn:
                    attention_mask[b, text_start + num_phn + 1:] = 0
            
            ignore_idx = -100
            labels = torch.full((B, seq_len), ignore_idx, dtype=torch.long, device=device)
            
            if has_split_prompt:
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
            
            llm_out = self.modules.LLM(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=False,
                return_dict=True,
            )
            
            ce_logits = llm_out.logits
            return p_ctc, ce_logits, {"labels": labels}, wav_lens
        
        else:
            # Validation/Test stage
            attention_mask = torch.ones(B, seq_len, dtype=torch.long, device=device)
            
            if use_prompt and hasattr(self, "prompt_prefix_embed") and hasattr(self, "prompt_suffix_embed"):
                text_start = self.prompt_prefix_embed.size(0) + Ts + self.prompt_suffix_embed.size(0)
            else:
                sep_pos = prompt_len
                speech_start = sep_pos + 1
                speech_end = speech_start + Ts
                bos_pos = speech_end
                text_start = bos_pos + 1
            
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
            
            ignore_idx = -100
            labels = torch.full((B, seq_len), ignore_idx, dtype=torch.long, device=device)
            
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

    def compute_objectives(self, predictions, batch, stage):
        """
        Compute separated loss for canonical, perceived, and error components.
        
        Interleaved sequence format: [can_1, perc_1, err_1, can_2, perc_2, err_2, ...]
        
        This method separates the loss computation for:
        1. Canonical phonemes (weight: canonical_weight, default 0.25)
        2. Perceived phonemes (weight: perceived_weight, default 0.25)
        3. Error labels (weight: error_weight, default 0.5) - MOST IMPORTANT
        
        Configuration in hparams:
            canonical_weight: 0.25
            perceived_weight: 0.25
            error_weight: 0.5
        
        Returns combined loss with configurable weights.
        """
        ids = batch.id
        wavs, wav_lens = batch.sig
        targets, target_lens = batch.phn_encoded_target
        
        if len(predictions) == 4:
            p_ctc, ce_logits, ce_targets, lens_for_ctc = predictions
        else:
            raise ValueError(f"Expected 4 values from compute_forward, got {len(predictions)}")
        
        # ===== CTC Loss (auxiliary task on canonical phonemes) =====
        T = p_ctc.size(1)
        clipped_target_lens = torch.minimum(target_lens, torch.full_like(target_lens, T))
        loss_ctc = self.hparams.ctc_cost(p_ctc.float(), targets, lens_for_ctc, clipped_target_lens)
        
        # ===== LLM Loss - Separated by Component =====
        loss_canonical = torch.tensor(0.0, device=self.device)
        loss_perceived = torch.tensor(0.0, device=self.device)
        loss_error = torch.tensor(0.0, device=self.device)
        loss_ce = torch.tensor(0.0, device=self.device)
        
        if stage != sb.Stage.TEST:
            if ce_logits is not None and isinstance(ce_targets, dict) and "labels" in ce_targets:
                labels = ce_targets["labels"]
                if labels is not None:
                    B, seq_len, vocab_size = ce_logits.shape
                    ignore_idx = -100
                    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_idx, reduction='none')
                    
                    # Compute loss for each position (not averaged yet)
                    ce_loss_all = ce_loss_fn(
                        ce_logits.reshape(-1, vocab_size),
                        labels.reshape(-1)
                    ).reshape(B, seq_len)
                    
                    # Create masks for each component (canonical, perceived, error)
                    # Positions in interleaved sequence: [can, perc, err, can, perc, err, ...]
                    # Index mod 3:  0,   1,    2,   3,   4,    5, ...
                    #              can, perc, err, can, perc, err, ...
                    can_mask = torch.zeros_like(labels, dtype=torch.bool)
                    perc_mask = torch.zeros_like(labels, dtype=torch.bool)
                    err_mask = torch.zeros_like(labels, dtype=torch.bool)
                    
                    for b in range(B):
                        # Find valid sequence length (before padding)
                        num_valid = (labels[b] != ignore_idx).sum().item()
                        
                        if num_valid > 0:
                            # For interleaved: positions 0,3,6,... are canonical
                            #                  positions 1,4,7,... are perceived
                            #                  positions 2,5,8,... are errors
                            for pos in range(num_valid):
                                pos_mod = pos % 3
                                if pos_mod == 0:
                                    can_mask[b, pos] = True
                                elif pos_mod == 1:
                                    perc_mask[b, pos] = True
                                elif pos_mod == 2:
                                    err_mask[b, pos] = True
                    
                    # Compute loss for each component (average over valid positions)
                    ce_loss_canonical = ce_loss_all[can_mask].mean() if can_mask.sum() > 0 else torch.tensor(0.0, device=self.device)
                    ce_loss_perceived = ce_loss_all[perc_mask].mean() if perc_mask.sum() > 0 else torch.tensor(0.0, device=self.device)
                    ce_loss_error = ce_loss_all[err_mask].mean() if err_mask.sum() > 0 else torch.tensor(0.0, device=self.device)
                    
                    loss_canonical = ce_loss_canonical
                    loss_perceived = ce_loss_perceived
                    loss_error = ce_loss_error
                    
                    # Combine LLM losses with weights (from hparams)
                    can_weight = getattr(self.hparams, "canonical_weight", 0.25)
                    perc_weight = getattr(self.hparams, "perceived_weight", 0.25)
                    err_weight = getattr(self.hparams, "error_weight", 0.5)  # Error is most important
                    
                    # Normalize weights to sum to 1
                    total_weight = can_weight + perc_weight + err_weight
                    can_weight_norm = can_weight / total_weight
                    perc_weight_norm = perc_weight / total_weight
                    err_weight_norm = err_weight / total_weight
                    
                    loss_ce = (can_weight_norm * loss_canonical + 
                              perc_weight_norm * loss_perceived + 
                              err_weight_norm * loss_error)

        # ===== Metrics =====
        if stage != sb.Stage.TRAIN:
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

        # ===== Combined Loss =====
        ctc_weight = getattr(self.hparams, "ctc_weight", 0.3)
        loss = ctc_weight * loss_ctc + (1.0 - ctc_weight) * loss_ce
        
        # ===== Logging - Separated Components =====
        if stage == sb.Stage.TRAIN:
            self.train_stats.setdefault("ctc_loss", []).append(float(loss_ctc.detach().cpu()))
            self.train_stats.setdefault("canonical_loss", []).append(float(loss_canonical.detach().cpu()))
            self.train_stats.setdefault("perceived_loss", []).append(float(loss_perceived.detach().cpu()))
            self.train_stats.setdefault("error_loss", []).append(float(loss_error.detach().cpu()))
            self.train_stats.setdefault("ce_loss", []).append(float(loss_ce.detach().cpu()))
            self.train_stats.setdefault("total_loss", []).append(float(loss.detach().cpu()))
        else:
            self.valid_stats.setdefault("ctc_loss", []).append(float(loss_ctc.detach().cpu()))
            self.valid_stats.setdefault("canonical_loss", []).append(float(loss_canonical.detach().cpu()))
            self.valid_stats.setdefault("perceived_loss", []).append(float(loss_perceived.detach().cpu()))
            self.valid_stats.setdefault("error_loss", []).append(float(loss_error.detach().cpu()))
            self.valid_stats.setdefault("ce_loss", []).append(float(loss_ce.detach().cpu()))
            self.valid_stats.setdefault("total_loss", []).append(float(loss.detach().cpu()))
        
        return loss
    
    def evaluate_batch(self, batch, stage):
        """Compute loss for validation/test batches."""
        predictions = self.compute_forward(batch, stage=stage)
        loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()

    @torch.no_grad()
    def inference_batch(self, batch, max_new_tokens=150, do_sample=False, temperature=1.0):
        """Inference on a batch with interleaved sequence parsing."""
        self._ensure_initialized()
        
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        ids = batch.id
        
        # Audio encoding
        try:
            Z, _ = self.hparams.audio_encoder_modules(wavs)
        except:
            Z = self.hparams.audio_encoder_modules(wavs)
        
        # CTC prediction
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
        
        if hasattr(self.modules, "projector") and self.modules.projector is not None:
            Z = self.modules.projector(Z)
        
        B, Ts, H = Z.shape
        device = self.device
        tok = self.hparams.LLM_tokenizer
        embed_fn = self.modules.LLM.get_input_embeddings()
        
        llm_dtype = embed_fn.weight.dtype
        
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
        
        SEP_embed = embed_fn(SEP)
        BOS_embed = embed_fn(BOS)
        
        use_prompt = getattr(self.hparams, "use_prompt", False)
        has_split_prompt = use_prompt and hasattr(self, "prompt_prefix_embed") and hasattr(self, "prompt_suffix_embed")
        has_prompt = use_prompt and hasattr(self, "prompt_embed") and self.prompt_embed is not None
        
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
        
        if inputs_embeds.dtype != llm_dtype:
            inputs_embeds = inputs_embeds.to(llm_dtype)
        
        attention_mask = torch.ones(B, inputs_embeds.size(1), dtype=torch.long, device=device)
        
        gen_out = self.modules.LLM.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            pad_token_id=PAD_ID,
            eos_token_id=EOS_ID,
            bos_token_id=BOS_ID,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            use_cache=True,
            max_new_tokens=max_new_tokens,
        )
        
        generated_text = tok.batch_decode(gen_out, skip_special_tokens=True)
        
        # Parse interleaved sequences
        results = []
        for text in generated_text:
            tokens = text.split()
            canonical_preds = []
            perceived_preds = []
            error_preds = []
            
            # Parse triplets: (canonical, perceived, error)
            for i in range(0, len(tokens), 3):
                if i + 2 < len(tokens):
                    can = tokens[i]
                    perc = tokens[i+1]
                    err = tokens[i+2]
                    
                    canonical_preds.append(can)
                    perceived_preds.append(perc)
                    error_preds.append(err)
            
            results.append({
                "canonical": " ".join(canonical_preds),
                "perceived": " ".join(perceived_preds),
                "errors": " ".join(error_preds),
            })
        
        return {
            "ids": ids,
            "results": results,
            "ctc_predictions": ctc_predictions,
        }

    def on_stage_start(self, stage, epoch):
        """Initialize metrics for the stage."""
        self.ctc_metrics = self.hparams.ctc_stats()
        self.llm_metrics = self.hparams.llm_stats()
        
        if stage != sb.Stage.TRAIN:
            self.per_metrics = self.hparams.per_stats()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Log statistics at end of stage."""
        stage_stats = {"loss": stage_loss}
        
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        
        elif stage == sb.Stage.VALID:
            stage_stats["epoch"] = epoch
            per = self.per_metrics.summarize("error_rate")
            stage_stats["per"] = per
            
            self.hparams.train_logger.log_stats(stats_meta=stage_stats)
            
            if self.checkpointer is not None:
                self.checkpointer.save_and_keep_only(
                    meta={"PER": per},
                    num_to_keep=3,
                )
            
            if wandb.run is not None:
                wandb.log({
                    "valid_loss": stage_loss,
                    "valid_per": per,
                    "epoch": epoch,
                })
        
        elif stage == sb.Stage.TEST:
            per = self.per_metrics.summarize("error_rate")
            stage_stats["per"] = per
            
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )

    def init_optimizers(self):
        """Initialize optimizers."""
        trainable_params = list(self.hparams.trainable_model.parameters())
        self.adam_optimizer = self.hparams.adam_opt_class(trainable_params)
        
        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("adam_opt", self.adam_optimizer)
            self.checkpointer.add_recoverable("tokenizer", self.label_encoder)

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

        # Wrap PeftModels in checkpointer to only save adapters
        if self.checkpointer is not None and PeftModel is not None:
            # We iterate over keys to find PeftModels and wrap them
            keys_to_wrap = []
            for name, obj in self.checkpointer.recoverables.items():
                # Handle DDP wrapped objects
                real_obj = obj
                if hasattr(obj, "module"):
                    real_obj = obj.module
                
                if isinstance(real_obj, PeftModel):
                    keys_to_wrap.append(name)
            
            for name in keys_to_wrap:
                print(f"[Checkpointer] Wrapping '{name}' with PeftAdapterRecoverable to save ONLY adapters.")
                self.checkpointer.recoverables[name] = PeftAdapterRecoverable(self.checkpointer.recoverables[name])

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

    def fit_batch(self, batch):
        """Fit a single batch."""
        if self.hparams.auto_mix_prec:
            self.adam_optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                predictions = self.compute_forward(batch, sb.Stage.TRAIN)
                loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)
            
            self.scaler.scale(loss / self.hparams.gradient_accumulation).backward()
            self.scaler.unscale_(self.adam_optimizer)
            
            if any(p.requires_grad for p in self.adam_optimizer.param_groups[0]['params']):
                self.scaler.step(self.adam_optimizer)
            
            self.scaler.update()
        else:
            predictions = self.compute_forward(batch, sb.Stage.TRAIN)
            loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)
            
            (loss / self.hparams.gradient_accumulation).backward()
            
            if self.step % self.hparams.gradient_accumulation == 0:
                self.adam_optimizer.step()
                self.adam_optimizer.zero_grad()
        
        return loss.detach().cpu()
