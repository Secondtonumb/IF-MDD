"""
SSL_LLM with sequential interleaved canonical-perceived-error prediction.

Model predicts interleaved sequences:
Input:  [prompt] [speech] [BOS]
Output: [can_1] [perc_1] [err_1] [can_2] [perc_2] [err_2] ...

where err is one of: "=", "S", "D", "I"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaForCausalLM
import speechbrain as sb
from speechbrain.utils.data_utils import undo_padding
import numpy as np
from tqdm import tqdm
import wandb
from types import SimpleNamespace
import logging

logger = logging.getLogger(__name__)


def phn_list_to_seq(batch):
    """Convert list of phoneme lists to space-separated strings."""
    result = []
    for phn_list in batch:
        result.append(" ".join(x for x in phn_list))
    return result


class SSL_LLM_Sequential_Interleaved(sb.Brain):
    """
    LLM-based phoneme transcriber with sequential error detection.
    
    Predicts interleaved sequences: [canonical] [perceived] [error] [canonical] ...
    """
    
    def __init__(self, *args, patience=20, **kwargs):
        super().__init__(*args, **kwargs)
        self.patience = patience
        self.no_improve_epochs = 0
        self.best_per = float('inf')
        self.best_mpd_f1 = float('-inf')
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_valid_loss = float('inf')
        self.best_valid_loss_list = []
        
        self.train_stats = {
            "loss": [],
            "ctc_loss": [],
            "llm_loss": [],
        }
        self.valid_stats = {
            "loss": [],
            "ctc_loss": [],
            "llm_loss": [],
            "per": [],
        }

    def _ensure_initialized(self):
        """Lazy initialization for LLM components."""
        llm_handle = self.modules.LLM
        
        # Handle DDP wrapping
        if isinstance(llm_handle, torch.nn.parallel.DistributedDataParallel):
            llm_handle = llm_handle.module
        
        embed_fn = llm_handle.get_input_embeddings()

        # Initialize LayerNorm if needed
        if not hasattr(self, "llm_norm") or self.llm_norm is None:
            hidden_size = self.hparams.LLM_DIM
            self.llm_norm = nn.LayerNorm(hidden_size).to(self.device)
            logger.info(f"[Lazy Init] Created llm_norm with hidden_size={hidden_size}")
        
        # Initialize prompt embeddings if needed
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
                
                logger.info(f"[Lazy Init] Created soft prompt: {self.prompt_embed.shape}")
            
            elif prompt_type in ["text", "discrete"]:
                tok = self.hparams.LLM_tokenizer
                PLACEHOLDER = "<<<SPEECH_EMBEDDING_HERE>>>"

                chat_structure = [
                    {
                        "role": "system",
                        "content": """You are an expert Quranic Arabic phoneme transcriber.
Your mission is to convert Quranic recitation into precise phoneme sequences.

Output format: [canonical_phoneme] [perceived_phoneme] [error_label] ...
where error_label is: "=" (correct), "S" (substitution), "D" (deletion), "I" (insertion)"""
                    },
                    {
                        "role": "user",
                        "content": f"""{PLACEHOLDER}

Transcribe this Quranic speech into the interleaved format:
1. [canonical_1] [perceived_1] [error_1]
2. [canonical_2] [perceived_2] [error_2]
... and so on"""
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
                logger.info(f"[Lazy Init] Generated Llama 3 Prompt via Template")

    def compute_forward(self, batch, stage):
        """
        Forward pass for interleaved sequence prediction.
        
        Returns:
            dict with keys: p_ctc, llm_logits, targets, wav_lens
        """
        self._ensure_initialized()
        
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "augmentation"):
            wavs = self.hparams.augmentation(wavs)

        # ===== Audio Encoding =====
        wav_feats = self.modules.perceived_ssl(wavs)  # [B, T, 1024]
        
        try:
            Z, _ = self.hparams.audio_encoder_modules(wavs)
        except:
            Z = self.hparams.audio_encoder_modules(wavs)
        
        # CTC branch
        ctc_logits = self.modules.ctc_lin(Z)
        p_ctc = self.hparams.log_softmax(ctc_logits)

        if hasattr(self.modules, "projector") and self.modules.projector is not None:
            Z = self.modules.projector(Z)  # [B, T, H]
        
        B, Ts, H = Z.shape
        device = self.device
        tok = self.hparams.LLM_tokenizer
        embed_fn = self.modules.LLM.get_input_embeddings()
        
        # ===== Tokenize interleaved target sequence =====
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
        phn_embed = embed_fn(phn_ids)  # [B, L_phn, H]
        
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
        attention_mask = torch.ones(B, seq_len, dtype=torch.long, device=device)
        
        # Mask padding in target sequence
        sep_pos = prompt_len if not has_split_prompt else 0
        speech_start = sep_pos + 1
        speech_end = speech_start + Ts
        bos_pos = speech_end
        text_start = bos_pos + 1
        
        for b in range(B):
            num_phn = int(phn_mask[b].sum().item())
            if num_phn < L_phn:
                attention_mask[b, text_start + num_phn + 1:] = 0
        
        if stage == sb.Stage.TRAIN:
            # ===== Build labels for causal LM =====
            ignore_idx = -100
            labels = torch.full((B, seq_len), ignore_idx, dtype=torch.long, device=device)
            
            for b in range(B):
                num_phn = int(phn_mask[b].sum().item())
                if num_phn > 0:
                    # BOS predicts first token, tokens predict next token
                    labels[b, bos_pos : text_start + num_phn - 1] = phn_ids[b, : num_phn]
                    labels[b, text_start + num_phn - 1] = EOS_ID
            
            # Run forward pass
            llm_out = self.modules.LLM(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=False,
                return_dict=True,
            )
            
            return {
                "p_ctc": p_ctc,
                "llm_loss": llm_out.loss,
                "llm_logits": llm_out.logits,
                "labels": labels,
                "wav_lens": wav_lens,
                "targets": batch.phn_encoded_target,
                "target_lens": batch.phn_encoded_target[1] if isinstance(batch.phn_encoded_target, tuple) else None,
            }
        
        else:
            # Validation/Test stage
            llm_out = self.modules.LLM(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=False,
                return_dict=True,
            )
            
            return {
                "p_ctc": p_ctc,
                "llm_loss": torch.tensor(0.0, device=device),
                "llm_logits": llm_out.logits,
                "labels": None,
                "wav_lens": wav_lens,
                "targets": batch.phn_encoded_target,
                "target_lens": batch.phn_encoded_target[1] if isinstance(batch.phn_encoded_target, tuple) else None,
            }

    def compute_objectives(self, predictions, batch, stage):
        """Compute losses for both CTC and LLM tasks."""
        
        p_ctc = predictions["p_ctc"]
        llm_loss = predictions["llm_loss"]
        wav_lens = predictions["wav_lens"]
        targets = predictions["targets"]
        target_lens = predictions["target_lens"]
        
        # ===== CTC Loss =====
        T = p_ctc.size(1)
        clipped_target_lens = torch.minimum(target_lens, torch.full_like(target_lens, T))
        loss_ctc = self.hparams.ctc_cost(p_ctc.float(), targets, wav_lens, clipped_target_lens)
        
        # ===== Combined Loss =====
        ctc_weight = getattr(self.hparams, "ctc_weight", 0.3)
        loss = ctc_weight * loss_ctc + (1.0 - ctc_weight) * llm_loss
        
        # ===== Logging =====
        if stage == sb.Stage.TRAIN:
            self.train_stats["ctc_loss"].append(float(loss_ctc.detach().cpu()))
            self.train_stats["llm_loss"].append(float(llm_loss.detach().cpu()))
            self.train_stats["loss"].append(float(loss.detach().cpu()))
        else:
            self.valid_stats["ctc_loss"].append(float(loss_ctc.detach().cpu()))
            self.valid_stats["llm_loss"].append(float(llm_loss.detach().cpu()))
            self.valid_stats["loss"].append(float(loss.detach().cpu()))
        
        return loss
    
    def evaluate_batch(self, batch, stage):
        """Compute loss for validation/test batches."""
        predictions = self.compute_forward(batch, stage=stage)
        loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()

    @torch.no_grad()
    def inference_batch(self, batch, max_new_tokens=150, do_sample=False, temperature=1.0):
        """
        Inference on a batch.
        
        Generates interleaved sequences and parses them.
        """
        self._ensure_initialized()
        
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        ids = batch.id
        
        # Audio encoding
        try:
            Z, _ = self.hparams.audio_encoder_modules(wavs)
        except:
            Z = self.hparams.audio_encoder_modules(wavs)
        
        # CTC branch
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
        
        # Special tokens
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
        
        # Check for prompt
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
        
        # Generate
        gen_out = self.modules.LLM.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=PAD_ID,
            eos_token_id=EOS_ID,
            bos_token_id=BOS_ID,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            use_cache=True,
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
                    err = tokens[i+2]  # "=", "S", "D", "I"
                    
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
        """Log statistics at the end of each stage."""
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
        """Initialize at the start of training."""
        self._compile()
        self._wrap_distributed()
        self.init_optimizers()
        
        if self.checkpointer is not None:
            self.checkpointer.recover_if_possible(min_key="PER")

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
