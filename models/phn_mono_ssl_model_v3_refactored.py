import os
import sys
import torch
import torch.nn as nn
import logging
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from mpd_eval_v4 import MpdStats
import librosa
import json
import wandb
import time
import torchaudio
from speechbrain.inference.text import GraphemeToPhoneme
from torch.nn.functional import kl_div
from speechbrain.nnet.loss.guidedattn_loss import GuidedAttentionLoss
import re

# ============================================================================
# Utility Functions
# ============================================================================

def gather_ctc_aligned_reps(encoded, targets, target_lens):
    """Placeholder: gather CTC-aligned representations for each token"""
    B, T, D = encoded.size()
    L = targets.size(1)
    avg_len = T // L
    reps = []
    for i in range(L):
        reps.append(encoded[:, i * avg_len : (i + 1) * avg_len].mean(dim=1))
    return torch.stack(reps, dim=1)  # [B, L, D]

def simplify_phoneme(p):
    """Remove digits from phoneme, e.g., 'AA1' -> 'aa'"""
    return re.sub(r"\d", "", p).lower()


# ============================================================================
# Unified Encoder Manager
# ============================================================================

class EncoderManager(nn.Module):
    """
    Unified encoder manager that handles different encoder types:
    - None: Direct pass-through
    - Linear: Simple linear projection
    - Conformer: Conformer encoder with relative positional encoding
    - Zipformer: Zipformer encoder
    - RVQ: Residual Vector Quantization
    
    Usage:
        encoder = EncoderManager(
            encoder_type='conformer',  # or 'linear', 'zipformer', 'rvq', None
            modules=self.modules,
            hparams=self.hparams,
            device=self.device
        )
        output = encoder(features, wav_lens)
    """
    
    def __init__(self, encoder_type, modules, hparams, device):
        super().__init__()
        self.encoder_type = encoder_type
        self.modules = modules
        self.hparams = hparams
        self.device = device
        
        # Store references to relevant modules
        self.enc = getattr(modules, 'enc', None)
        self.conformer_encoder = getattr(modules, 'ConformerEncoder', None)
        self.zipformer_encoder = getattr(modules, 'ZipformerEncoder', None)
        self.rvq = getattr(modules, 'RVQ', None)
        
    def forward(self, features, wav_lens=None):
        """
        Args:
            features: [B, T, D] SSL features
            wav_lens: [B] relative lengths (optional)
            
        Returns:
            encoded: [B, T, D] encoded features
            extras: dict with additional outputs (commitment_loss, codebook_loss, etc.)
        """
        extras = {}
        x = features
        
        # Step 1: Linear projection (enc)
        if self.enc is not None:
            x = self.enc(x)
        
        # Step 2: Apply specific encoder type
        if self.encoder_type == 'conformer' and self.conformer_encoder is not None:
            from speechbrain.nnet.attention import RelPosEncXL
            pos_emb = RelPosEncXL(emb_dim=self.hparams.dnn_neurons)(x).to(self.device)
            x, _ = self.conformer_encoder(x, pos_embs=pos_emb)
            
        elif self.encoder_type == 'zipformer' and self.zipformer_encoder is not None:
            x = self.zipformer_encoder(x.permute(1, 0, 2))  # [T, B, D]
            x = x.permute(1, 0, 2)  # [B, T, D]
        
        # Step 3: Optional RVQ
        if self.encoder_type == 'rvq' and self.rvq is not None:
            x = x.transpose(1, 2)  # [B, T, D] -> [B, D, T]
            discrete_embeddings, codes, latents, commitment_loss, codebook_loss = self.rvq(x)
            x = discrete_embeddings.transpose(1, 2)  # [B, D, T] -> [B, T, D]
            extras['commitment_loss'] = commitment_loss
            extras['codebook_loss'] = codebook_loss
            extras['discrete_embeddings'] = discrete_embeddings
            extras['codes'] = codes
            
        return x, extras


# ============================================================================
# Unified CTC Loss Manager
# ============================================================================

class CTCLossManager:
    """
    Unified CTC loss manager that handles different loss types:
    - vanilla: Standard CTC loss
    - label_prior: CTC with label priors (CTCLossWithLabelPriors)
    - ottc: Optimal Transport CTC loss
    - crctc: Consistency-Regularized CTC loss
    - crottc: Consistency-Regularized OTTC loss (TODO)
    
    Usage:
        loss_manager = CTCLossManager(
            loss_type='crctc',
            ctc_cost=hparams.ctc_cost,
            hparams=hparams
        )
        loss = loss_manager.compute_loss(p_ctc, targets, wav_lens, target_lens, stage)
    """
    
    def __init__(self, loss_type, ctc_cost, hparams, blank_index=0):
        self.loss_type = loss_type
        self.ctc_cost = ctc_cost
        self.hparams = hparams
        self.blank_index = blank_index
        
        # CR-CTC specific attributes
        if loss_type == 'crctc' or loss_type == 'crottc':
            self.cr_loss_weight = getattr(hparams, "cr_loss_weight", 0.1)
            self.cr_loss_masked_scale = getattr(hparams, "cr_loss_masked_scale", 1.0)
            # CR loss tracking for logging
            self.cr_loss_sum = 0.0
            self.cr_loss_count = 0
            self.ctc_loss_sum = 0.0
            self.ctc_loss_count = 0

            
    def compute_loss(self, p_ctc, targets, wav_lens, target_lens, stage, extras=None):
        """
        Compute CTC loss based on loss type.
        
        Args:
            p_ctc: [B, T, C] or [2*B, T, C] (for CR-CTC) log probabilities
            targets: [B, L] target labels
            wav_lens: [B] or [2*B] (for CR-CTC) relative lengths
            target_lens: [B] relative target lengths
            stage: sb.Stage.TRAIN/VALID/TEST
            extras: dict with additional info (logits, masks, etc.)
            
        Returns:
            loss: scalar loss
            loss_dict: dict with loss components for logging
        """
        from utils.losses.CTCLossWithLabelPriors import CTCLossWithLabelPriors
        from utils.losses.ot_loss import batched_ottc_loss_bucketized
        
        loss_dict = {}
        
        # Determine actual loss type
        if isinstance(self.ctc_cost, CTCLossWithLabelPriors):
            actual_loss_type = 'label_prior'
        elif self.ctc_cost == batched_ottc_loss_bucketized:
            if self.loss_type == 'crottc':
                actual_loss_type = 'crottc'
            else:
                actual_loss_type = 'ottc'
        else:
            actual_loss_type = self.loss_type if self.loss_type else 'vanilla'
        
        # ============ Vanilla CTC ============
        if actual_loss_type == 'vanilla':
            loss = self.ctc_cost(p_ctc, targets, wav_lens, target_lens)
            loss_dict['ctc_loss'] = loss.detach()  # Keep on GPU
            
        # ============ CTC with Label Priors ============
        elif actual_loss_type == 'label_prior':
            p_ctc_transposed = p_ctc.permute(1, 0, 2)  # (B, T, C) -> (T, B, C)
            abs_wav_lens = (wav_lens * p_ctc.shape[1]).to(torch.int32)
            abs_target_lens = (target_lens * targets.shape[1]).to(torch.int32)
            
            step_type_dict = {
                sb.Stage.TRAIN: "train",
                sb.Stage.VALID: "val",
                sb.Stage.TEST: "test"
            }
            
            loss = self.ctc_cost(
                log_probs=p_ctc_transposed,
                targets=targets,
                input_lengths=abs_wav_lens,
                target_lengths=abs_target_lens,
                step_type=step_type_dict[stage]
            )
            loss_dict['ctc_loss'] = loss.detach()  # Keep on GPU
            
        # ============ OTTC Loss ============
        
        elif actual_loss_type == 'crottc':
            if extras and 'logits' in extras and 'weights_logits' in extras and 'weights_labels' in extras:
                logits = extras['logits']
                weights_logits = extras['weights_logits']
                weights_labels = extras['weights_labels']
                # duplicate weights_labels for CR-OTTC
                weights_labels_dup = weights_labels.repeat(2, 1) if weights_labels.dim() > 1 else weights_labels.repeat(2)
                
                labels_mask = (targets != self.blank_index).float()
                # duplicate labels_mask for CR-OTTC
                labels_mask_dup = labels_mask.repeat(2, 1) if labels_mask.dim() > 1 else labels_mask.repeat(2)
                targets_combined = targets.repeat(2, 1) if targets.dim() > 1 else targets.repeat(2)
                target_lens_combined = target_lens.repeat(2)
                one_hot_labels = torch.nn.functional.one_hot(targets_combined, num_classes=logits.shape[-1])
                
                # import pdb; pdb.set_trace()
                
                loss, _, _, _ = self.ctc_cost(
                    x=logits,
                    y=one_hot_labels,
                    a=weights_logits,
                    b=weights_labels_dup,
                    amask=None,
                    bmask=labels_mask_dup,
                    euclidian=False,
                    jsd=False,
                )
                
                loss_dict['ottc_loss'] = loss.detach()  # Keep on GPU
            else:
                # Fallback to vanilla CTC if not in training
                from speechbrain.nnet.losses import ctc_loss
                loss = ctc_loss(p_ctc, targets, wav_lens, target_lens, blank_index=self.blank_index)
                loss_dict['ctc_loss'] = loss.detach()  # Keep on GPU
                
        # ============ CR-CTC Loss ============
        elif actual_loss_type == 'crctc':
            if extras and extras.get('is_crctc_mode', False):
                # CR-CTC mode: p_ctc is [2*B, T, C]
                B = len(targets)
                
                # Duplicate targets
                targets_combined = targets.repeat(2, 1) if targets.dim() > 1 else targets.repeat(2)
                target_lens_combined = target_lens.repeat(2)
                
                # Compute CTC loss on combined batch
                loss_ctc = self.ctc_cost(p_ctc, targets_combined, wav_lens, target_lens_combined)
                
                # Compute CR loss
                # import pdb; pdb.set_trace()
                p_ctc_1, p_ctc_2 = p_ctc.chunk(2, dim=0)
                cr_loss = self._compute_cr_loss(
                    p_ctc_1, p_ctc_2, 
                    wav_lens[:B],
                    None  # Don't use time_mask in loss computation
                )
                
                # Combined loss
                loss = 0.5 * loss_ctc + self.cr_loss_weight * cr_loss
                loss_dict['ctc_loss'] = (0.5 * loss_ctc).detach()  # Keep on GPU
                loss_dict['cr_loss'] = cr_loss.detach()  # Keep on GPU
                
            else:
                # Standard mode
                loss = self.ctc_cost(p_ctc, targets, wav_lens, target_lens)
                loss_dict['ctc_loss'] = loss.detach()  # Keep on GPU
        
        # =========== CR-OTTC Loss ============
        elif actual_loss_type == 'ottc':
            if extras and 'logits' in extras and 'weights_logits' in extras and 'weights_labels' in extras:
                logits = extras['logits']
                weights_logits = extras['weights_logits']
                weights_labels = extras['weights_labels']
                
                labels_mask = (targets != self.blank_index).float()
                one_hot_labels = torch.nn.functional.one_hot(targets, num_classes=logits.shape[-1])
                
                loss, _, _, _ = self.ctc_cost(
                    x=logits,
                    y=one_hot_labels,
                    a=weights_logits,
                    b=weights_labels,
                    amask=None,
                    bmask=labels_mask,
                    euclidian=False,
                    jsd=False,
                )
                
                
                if extras and extras.get('is_crctc_mode', False):
                    # Compute CR loss
                    B = len(targets)
                    p_ctc_1, p_ctc_2 = p_ctc.chunk(2, dim=0)
                    cr_loss = self._compute_cr_loss(
                        p_ctc_1, p_ctc_2, 
                        wav_lens[:B],
                        None  # Don't use time_mask in loss computation
                    )
                    # Combined loss
                    loss = 0.5 * loss + self.cr_loss_weight * cr_loss
                    loss_dict['cr_loss'] = cr_loss.detach()  # Keep on GPU
                    
                loss_dict['ottc_loss'] = loss.detach()  # Keep on GPU
                    
            else:
                # Fallback to vanilla CTC if not in training
                from speechbrain.nnet.losses import ctc_loss
                loss = ctc_loss(p_ctc, targets, wav_lens, target_lens, blank_index=self.blank_index)
                loss_dict['ctc_loss'] = loss.detach()  # Keep on GPU
            
        else:
            # Default to vanilla
            loss = self.ctc_cost(p_ctc, targets, wav_lens, target_lens)
            loss_dict['ctc_loss'] = loss.detach()  # Keep on GPU
            
        return loss, loss_dict
    
    def _compute_cr_loss(self, p_ctc_1, p_ctc_2, wav_lens, time_mask=None):
        """Compute Consistency Regularization loss for CR-CTC (GPU-optimized)"""
        B, T, C = p_ctc_1.shape
        device = p_ctc_1.device
        
        # Symmetric KL divergence
        kl_1_to_2 = torch.nn.functional.kl_div(
            input=p_ctc_1, target=p_ctc_2.detach(),
            reduction="none", log_target=True
        )
        kl_2_to_1 = torch.nn.functional.kl_div(
            input=p_ctc_2, target=p_ctc_1.detach(),
            reduction="none", log_target=True
        )
        cr_loss = kl_1_to_2 + kl_2_to_1  # [B, T, C]
        
        # Length mask to ignore padding (optimized)
        abs_lens = (wav_lens * T).long()
        # Create arange once and reuse
        time_indices = torch.arange(T, device=device, dtype=abs_lens.dtype).unsqueeze(0)
        length_mask = (time_indices >= abs_lens.unsqueeze(1)).unsqueeze(-1)
        
        cr_loss = cr_loss.masked_fill(length_mask, 0.0)
        
        # Average over valid positions (avoid creating large intermediate tensors)
        num_valid = (T * C * B) - length_mask.sum()
        cr_loss = cr_loss.sum() / torch.clamp(num_valid, min=1)
        
        return cr_loss


# ============================================================================
# Base Model with Unified Architecture
# ============================================================================

class PhnMonoSSLModel(sb.Brain):
    """
    Unified base model for phoneme-level mispronunciation detection.
    
    Architecture: SSL -> Encoder -> CTC
    
    Supports:
    - Encoder types: None, Linear, Conformer, Zipformer, RVQ
    - CTC loss types: Vanilla, Label Prior, OTTC, CR-CTC
    - Configurable via hparams without code changes
    """
    
    def __init__(self, *args, patience=20, **kwargs):
        super().__init__(*args, **kwargs)
        self.patience = patience
        self.no_improve_epochs = 0
        self.best_per_list = []
        self.best_mpd_f1_list = []
        self.best_per = float('inf')
        self.best_mpd_f1 = float('-inf')
        self.last_improved_epoch = 0
        self.best_valid_loss = float('inf')
        self.best_valid_loss_list = []
        
        # CR-CTC loss tracking (will be reset each epoch)
        self.cr_loss_sum = 0.0
        self.cr_loss_count = 0
        self.ctc_loss_sum = 0.0
        self.ctc_loss_count = 0
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if getattr(self.modules, "perceived_ssl", None) is not None:
            self.modules.perceived_ssl.to(self.device)
        if getattr(self.modules, "canonical_ssl", None) is not None:
            self.modules.canonical_ssl.to(self.device)
        
        # Initialize unified components
        self._init_encoder_manager()
        self._init_loss_manager()
        
    def _init_encoder_manager(self):
        """Initialize encoder manager based on hparams"""
        encoder_type = getattr(self.hparams, 'encoder_type', None)
        
        # Auto-detect encoder type if not specified
        if encoder_type is None:
            if getattr(self.modules, 'RVQ', None) is not None:
                encoder_type = 'rvq'
            elif getattr(self.modules, 'ConformerEncoder', None) is not None:
                encoder_type = 'conformer'
            elif getattr(self.modules, 'ZipformerEncoder', None) is not None:
                encoder_type = 'zipformer'
            elif getattr(self.modules, 'enc', None) is not None:
                encoder_type = 'linear'
            else:
                encoder_type = None
        
        self.encoder_manager = EncoderManager(
            encoder_type=encoder_type,
            modules=self.modules,
            hparams=self.hparams,
            device=self.device
        )
        
    def _init_loss_manager(self):
        """Initialize loss manager based on hparams"""
        loss_type = getattr(self.hparams, 'ctc_loss_type', 'vanilla')
        
        # import pdb; pdb.set_trace()
        
        self.loss_manager = CTCLossManager(
            loss_type=loss_type,
            ctc_cost=self.hparams.ctc_cost,
            hparams=self.hparams,
            blank_index=self.hparams.blank_index
        )
    
    def create_attention_mask_from_input_sequence(self, input_sequence):
        """Create attention mask from input sequence lengths (optimized for GPU)"""
        batch_size = input_sequence.size(0)
        max_len = input_sequence.max().item()  # Single sync point
        # Create arange directly on GPU
        indices = torch.arange(max_len, device=self.device, dtype=input_sequence.dtype).unsqueeze(0)
        attention_mask = (indices < input_sequence.unsqueeze(1)).float()
        return attention_mask
    
    def check_gradients(self, loss):
        """Check if gradients are finite"""
        if not torch.isfinite(loss):
            print("Warning: loss is not finite, skipping step")
            return False
        return True
    
    def compute_forward(self, batch, stage):
        """
        Unified forward pass supporting all configurations.
        
        Returns:
            Depending on configuration, returns:
            - (p_ctc, wav_lens) for vanilla
            - (p_ctc, wav_lens, commitment_loss, codebook_loss) for RVQ
            - (p_ctc, logits, weights_logits, weights_labels, wav_lens) for OTTC
            - (p_ctc, wav_lens, time_mask, is_crctc_mode) for CR-CTC
        """
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        
        # check if OTTC mode is enabled
        use_ottc = (
            self.loss_manager.loss_type == 'ottc' and 
            stage == sb.Stage.TRAIN and
            getattr(self.hparams, "use_ottc", True)
        )
        
        # Check if CR-CTC mode is enabled (requires augmentation)
        use_crctc = (
            self.loss_manager.loss_type == 'crctc' and 
            stage == sb.Stage.TRAIN and
            getattr(self.hparams, "use_crctc", True) and
            hasattr(self.hparams, "augmentation")  # CR-CTC requires augmentation
        )

        # Check if CT-OTTC mode is enabled
        use_crottc = (
            self.loss_manager.loss_type == 'crottc' and
            stage == sb.Stage.TRAIN and
            getattr(self.hparams, "use_ottc", True) and
            hasattr(self.hparams, "augmentation")  # OTTC requires augmentation
        )
        
        # Apply augmentation in training,
        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "speed_augmentation"):
                wavs = self.hparams.speed_augmentation(wavs)
            
            # Essential Augmentation for for CR-CTC / CR-OTTC
            if use_crctc or use_crottc:
                # CR-CTC: Apply augmentation twice to get two different views
                wavs_1, wav_lens_1 = self.hparams.augmentation.forward(wavs, lengths=wav_lens)
                wavs_2, wav_lens_2 = self.hparams.augmentation.forward(wavs, lengths=wav_lens)
                
                # Extract SSL features for both augmented versions
                feats_1 = self.modules.perceived_ssl(wavs_1)
                feats_2 = self.modules.perceived_ssl(wavs_2)
                
                # Concatenate both versions: [2*B, T, D]
                feats = torch.cat([feats_1, feats_2], dim=0)
                wav_lens = wav_lens.repeat(2)
            else:
                # Standard augmentation (non-CR-CTC)
                if hasattr(self.hparams, "augmentation"):
                    wavs = self.hparams.augmentation(wavs)
                
                # Extract SSL features
                feats = self.modules.perceived_ssl(wavs)
        else:
            # Extract SSL features (no augmentation)
            feats = self.modules.perceived_ssl(wavs)
        
        # Encode features
        x, encoder_extras = self.encoder_manager(feats, wav_lens)
        
        # CTC output layer
        logits = self.modules.ctc_lin(x)
        p_ctc = self.hparams.log_softmax(logits)
        
        # Handle OTTC-specific outputs
        if use_crottc or use_ottc:
            if hasattr(self.modules, "lm_weight") and stage != sb.Stage.TEST:
                targets, target_lens = batch.phn_encoded_target
                labels_mask = (targets != self.hparams.blank_index).float()
                
                weights_logits = self.modules.lm_weight(x)
                lens_abs = (wav_lens * feats.shape[-2]).int()
                output_mask = self.create_attention_mask_from_input_sequence(lens_abs)
                
                import torch.nn.functional as F
                # Optimize: use in-place operations and avoid unnecessary copies
                weights_logits = weights_logits.squeeze()
                weights_logits = weights_logits.masked_fill(output_mask == 0, -torch.inf)
                weights_logits = F.softmax(weights_logits, dim=-1)
                
                # Optimize: use clamp to avoid division by zero without sync
                label_sums = labels_mask.sum(dim=1, keepdim=True).clamp(min=1e-8)
                weights_labels = labels_mask / label_sums
                
                return p_ctc, logits, weights_logits, weights_labels, wav_lens
        
        # Handle RVQ outputs
        if 'commitment_loss' in encoder_extras:
            return (p_ctc, wav_lens, 
                    encoder_extras['commitment_loss'], 
                    encoder_extras['codebook_loss'])
        
        # Handle CR-CTC outputs
        if use_crctc:
            return p_ctc, wav_lens, None, True  # True indicates CR-CTC mode
        
        # Standard output
        return p_ctc, wav_lens
    
    def compute_objectives(self, predictions, batch, stage):
        """Unified objective computation"""
        # Parse predictions based on type
        if len(predictions) == 2:
            p_ctc, wav_lens = predictions
            extras = {}
        elif len(predictions) == 4:
            if isinstance(predictions[3], bool):  # CR-CTC
                p_ctc, wav_lens, time_mask, is_crctc_mode = predictions
                # import pdb; pdb.set_trace()
                extras = {'time_mask': time_mask, 'is_crctc_mode': is_crctc_mode}
            else:  # RVQ
                p_ctc, wav_lens, commitment_loss, codebook_loss = predictions
                extras = {'commitment_loss': commitment_loss, 'codebook_loss': codebook_loss}
        elif len(predictions) == 5:  # OTTC or CR-OTTC
            p_ctc, logits, weights_logits, weights_labels, wav_lens = predictions
            extras = {'logits': logits, 'weights_logits': weights_logits, 'weights_labels': weights_labels}
        else:
            raise ValueError(f"Unexpected predictions format: {len(predictions)} elements")
        
        # Get targets
        ids = batch.id
        targets, target_lens = batch.phn_encoded_target
        
        # Handle target selection (canonical/perceived/target)
        if stage != sb.Stage.TRAIN:
            canonicals, canonical_lens = batch.phn_encoded_canonical
            perceiveds, perceived_lens = batch.phn_encoded_perceived
            
            training_target = getattr(self.hparams, 'training_target', 'target')
            if training_target == "canonical":
                targets = canonicals
                target_lens = canonical_lens
            elif training_target == "perceived":
                targets = perceiveds
                target_lens = perceived_lens
        
        # Compute loss using loss manager
        loss, loss_dict = self.loss_manager.compute_loss(
            p_ctc, targets, wav_lens, target_lens, stage, extras
        )
        
        # Track CR-CTC / CR-OTTC losses for epoch-level logging (keep on GPU)
        if stage == sb.Stage.TRAIN:
            if 'cr_loss' in loss_dict:
                self.cr_loss_sum += loss_dict['cr_loss'].item()  # Only sync when accumulating
                self.cr_loss_count += 1
            if 'ctc_loss' in loss_dict:
                self.ctc_loss_sum += loss_dict['ctc_loss'].item()  # Only sync when accumulating
                self.ctc_loss_count += 1
        
        # Add RVQ losses if present
        if 'commitment_loss' in extras:
            loss = loss + extras['commitment_loss'] + extras['codebook_loss']
            loss_dict['commitment_loss'] = extras['commitment_loss'].detach()  # Keep on GPU
            loss_dict['codebook_loss'] = extras['codebook_loss'].detach()  # Keep on GPU
        
        # Evaluation metrics
        if stage != sb.Stage.TRAIN:
            # For CR-CTC, use first half for evaluation
            if extras.get('is_crctc_mode', False):
                p_ctc_eval = p_ctc[:len(ids)]
                wav_lens_eval = wav_lens[:len(ids)]
            else:
                p_ctc_eval = p_ctc
                wav_lens_eval = wav_lens
            
            sequence = sb.decoders.ctc_greedy_decode(
                p_ctc_eval, wav_lens_eval, blank_id=self.hparams.blank_index
            )
            
            # CTC metrics
            from utils.losses.CTCLossWithLabelPriors import CTCLossWithLabelPriors
            if isinstance(self.hparams.ctc_cost, CTCLossWithLabelPriors):
                try:
                    self.ctc_metrics.append(
                        ids,
                        log_probs=p_ctc_eval.permute(1, 0, 2),
                        targets=targets,
                        input_lengths=(wav_lens_eval * p_ctc_eval.shape[1]).to(torch.int32),
                        target_lengths=(target_lens * targets.shape[1]).to(torch.int32)
                    )
                except:
                    self.ctc_metrics.append(ids, p_ctc_eval, targets, wav_lens_eval, target_lens)
            else:
                self.ctc_metrics.append(ids, p_ctc_eval, targets, wav_lens_eval, target_lens)
            
            # PER metrics
            self.per_metrics.append(
                ids=ids,
                predict=sequence,
                target=targets,
                predict_len=None,
                target_len=target_lens,
                ind2lab=self.label_encoder.decode_ndim,
            )
            
            # MPD metrics
            canonicals, canonical_lens = batch.phn_encoded_canonical
            perceiveds, perceived_lens = batch.phn_encoded_perceived
            self.mpd_metrics.append(
                ids=ids,
                predict=sequence,
                canonical=canonicals,
                perceived=perceiveds,
                predict_len=None,
                canonical_len=canonical_lens,
                perceived_len=perceived_lens,
                ind2lab=self.label_encoder.decode_ndim,
            )
        
        return loss
    
    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        predictions = self.compute_forward(batch, stage=stage)
        loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()
    
    def _load_token_names(self):
        """Load token names from label encoder file"""
        token_names = {}
        label_encoder_path = os.path.join(self.hparams.save_folder, "label_encoder.txt")
        if os.path.exists(label_encoder_path):
            with open(label_encoder_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and '=>' in line:
                        try:
                            name, idx = line.split('=>')
                            name = name.strip().strip("'\"")
                            idx = idx.strip()
                            token_names[int(idx)] = name
                        except:
                            pass
        return token_names
    
    def on_stage_start(self, stage, epoch):
        """Gets called when a stage starts"""
        self.ctc_metrics = self.hparams.ctc_stats()
        if hasattr(self.hparams, "augmentation"):
            self.modules.perceived_ssl.model.config.apply_spec_augment = True
        
        # Reset CR-CTC loss tracking at the start of each training epoch
        if stage == sb.Stage.TRAIN:
            self.cr_loss_sum = 0.0
            self.cr_loss_count = 0
            self.ctc_loss_sum = 0.0
            self.ctc_loss_count = 0
        
        if stage != sb.Stage.TRAIN:
            self.per_metrics = self.hparams.per_stats()
            self.mpd_metrics = MpdStats()
    
    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch"""
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
            
            # Compute average CR-CTC losses for the epoch
            if self.loss_manager.loss_type == 'crctc' or self.loss_manager.loss_type == 'crottc':
                # import pdb; pdb.set_trace()
                self.avg_cr_loss = (self.cr_loss_sum / max(1, self.cr_loss_count)
                                   if self.cr_loss_count > 0 else 0.0)
                self.avg_ctc_loss_train = (self.ctc_loss_sum / max(1, self.ctc_loss_count)
                                          if self.ctc_loss_count > 0 else 0.0)
        else:
            per = self.per_metrics.summarize("error_rate")
            mpd_f1 = self.mpd_metrics.summarize("mpd_f1")
        
        if stage == sb.Stage.VALID:
            # Prepare train stats
            train_stats = {"loss": self.train_loss}
            if self.loss_manager.loss_type == 'crctc' or self.loss_manager.loss_type == 'crottc':
                train_stats["ctc_loss"] = getattr(self, 'avg_ctc_loss_train', 0.0)
                train_stats["cr_loss"] = getattr(self, 'avg_cr_loss', 0.0)
            
            # Prepare valid stats
            valid_stats = {
                "loss": stage_loss,
                "ctc_loss": self.ctc_metrics.summarize("average"),
                "PER": per,
                "mpd_f1": mpd_f1
            }
            
            self.hparams.train_logger.log_stats(
                stats_meta={
                    "epoch": epoch,
                    "lr_adam": self.adam_optimizer.param_groups[0]["lr"],
                    "lr_pretrained": self.pretrained_opt_class.param_groups[0]["lr"],
                },
                train_stats=train_stats,
                valid_stats=valid_stats,
            )
            
            # Save checkpoints
            improved = False
            if per < self.best_per:
                self.best_per = per
                improved = True
            
            if mpd_f1 > self.best_mpd_f1:
                self.best_mpd_f1 = mpd_f1
                improved = True
            
            ckpt_name = f"{epoch:03d}_PER_{per:.4f}_F1_{mpd_f1:.4f}.ckpt"
            max_save_models = getattr(self.hparams, 'max_save_models', 3)
            self.checkpointer.save_and_keep_only(
                meta={"PER": per, "mpd_f1": mpd_f1, "epoch": epoch},
                name=ckpt_name,
                num_to_keep=max_save_models,
                importance_keys=[lambda ckpt: (-ckpt.meta["PER"], ckpt.meta["mpd_f1"])]
            )
            
            # Early stopping
            if stage_loss < self.best_valid_loss or len(self.best_valid_loss_list) < 10:
                if stage_loss < self.best_valid_loss:
                    self.best_valid_loss = stage_loss
                    improved = True
                self.best_valid_loss_list.append((stage_loss, epoch, ckpt_name))
                self.best_valid_loss_list.sort(key=lambda x: x[0])
                self.best_valid_loss_list = self.best_valid_loss_list[:10]
            
            if improved:
                self.no_improve_epochs = 0
                self.last_improved_epoch = epoch
            else:
                self.no_improve_epochs += 1
            
            # Wandb logging
            wandb_dict = {
                "epoch": epoch,
                "train_loss": self.train_loss,
                "valid_loss": stage_loss,
                "valid_ctc_loss": self.ctc_metrics.summarize("average"),
                "PER": per,
                "mpd_f1": mpd_f1,
            }
            
            # Add CR-CTC specific metrics
            if self.loss_manager.loss_type == 'crctc' or self.loss_manager.loss_type == 'crottc':
                wandb_dict["train_ctc_loss"] = getattr(self, 'avg_ctc_loss_train', 0.0)
                wandb_dict["train_cr_loss"] = getattr(self, 'avg_cr_loss', 0.0)
            
            wandb.log(wandb_dict, step=epoch)
            
            # Early stop if patience exceeded
            if self.no_improve_epochs >= self.patience:
                print(f"Early stopping at epoch {epoch}, no improvement for {self.patience} epochs.")
                raise KeyboardInterrupt("Early stopping triggered")
        
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats={"loss": stage_loss, "PER": per, "mpd_f1": mpd_f1},
            )
            
            with open(self.hparams.per_file, "w") as w:
                self.per_metrics.write_stats(w)
            
            with open(self.hparams.mpd_file, "w") as m:
                self.mpd_metrics.write_stats(m)
    
    def fit_batch(self, batch):
        """Fit one batch"""
        if self.hparams.auto_mix_prec:
            self.pretrained_opt_class.zero_grad()
            self.adam_optimizer.zero_grad()
            
            with torch.amp.autocast("cuda"):
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            
            self.scaler.scale(loss / self.hparams.gradient_accumulation).backward()
            self.scaler.unscale_(self.pretrained_opt_class)
            self.scaler.unscale_(self.adam_optimizer)
            
            if self.check_gradients(loss):
                if any(p.requires_grad for p in self.pretrained_opt_class.param_groups[0]['params']):
                    self.scaler.step(self.pretrained_opt_class)
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
    
    def init_optimizers(self):
        """Initialize optimizers"""
        self.adam_optimizer = self.hparams.adam_opt_class(
            self.hparams.model.parameters(),
        )
        self.pretrained_opt_class = self.hparams.pretrained_opt_class(
            self.modules.perceived_ssl.parameters(),
        )
        
        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("adam_opt", self.adam_optimizer)
            self.checkpointer.add_recoverable("pretrained_opt", self.pretrained_opt_class)
            self.checkpointer.add_recoverable("tokenizer", self.label_encoder)
    
    def on_fit_start(self):
        """Gets called at the beginning of fit()"""
        self._compile()
        self._wrap_distributed()
        self.init_optimizers()
        
        # Load pretrained components if specified
        if getattr(self.hparams, 'load_pretrained_components', False):
            pretrained_path = getattr(self.hparams, 'pretrained_model_path', '')
            components = getattr(self.hparams, 'components_to_load', ['ssl', 'enc', "ctc_head"])
            freeze_loaded = getattr(self.hparams, 'freeze_loaded_components', True)
            
            if pretrained_path and os.path.exists(pretrained_path):
                try:
                    self.load_pretrained_components(
                        checkpoint_path=pretrained_path,
                        components_to_load=components,
                        freeze_loaded=freeze_loaded
                    )
                except Exception as e:
                    print(f"❌ Failed to load pretrained components: {e}")
        elif self.checkpointer is not None:
            self.checkpointer.recover_if_possible(min_key="PER")
    
    def load_pretrained_components(self, checkpoint_path, components_to_load=None, freeze_loaded=True):
        """Load specific components from a pretrained model"""
        if components_to_load is None:
            components_to_load = ['ssl']
        
        print(f"\n🔄 Loading pretrained components from: {checkpoint_path}")
        print(f"   Components to load: {components_to_load}")
        
        from speechbrain.utils.parameter_transfer import Pretrainer
        
        pretrainer = Pretrainer(
            collect_in=self.hparams.pretrained_model_path,
            loadables={
                "perceived_ssl": self.modules.perceived_ssl,
                "model": self.hparams.model,
            },
            paths={
                "perceived_ssl": "perceived_ssl.ckpt",
                "model": "model.ckpt",
            },
        )
        
        paths = pretrainer.collect_files(default_source=self.hparams.pretrained_model_path)
        pretrainer.load_collected()
        
        # Freeze loaded components if requested
        if freeze_loaded:
            for component in components_to_load:
                if component == 'ssl':
                    for param in self.modules.perceived_ssl.parameters():
                        param.requires_grad = False
                    print("   🔒 SSL model frozen")
                elif component == 'enc':
                    if hasattr(self.modules, 'enc'):
                        for param in self.modules.enc.parameters():
                            param.requires_grad = False
                        print("   🔒 Encoder projection frozen")
                elif component == 'ctc_head':
                    for param in self.modules.ctc_lin.parameters():
                        param.requires_grad = False
                    print("   🔒 CTC head frozen")


# ============================================================================
# Specialized Models (继承重构后的基类)
# ============================================================================

class PhnMonoSSLModel_DualCTCHead(PhnMonoSSLModel):
    """
    Dual CTC heads for perceived and canonical phonemes.
    - Perceived feature from middle SSL layers
    - Canonical from last layer
    """
    
    def compute_forward(self, batch, stage):
        """Dual-head forward pass"""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        
        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs)
        
        # Extract features from multiple SSL layers
        feats = self.modules.perceived_ssl(wavs)
        assert feats.dim() == 4  # (B, L, T, D)
        
        feats_cano = feats[self.hparams.canonical_ssl_emb_layer]
        feats_perc = feats[self.hparams.preceived_ssl_emb_layer]
        
        # Encode separately or shared
        if (self.hparams.preceived_ssl_emb_layer == self.hparams.canonical_ssl_emb_layer and
            self.hparams.shareenc and getattr(self.modules, "enc_cano", None) is None):
            x_perc = self.modules.enc(feats_perc)
            x_cano = x_perc
        else:
            try:
                x_cano = self.modules.enc_cano(feats_cano)
                x_perc = self.modules.enc(feats_perc)
            except:
                if self.hparams.preceived_ssl_emb_layer != self.hparams.canonical_ssl_emb_layer:
                    raise ValueError("Please define a separate encoder for Canonical feature")
        
        # Apply Conformer if exists
        if getattr(self.modules, "ConformerEncoder", None) is not None:
            from speechbrain.nnet.attention import RelPosEncXL
            pos_emb = RelPosEncXL(emb_dim=self.hparams.dnn_neurons)(x_perc).to(self.device)
            x_perc, _ = self.modules.ConformerEncoder(x_perc, pos_embs=pos_emb)
        
        # Dual CTC heads
        logits_cano = self.modules.ctc_lin(x_cano)
        p_ctc_cano = self.hparams.log_softmax(logits_cano)
        
        logits_perc = self.modules.ctc_perc_lin(x_perc)
        p_ctc_perc = self.hparams.log_softmax(logits_perc)
        
        return p_ctc_cano, p_ctc_perc, wav_lens
    
    def compute_objectives(self, predictions, batch, stage):
        """Dual CTC loss computation"""
        p_ctc_cano, p_ctc_perc, wav_lens = predictions
        ids = batch.id
        targets, target_lens = batch.phn_encoded_target
        canonicals, canonical_lens = batch.phn_encoded_canonical
        perceiveds, perceived_lens = batch.phn_encoded_perceived
        
        # Dual losses
        loss_cano = self.hparams.ctc_cost(p_ctc_cano, canonicals, wav_lens, canonical_lens)
        loss_perc = self.hparams.ctc_cost(p_ctc_perc, perceiveds, wav_lens, perceived_lens)
        
        loss = loss_cano + loss_perc
        
        # Evaluation metrics
        if stage != sb.Stage.TRAIN:
            sequence_cano = sb.decoders.ctc_greedy_decode(
                p_ctc_cano, wav_lens, blank_id=self.hparams.blank_index
            )
            sequence_perc = sb.decoders.ctc_greedy_decode(
                p_ctc_perc, wav_lens, blank_id=self.hparams.blank_index
            )
            
            self.ctc_metrics.append(ids, p_ctc_cano, canonicals, wav_lens, canonical_lens)
            self.per_metrics.append(
                ids=ids, predict=sequence_cano, target=canonicals,
                predict_len=None, target_len=canonical_lens,
                ind2lab=self.label_encoder.decode_ndim,
            )
            
            self.mpd_metrics.append(
                ids=ids, predict=sequence_perc,
                canonical=canonicals, perceived=perceiveds,
                predict_len=None,
                canonical_len=canonical_lens, perceived_len=perceived_lens,
                ind2lab=self.label_encoder.decode_ndim,
            )
        
        return loss


class PhnMonoSSLModel_withcanoPhnEmb_HMA_CTC(PhnMonoSSLModel):
    """
    Hybrid Model with Attention: [Attn, SSL] -> Linear -> CTC
    Uses canonical phoneme embeddings with cross-attention.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Move canonical phoneme modules to device
        if self.modules.CanonicalPhonemeEmbedding:
            self.modules.CanonicalPhonemeEmbedding.to(self.device)
        if self.modules.CanonicalPhonemeLSTM:
            self.modules.CanonicalPhonemeLSTM.to(self.device)
        if self.modules.CanonicalPhonemeLinear:
            self.modules.CanonicalPhonemeLinear.to(self.device)
        if self.modules.cross_attention:
            self.modules.cross_attention.to(self.device)
        if self.modules.attn_proj:
            self.modules.attn_proj.to(self.device)
        if self.modules.out_sequence:
            self.modules.out_sequence.to(self.device)
    
    def compute_forward(self, batch, stage):
        """Forward with canonical phoneme attention"""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        canonicals, canonical_lens = batch.phn_encoded_canonical
        
        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs)
        
        # SSL features
        feats = self.modules.perceived_ssl(wavs)
        x = self.modules.enc(feats)
        
        # CTC branch
        logits = self.modules.ctc_lin(x)
        p_ctc = self.hparams.log_softmax(logits)
        
        # Canonical phoneme embeddings
        canonical_embeds = self.modules.CanonicalPhonemeEmbedding(canonicals)
        canonical_lstm_out, _ = self.modules.CanonicalPhonemeLSTM(canonical_embeds)
        canonical_out = self.modules.CanonicalPhonemeLinear(canonical_lstm_out)
        
        # Cross-attention
        attn_output, attn_map = self.modules.cross_attention(
            query=x,
            key=canonical_lstm_out,
            value=canonical_out,
        )
        attn_output_logits = self.modules.attn_proj(attn_output)
        p_attn = self.hparams.log_softmax(attn_output_logits)
        
        # Combined output
        concat_hidden = torch.cat((logits, attn_output_logits), dim=-1)
        out_logits = self.modules.out_sequence(concat_hidden)
        p_out = self.hparams.log_softmax(out_logits)
        
        return p_out, p_ctc, p_attn, wav_lens, attn_map
    
    def compute_objectives(self, predictions, batch, stage):
        """Compute objectives for HMA model"""
        p_out, p_ctc, p_attn, wav_lens, attn_map = predictions
        
        ids = batch.id
        targets, target_lens = batch.phn_encoded_target
        canonicals, canonical_lens = batch.phn_encoded_canonical
        perceiveds, perceived_lens = batch.phn_encoded_perceived
        
        # Compute losses
        loss_ctc = self.hparams.ctc_cost(p_ctc, targets, wav_lens, target_lens)
        loss_attn = self.hparams.ctc_cost(p_attn, targets, wav_lens, target_lens)
        loss_out = self.hparams.ctc_cost(p_out, targets, wav_lens, target_lens)
        
        loss = loss_attn  # or combine: loss_ctc + loss_attn + loss_out
        
        # Evaluation
        if stage != sb.Stage.TRAIN:
            sequence_out = sb.decoders.ctc_greedy_decode(
                p_out, wav_lens, blank_id=self.hparams.blank_index
            )
            
            self.ctc_metrics.append(ids, p_out, targets, wav_lens, target_lens)
            self.per_metrics.append(
                ids=ids, predict=sequence_out, target=targets,
                predict_len=None, target_len=target_lens,
                ind2lab=self.label_encoder.decode_ndim,
            )
            self.mpd_metrics.append(
                ids=ids, predict=sequence_out,
                canonical=canonicals, perceived=perceiveds,
                predict_len=None,
                canonical_len=canonical_lens, perceived_len=perceived_lens,
                ind2lab=self.label_encoder.decode_ndim,
            )
        
        return loss
