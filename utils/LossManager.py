# ============================================================================
# Unified CTC Loss Manager
# ============================================================================

import torch
from torch import nn
import speechbrain as sb

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
                
                loss_dict['ctc_loss'] = loss.detach()  # Keep on GPU
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
        
        # =========== CROTTC Loss ============
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
                
                loss_dict['ctc_loss'] = loss.detach()  # Keep on GPU
                if extras and extras.get('is_crottc_mode', False):
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
