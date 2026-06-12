# ============================================================================
# Unified CTC Loss Manager
# ============================================================================

import torch
from torch import nn
import torch.nn.functional as F
import speechbrain as sb

class CTCLossManager:
    """
    Unified CTC loss manager that handles different loss types:
    - vanilla: Standard CTC loss
    - label_prior: CTC with label priors (CTCLossWithLabelPriors)
    - ottc: Optimal Transport CTC loss
    - crctc: Consistency-Regularized CTC loss
    - crottc: Consistency-Regularized OTTC loss
    
    Contrastive addon (use_contrastive=True):
    - Can be combined with ANY main loss type above
    - Computes MDDContrastiveLoss using perceived vs canonical targets
    - Supports both CTC-style and OTTC-style contrastive computation
    
    Usage:
        # YAML config:
        #   ctc_loss_type: "crottc"
        #   use_contrastive: True
        #   contrastive_weight: 1.0
        #   MDDContrastiveLoss_margin: 16.0
        
        loss_manager = CTCLossManager(
            loss_type='crottc',
            ctc_cost=hparams.ctc_cost,
            hparams=hparams
        )
        loss, loss_dict = loss_manager.compute_loss(
            p_ctc, targets, wav_lens, target_lens, stage,
            extras={'canonicals': canonicals, 'canonical_lens': canonical_lens,
                    'perceiveds': perceiveds, 'perceived_lens': perceived_lens,
                    'logits': logits, 'weights_logits': weights_logits, ...}
        )
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
        
        # ====== Contrastive loss addon (works with any main loss type) ======
        # Enable if: loss_type == 'contrastive' (legacy) OR hparams.use_contrastive == True
        self.use_contrastive = (
            loss_type == 'contrastive' or 
            getattr(hparams, 'use_contrastive', False)
        )
        
        if self.use_contrastive:
            self.contrastive_margin = getattr(
                hparams, 'contrastive_margin',
                getattr(hparams, 'MDDContrastiveLoss_margin', 16.0)
            )
            self.contrastive_weight = getattr(hparams, 'contrastive_weight', 1.0)
            
            # Use user-configured MDDContrastiveLoss from hparams if available
            user_mdd = getattr(hparams, 'MDDContrastiveLoss', None)
            if user_mdd is not None:
                self.mdd_contrastive_loss = user_mdd
            else:
                # Create default with nn.CTCLoss
                from utils.losses.CTCContrastiveLoss import MDDContrastiveLoss
                self.contrastive_ctc_loss = nn.CTCLoss(
                    blank=blank_index, reduction='none', zero_infinity=True
                )
                self.mdd_contrastive_loss = MDDContrastiveLoss(
                    ctc_loss=self.contrastive_ctc_loss,
                    margin=self.contrastive_margin
                )

            
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
            
        # ============ Contrastive Loss (CTC + MDDContrastiveLoss) ============
        # LEGACY: standalone contrastive mode (loss_type='contrastive')
        # For addon mode (use_contrastive=True with other loss types), see below
        elif actual_loss_type == 'contrastive':
            # Standard CTC loss on targets (perceived annotations)
            loss_ctc = self.ctc_cost(p_ctc, targets, wav_lens, target_lens)
            loss_dict['ctc_loss'] = loss_ctc.detach()
            loss = loss_ctc
            # Contrastive addon is applied below in the shared addon section
            
        else:
            # Default to vanilla
            loss = self.ctc_cost(p_ctc, targets, wav_lens, target_lens)
            loss_dict['ctc_loss'] = loss.detach()  # Keep on GPU
        
        # ================================================================
        # Contrastive Loss Addon (works with ANY main loss type above)
        # ================================================================
        if (self.use_contrastive and 
            stage == sb.Stage.TRAIN and 
            extras and
            extras.get('canonicals') is not None and 
            extras.get('canonical_lens') is not None and
            extras.get('perceiveds') is not None and 
            extras.get('perceived_lens') is not None):
            
            contrastive_loss = self._compute_contrastive_addon(
                p_ctc, wav_lens, extras, actual_loss_type
            )
            if contrastive_loss is not None:
                loss = loss + self.contrastive_weight * contrastive_loss
                loss_dict['contrastive_loss'] = contrastive_loss.detach()
            
        return loss, loss_dict
    
    def _compute_contrastive_addon(self, p_ctc, wav_lens, extras, actual_loss_type):
        """
        Compute contrastive loss addon. Auto-detects interface based on loss type.
        
        For CTC-compatible losses: uses MDDContrastiveLoss (nn.CTCLoss interface)
        For OTTC/CROTTC: computes OT distance contrastive directly
        
        Contrastive formula: max(-loss(X, perceived) + loss(X, canonical) + margin, 0)
        
        Args:
            p_ctc: [B, T, C] or [2*B, T, C] log probabilities
            wav_lens: [B] or [2*B] relative lengths
            extras: dict with canonicals, perceiveds, and optionally logits/weights
            actual_loss_type: str, the detected loss type
        
        Returns:
            contrastive_loss: scalar tensor, or None if unable to compute
        """
        canonicals = extras['canonicals']
        canonical_lens = extras['canonical_lens']
        perceiveds = extras['perceiveds']
        perceived_lens = extras['perceived_lens']
        B = canonicals.shape[0]
        
        # For CROTTC/CRCTC, p_ctc is [2*B, T, C], take first half
        if p_ctc.shape[0] > B:
            p_ctc_eval = p_ctc[:B]
            wav_lens_eval = wav_lens[:B]
        else:
            p_ctc_eval = p_ctc
            wav_lens_eval = wav_lens
        
        # Route to OTTC-style or CTC-style contrastive
        if actual_loss_type in ('ottc', 'crottc') and extras.get('logits') is not None:
            return self._contrastive_ottc(
                extras['logits'], extras.get('weights_logits'),
                canonicals, canonical_lens, perceiveds, perceived_lens, B
            )
        else:
            return self._contrastive_ctc(
                p_ctc_eval, wav_lens_eval,
                canonicals, canonical_lens, perceiveds, perceived_lens
            )
    
    def _contrastive_ctc(self, p_ctc, wav_lens, canonicals, canonical_lens, perceiveds, perceived_lens):
        """
        Contrastive loss using CTC-compatible interface (via MDDContrastiveLoss).
        """
        B = perceiveds.shape[0]
        
        # Prepare for CTCLoss format: [T, B, C]
        log_probs_t = p_ctc.permute(1, 0, 2)  # [B, T, C] -> [T, B, C]
        
        # Compute absolute lengths
        input_lengths = (wav_lens * p_ctc.shape[1]).to(torch.int32)
        
        # Flatten perceived targets
        abs_perceived_lens = (perceived_lens * perceiveds.shape[1]).to(torch.int32)
        perceived_flat = torch.cat([
            perceiveds[i, :abs_perceived_lens[i]] for i in range(B)
        ])
        
        # Flatten canonical targets
        abs_canonical_lens = (canonical_lens * canonicals.shape[1]).to(torch.int32)
        canonical_flat = torch.cat([
            canonicals[i, :abs_canonical_lens[i]] for i in range(B)
        ])
        
        # Compute contrastive loss via MDDContrastiveLoss
        contrastive_loss = self.mdd_contrastive_loss(
            log_probs=log_probs_t,
            target_annot=perceived_flat,
            target_ref=canonical_flat,
            input_lengths=input_lengths,
            annot_lengths=abs_perceived_lens,
            ref_lengths=abs_canonical_lens
        )
        return contrastive_loss
    
    def _contrastive_ottc(self, logits, weights_logits, canonicals, canonical_lens, 
                           perceiveds, perceived_lens, B):
        """
        Contrastive loss using OTTC interface.
        
        Computes: max(-OTTC_loss(X, perceived) + OTTC_loss(X, canonical) + margin, 0)
        
        Uses the same ctc_cost (batched_ottc_loss_bucketized) for both branches.
        """
        # For CROTTC: logits is [2*B, T, C], take first half
        if logits.shape[0] > B:
            logits_eval = logits[:B]
            weights_logits_eval = weights_logits[:B] if weights_logits is not None else None
        else:
            logits_eval = logits
            weights_logits_eval = weights_logits
        
        num_classes = logits_eval.shape[-1]
        
        # ---- Compute OTTC loss for perceived (L^e) ----
        perceived_mask = (perceiveds != self.blank_index).float()
        perceived_sums = perceived_mask.sum(dim=1, keepdim=True).clamp(min=1e-8)
        w_perceived = perceived_mask / perceived_sums
        one_hot_perceived = F.one_hot(perceiveds.long(), num_classes=num_classes).float()
        
        loss_perceived, _, _, _ = self.ctc_cost(
            x=logits_eval, y=one_hot_perceived,
            a=weights_logits_eval, b=w_perceived,
            amask=None, bmask=perceived_mask,
            euclidian=False, jsd=False,
        )
        
        # ---- Compute OTTC loss for canonical (L) ----
        canonical_mask = (canonicals != self.blank_index).float()
        canonical_sums = canonical_mask.sum(dim=1, keepdim=True).clamp(min=1e-8)
        w_canonical = canonical_mask / canonical_sums
        one_hot_canonical = F.one_hot(canonicals.long(), num_classes=num_classes).float()
        
        loss_canonical, _, _, _ = self.ctc_cost(
            x=logits_eval, y=one_hot_canonical,
            a=weights_logits_eval, b=w_canonical,
            amask=None, bmask=canonical_mask,
            euclidian=False, jsd=False,
        )
        
        # ---- Contrastive: max(-loss_perceived + loss_canonical + margin, 0) ----
        # 直觉：perceived (实际发音) 应该比 canonical (标准发音) 更容易匹配
        # 如果 loss_perceived < loss_canonical - margin, 说明模型能区分误读
        contrastive_loss = torch.clamp(
            -loss_perceived + loss_canonical + self.contrastive_margin, min=0.0
        )
        
        return contrastive_loss

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
