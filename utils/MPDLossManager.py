# ============================================================================
# Mispronunciation Detection (MPD) Loss Manager
# ============================================================================

import torch
from torch import nn
import speechbrain as sb


class MPDLossManager:
    """
    Manager for computing mispronunciation detection losses.
    
    Handles:
    - Binary classification loss (is_mispronunciation or not)
    - Multi-class classification loss (4-class: correct, substitution, insertion, deletion)
    
    Architecture:
    - h_mispro_bin: [B, T_c, 1] binary logits
    - h_mispro_cls: [B, T_c, 4] multi-class logits
    
    Usage:
        mpd_loss_manager = MPDLossManager(hparams=hparams, device=device)
        loss, loss_dict = mpd_loss_manager.compute_loss(
            h_mispro_bin, h_mispro_cls, batch, stage
        )
    """
    
    def __init__(self, hparams=None, device='cpu'):
        """
        Initialize MPD Loss Manager.
        
        Args:
            hparams: hyperparameters object (optional)
            device: torch device
        """
        self.hparams = hparams
        self.device = device
        
        # Loss weights for combining binary and multi-class losses
        self.binary_loss_weight = getattr(hparams, 'mpd_binary_loss_weight', 0.5) if hparams else 0.5
        self.cls_loss_weight = getattr(hparams, 'mpd_cls_loss_weight', 0.5) if hparams else 0.5
        
        # Loss functions
        self.binary_loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
        self.cls_loss_fn = nn.CrossEntropyLoss(reduction='mean')
    
    def compute_loss(self, h_mispro_bin, h_mispro_cls, batch, stage=sb.Stage.TRAIN):
        """
        Compute MPD losses for both binary and multi-class branches.
        
        Args:
            h_mispro_bin: [B, T_c, 1] binary logits (mispronunciation detection)
            h_mispro_cls: [B, T_c, 4] multi-class logits (error type classification)
            batch: batch object (should contain MPD labels if available)
            stage: sb.Stage.TRAIN/VALID/TEST
        
        Returns:
            loss: scalar total loss (weighted combination)
            loss_dict: dict with {'binary_loss': ..., 'cls_loss': ..., 'total_mpd_loss': ...}
        """
        loss_dict = {}
        
        # Check if we have MPD labels in the batch
        has_binary_labels = hasattr(batch, 'mpd_binary_labels') and batch.mpd_binary_labels is not None
        has_cls_labels = hasattr(batch, 'mpd_cls_labels') and batch.mpd_cls_labels is not None
        import pdb; pdb.set_trace()
        
        if not has_binary_labels and not has_cls_labels:
            # No MPD labels available, return zero loss
            loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
            loss_dict['binary_loss'] = loss.detach()
            loss_dict['cls_loss'] = loss.detach()
            loss_dict['total_mpd_loss'] = loss.detach()
            return loss, loss_dict
        
        device = h_mispro_bin.device
        total_loss = 0.0
        
        # ===== Binary Loss =====
        if has_binary_labels:
            binary_labels = batch.mpd_binary_labels.to(device)
            
            # binary_labels shape: [B, T_c] with values 0 (correct) or 1 (mispronounced)
            # h_mispro_bin shape: [B, T_c, 1]
            h_mispro_bin_squeezed = h_mispro_bin.squeeze(-1)  # [B, T_c]
            binary_labels = binary_labels.float()  # Ensure float type for BCEWithLogitsLoss
            
            binary_loss = self.binary_loss_fn(h_mispro_bin_squeezed, binary_labels)
            loss_dict['binary_loss'] = binary_loss.detach()
            total_loss += self.binary_loss_weight * binary_loss
        else:
            loss_dict['binary_loss'] = torch.tensor(0.0, device=device, dtype=torch.float32)
        
        # ===== Multi-class Loss =====
        if has_cls_labels:
            cls_labels = batch.mpd_cls_labels.to(device)  # [B, T_c]
            
            # h_mispro_cls shape: [B, T_c, 4]
            # Need to reshape for CrossEntropyLoss: (N, C)
            B, T_c, num_classes = h_mispro_cls.shape
            
            # Flatten: [B*T_c, 4]
            h_mispro_cls_flat = h_mispro_cls.reshape(-1, num_classes)
            cls_labels_flat = cls_labels.reshape(-1)  # [B*T_c]
            
            cls_loss = self.cls_loss_fn(h_mispro_cls_flat, cls_labels_flat)
            loss_dict['cls_loss'] = cls_loss.detach()
            total_loss += self.cls_loss_weight * cls_loss
        else:
            loss_dict['cls_loss'] = torch.tensor(0.0, device=device, dtype=torch.float32)
        
        # ===== Total MPD Loss =====
        loss_dict['total_mpd_loss'] = total_loss.detach()
        
        return total_loss, loss_dict
    
    def compute_loss_with_weight_mask(self, h_mispro_bin, h_mispro_cls, batch, 
                                      cano_lens=None, stage=sb.Stage.TRAIN):
        """
        Compute MPD losses with optional masking based on canonical phoneme lengths.
        
        Args:
            h_mispro_bin: [B, T_c, 1] binary logits
            h_mispro_cls: [B, T_c, 4] multi-class logits
            batch: batch object with MPD labels
            cano_lens: [B] canonical sequence lengths (for masking)
            stage: training stage
        
        Returns:
            loss: scalar total loss
            loss_dict: dict with loss components
        """
        loss_dict = {}
        
        # has_cls_labels = hasattr(batch, 'mpd_cls_labels') and batch.mpd_cls_labels is not None
        
        has_cls_labels = hasattr(batch, 'mispro_label') and batch.mispro_label is not None # [0, 1, 2, 3]
        # import pdb; pdb.set_trace()
        if not has_cls_labels:
            loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
            loss_dict['binary_loss'] = loss.detach()
            loss_dict['cls_loss'] = loss.detach()
            loss_dict['total_mpd_loss'] = loss.detach()
            return loss, loss_dict
        has_binary_labels = True
        # create binary labels with batch.mispro_label 
        binary_labels = (batch.mispro_label[0] > 0).long()  # [B, T_c], 0: correct, 1: mispronounced
        
        device = h_mispro_bin.device
        B, T_c, _ = h_mispro_bin.shape
        
        # Create mask from canonical lengths if provided
        if cano_lens is not None:
            # cano_lens: [B] relative lengths (0-1)
            max_len = T_c
            abs_lens = (cano_lens * max_len).long()
            
            # Create mask: 1 for valid positions, 0 for padding
            indices = torch.arange(max_len, device=device, dtype=abs_lens.dtype).unsqueeze(0)
            mask = (indices < abs_lens.unsqueeze(1)).float()  # [B, T_c]
        else:
            mask = torch.ones(B, T_c, device=device)
        
        total_loss = 0.0
        
        # ===== Binary Loss with Masking =====
        if has_binary_labels:
            binary_labels = binary_labels.to(device).float()  # [B, T_c]
            h_bin_sq = h_mispro_bin.squeeze(-1)  # [B, T_c]
            
            binary_loss_unreduced = torch.nn.functional.binary_cross_entropy_with_logits(
                h_bin_sq, binary_labels, reduction='none'
            )  # [B, T_c]
            
            # Apply mask
            binary_loss_masked = (binary_loss_unreduced * mask).sum() / (mask.sum() + 1e-8)
            loss_dict['binary_loss'] = binary_loss_masked.detach()
            total_loss += self.binary_loss_weight * binary_loss_masked
        else:
            loss_dict['binary_loss'] = torch.tensor(0.0, device=device, dtype=torch.float32)
        
        # ===== Multi-class Loss with Masking =====
        if has_cls_labels:
            cls_labels = batch.mispro_label[0].to(device)  # [B, T_c]
            
            # Flatten
            h_cls_flat = h_mispro_cls.reshape(-1, 4)  # [B*T_c, 4]
            cls_labels_flat = cls_labels.reshape(-1)  # [B*T_c]
            mask_flat = mask.reshape(-1)  # [B*T_c]
            
            # Compute loss without reduction
            cls_loss_unreduced = torch.nn.functional.cross_entropy(
                h_cls_flat, cls_labels_flat, reduction='none'
            )  # [B*T_c]
            
            # Apply mask and compute mean
            cls_loss_masked = (cls_loss_unreduced * mask_flat).sum() / (mask_flat.sum() + 1e-8)
            loss_dict['cls_loss'] = cls_loss_masked.detach()
            total_loss += self.cls_loss_weight * cls_loss_masked
        else:
            loss_dict['cls_loss'] = torch.tensor(0.0, device=device, dtype=torch.float32)
        
        loss_dict['total_mpd_loss'] = total_loss.detach()
        
        return total_loss, loss_dict
