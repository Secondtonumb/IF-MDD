import os

import torch
import torch.nn

import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from mpd_eval_v4 import MpdStats


import wandb

from speechbrain.nnet.loss.guidedattn_loss import GuidedAttentionLoss
from speechbrain.lobes.models.dual_path import PyTorchPositionalEncoding
from speechbrain.nnet.attention import RelPosEncXL, RelPosMHAXL, RoPEMHA 
from speechbrain.lobes.models.VanillaNN import VanillaNN
from speechbrain.nnet.transducer import transducer_joint
from speechbrain.utils.metric_stats import MetricStats
from speechbrain.nnet.losses import ctc_loss
from torch.nn import functional as F
import pdb

from speechbrain.decoders import S2STransformerBeamSearcher, CTCScorer, ScorerBuilder

from speechbrain.lobes.models.transformer.TransformerASR import TransformerASR
from speechbrain.lobes.models.transformer.Transformer import get_lookahead_mask

from speechbrain.decoders.utils import (
    _update_mem,
    inflate_tensor,
    mask_by_condition,
)

import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.layers.utils import make_pad_mask
from utils.plot.plot_attn import plot_attention

class Trans_IFMDD_ConPCO_ver2(sb.Brain):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.        super().__init__(*args, **kwargs)
        self.patience = 20
        
        self.no_improve_epochs = 0
        self.last_improved_epoch = 0
        
        self.best_per_list = []  # List of (PER, epoch, ckpt_name)
        self.best_mpd_f1_list = []  # List of (mpd_f1, epoch, ckpt_name)
        self.best_per = float('inf')
        self.best_mpd_f1 = float('-inf')
        
        self.best_per_seq_list = []  # List of (PER_seq, epoch, ckpt_name)
        self.best_mpd_f1_seq_list = []
        self.best_per_seq = float('inf')
        self.best_mpd_f1_seq = float('-inf')
            
        self.best_valid_loss = float('inf')
        self.best_valid_loss_list = []  # List of (valid_loss, epoch, ckpt_name)
        
        # CTC freezing mechanism
        self.enable_ctc_freezing = getattr(self.hparams, 'enable_ctc_freezing', True)
        self.best_ctc_loss = float('inf')
        self.ctc_patience = getattr(self.hparams, 'ctc_patience', 5)
        self.ctc_threshold_factor = getattr(self.hparams, 'ctc_threshold_factor', 1.1)
        self.ctc_no_improve_epochs = 0
        self.encoder_frozen = False
        self.ssl_frozen = False
        self.ctc_threshold = None  # Will be set based on training progress
        
        # Track CTC loss history for threshold setting
        self.ctc_loss_history = []
        
        # Metric-based freezing mechanism
        self.enable_metric_freezing = getattr(self.hparams, 'enable_metric_freezing', False)
        self.freezing_metric = getattr(self.hparams, 'freezing_metric', 'PER')
        self.metric_patience = getattr(self.hparams, 'metric_patience', 10)
        self.per_threshold_factor = getattr(self.hparams, 'per_threshold_factor', 1.05)
        self.f1_threshold_factor = getattr(self.hparams, 'f1_threshold_factor', 0.95)
        self.min_epochs_before_metric_freeze = getattr(self.hparams, 'min_epochs_before_metric_freeze', 15)
        
        # Track metric history and convergence
        self.best_valid_per = float('inf')
        self.best_valid_f1 = float('-inf')
        self.metric_no_improve_epochs = 0
        self.per_threshold = None
        self.f1_threshold = None
        self.per_history = []
        self.f1_history = []
    
    def freeze_encoder_and_ssl(self):
        """Freeze encoder and SSL model parameters"""
        if not self.encoder_frozen:
            # Freeze TransASR (encoder part )
            for param in self.modules.TransASR.encoder.parameters():
                param.requires_grad = False
            for param in self.modules.TransASR.custom_src_module.parameters():
                param.requires_grad = False
            print("✓ Encoder (TransASR) and Encoder Prenet frozen")
            self.encoder_frozen = True
            
            # Also freeze the encoder projection layer if exists
            if hasattr(self.modules, 'enc'):
                for param in self.modules.enc.parameters():
                    param.requires_grad = False
                print("✓ Encoder projection layer frozen")
        
        if not self.ssl_frozen:
            # Freeze perceived SSL model
            for param in self.modules.perceived_ssl.parameters():
                param.requires_grad = False
            print("✓ Perceived SSL model frozen")
            self.ssl_frozen = True
            
        # Save the frozen state checkpoint
        if self.checkpointer is not None:
            current_epoch = self.hparams.epoch_counter.current
            ckpt_name = f"encoder_ssl_frozen_epoch_{current_epoch:03d}.ckpt"
            meta = {
                "epoch": current_epoch, 
                "encoder_frozen": True, 
                "ssl_frozen": True,
                "frozen_by_ctc": self.enable_ctc_freezing and not self.enable_metric_freezing,
                "frozen_by_metric": self.enable_metric_freezing,
                "freezing_metric": getattr(self, 'freezing_metric', 'unknown'),
                "best_ctc_loss": self.best_ctc_loss,
                "best_valid_per": self.best_valid_per,
                "best_valid_f1": self.best_valid_f1,
            }
            self.checkpointer.save_and_keep_only(
                meta=meta,
                name=ckpt_name,
                num_to_keep=1,
            )
            print(f"✓ Saved frozen model checkpoint: {ckpt_name}")
    
    def manual_freeze_encoder_ssl(self):
        """Manually trigger freezing of encoder and SSL model"""
        print("\n🔒 Manual freeze triggered...")
        self.freeze_encoder_and_ssl()
        self.enable_ctc_freezing = False  # Disable automatic checking
        print("✓ Automatic CTC freezing disabled")
    
    def unfreeze_encoder_ssl(self):
        """Unfreeze encoder and SSL model parameters"""
        if self.encoder_frozen:
            # Unfreeze TransASR (encoder part)
            for param in self.modules.TransASR.parameters():
                param.requires_grad = True
            print("✓ Encoder (TransASR) unfrozen")
            
            # Unfreeze encoder projection layer if exists
            if hasattr(self.modules, 'enc'):
                for param in self.modules.enc.parameters():
                    param.requires_grad = True
                print("✓ Encoder projection layer unfrozen")
            
            self.encoder_frozen = False
        
        if self.ssl_frozen:
            # Unfreeze perceived SSL model
            for param in self.modules.perceived_ssl.parameters():
                param.requires_grad = True
            print("✓ Perceived SSL model unfrozen")
            self.ssl_frozen = False
    
    def check_ctc_convergence(self, current_ctc_loss):
        """Check if CTC loss has converged and freeze encoder/SSL if needed"""
        if not self.enable_ctc_freezing:
            return
            
        current_epoch = self.hparams.epoch_counter.current
        
        # Don't start checking until we have enough epochs (at least 5 epochs)
        if current_epoch < 5:
            return
            
        self.ctc_loss_history.append(current_ctc_loss)
        
        # Only start checking after 10 epochs total to have enough history
        if len(self.ctc_loss_history) < 10:
            return
            
        # Set threshold based on configured factor
        if self.ctc_threshold is None:
            min_ctc = min(self.ctc_loss_history)
            self.ctc_threshold = min_ctc * self.ctc_threshold_factor
            print(f"✓ CTC threshold set to: {self.ctc_threshold:.6f} (factor: {self.ctc_threshold_factor})")
        
        # Check if current loss is close to the best
        if current_ctc_loss <= self.best_ctc_loss:
            self.best_ctc_loss = current_ctc_loss
            self.ctc_no_improve_epochs = 0
        else:
            self.ctc_no_improve_epochs += 1
        
        # Print progress every few epochs
        if current_epoch % 5 == 0:
            print(f"📊 CTC Convergence Check (Epoch {current_epoch}):")
            print(f"   Current CTC Loss: {current_ctc_loss:.6f}")
            print(f"   Best CTC Loss: {self.best_ctc_loss:.6f}")
            print(f"   No improve epochs: {self.ctc_no_improve_epochs}/{self.ctc_patience}")
            if self.ctc_threshold:
                print(f"   Threshold: {self.ctc_threshold:.6f}")
        
        # Freeze if CTC has converged (no improvement for ctc_patience epochs)
        # AND current loss is below threshold
        if (not self.encoder_frozen and 
            self.ctc_no_improve_epochs >= self.ctc_patience and 
            current_ctc_loss <= self.ctc_threshold):
            
            print(f"\n🔒 CTC Loss Converged! (No improvement for {self.ctc_patience} epochs)")
            print(f"   Current CTC Loss: {current_ctc_loss:.6f}")
            print(f"   Best CTC Loss: {self.best_ctc_loss:.6f}")
            print(f"   Threshold: {self.ctc_threshold:.6f}")
            print("   Freezing Encoder and SSL model...")
            
            self.freeze_encoder_and_ssl()
    
    def load_pretrained_components(self, checkpoint_path, components_to_load=None, freeze_loaded=True):
        """
        Load specific components from a pretrained model checkpoint
        
        Args:
            checkpoint_path (str): Path to the checkpoint directory or file
            components_to_load (list): List of components to load. 
                                     Options: ['ssl', 'encoder', 'ctc_head', 'decoder', 'enc_projection']
                                     If None, loads ['ssl', 'encoder'] by default
            freeze_loaded (bool): Whether to freeze the loaded components
        """
        if components_to_load is None:
            components_to_load = ['ssl']  # Default: load SSL 
        
        print(f"\n🔄 Loading pretrained components from: {checkpoint_path}")
        print(f"   Components to load: {components_to_load}")
        # pdb.set_trace()
                
        from speechbrain.utils.parameter_transfer import Pretrainer
        
        pretrainer = Pretrainer(
            collect_in=self.hparams.pretrained_model_path,      # 把文件收集到这个目录（用软链或拷贝）
            loadables={
                "perceived_ssl":     self.modules.perceived_ssl,
                "model":     self.hparams.model,
            },
            paths={
                # 只写文件名，后面用 default_source 指定“仓库/目录”
                "perceived_ssl":     "perceived_ssl.ckpt",
                "model":   "model.ckpt",
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
                    self.ssl_frozen = True
                    print("   🔒 SSL model frozen")
                    
                elif component == 'encoder':
                    for param in self.modules.TransASR.encoder.parameters():
                        param.requires_grad = False
                    if hasattr(self.modules.TransASR, 'custom_src_module'):
                        for param in self.modules.TransASR.custom_src_module.parameters():
                            param.requires_grad = False
                    self.encoder_frozen = True
                    print("   🔒 Encoder frozen")
                    
                elif component == 'enc':
                    if hasattr(self.modules, 'enc'):
                        for param in self.modules.enc.parameters():
                            param.requires_grad = False
                        print("   🔒 Encoder projection frozen")
                        
                elif component == 'ctc_head':
                    for param in self.modules.ctc_lin.parameters():
                        param.requires_grad = False
                    print("   🔒 CTC head frozen")
    
        # print(f"   ✅ Successfully loaded components: {loaded_components}")
        # return loaded_components
    
    def load_from_checkpoint_manual(self, checkpoint_path, ssl_only=False, encoder_only=False, 
                                  freeze_ssl=True, freeze_encoder=True):
        """
        Simplified method to manually load components from checkpoint
        
        Args:
            checkpoint_path (str): Path to checkpoint directory or file
            ssl_only (bool): Load only SSL model
            encoder_only (bool): Load only encoder (TransASR)
            freeze_ssl (bool): Whether to freeze SSL after loading
            freeze_encoder (bool): Whether to freeze encoder after loading
        """
        components = []
        if ssl_only:
            components = ['ssl']
        elif encoder_only:
            components = ['encoder']
        else:
            components = ['ssl', 'encoder']  # Default: both
        
        freeze_loaded = freeze_ssl or freeze_encoder
        
        return self.load_pretrained_components(
            checkpoint_path=checkpoint_path,
            components_to_load=components,
            freeze_loaded=freeze_loaded
        )
    
    def print_parameter_status(self):
        """Print the current status of model parameters (trainable vs frozen)"""
        print(f"\n📊 Model Parameter Status:")
        
        total_params = 0
        trainable_params = 0
        
        # Check each module
        modules_info = {}
        for name, module in self.modules.named_children():
            module_params = sum(p.numel() for p in module.parameters())
            module_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            total_params += module_params
            trainable_params += module_trainable
            
            status = "🔒 FROZEN" if module_trainable == 0 else "🔓 TRAINABLE"
            modules_info[name] = (module_trainable, module_params, status)
            print(f"   {name}: {module_trainable:,}/{module_params:,} params {status}")
        
        # Summary
        frozen_ratio = (total_params - trainable_params) / total_params * 100 if total_params > 0 else 0
        print(f"\n📈 Summary:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Frozen parameters: {total_params - trainable_params:,}")
        print(f"   Frozen ratio: {frozen_ratio:.1f}%")
        
        # Freezing status
        print(f"\n🔒 Component Status:")
        print(f"   SSL frozen: {self.ssl_frozen}")
        print(f"   Encoder frozen: {self.encoder_frozen}")
        
        return modules_info, total_params, trainable_params

    def check_metric_convergence(self, current_per, current_f1):
        """Check if validation metrics have converged and freeze encoder/SSL if needed"""
        if not self.enable_metric_freezing:
            return
            
        current_epoch = self.hparams.epoch_counter.current
        
        # Don't start checking until minimum epochs
        if current_epoch < self.min_epochs_before_metric_freeze:
            return
            
        # Track metric history
        self.per_history.append(current_per)
        self.f1_history.append(current_f1)
        
        # Set thresholds based on history (need at least 10 epochs of data)
        if len(self.per_history) >= 10:
            if self.per_threshold is None:
                min_per = min(self.per_history)
                self.per_threshold = min_per * self.per_threshold_factor
                print(f"✓ PER threshold set to: {self.per_threshold:.4f} (factor: {self.per_threshold_factor})")
            
            if self.f1_threshold is None:
                max_f1 = max(self.f1_history)
                self.f1_threshold = max_f1 * self.f1_threshold_factor
                print(f"✓ F1 threshold set to: {self.f1_threshold:.4f} (factor: {self.f1_threshold_factor})")
        
        # Check for improvement based on selected metric
        improved = False
        if self.freezing_metric == "PER":
            if current_per <= self.best_valid_per:
                self.best_valid_per = current_per
                improved = True
        elif self.freezing_metric == "F1":
            if current_f1 >= self.best_valid_f1:
                self.best_valid_f1 = current_f1
                improved = True
        elif self.freezing_metric == "both":
            if current_per <= self.best_valid_per or current_f1 >= self.best_valid_f1:
                if current_per <= self.best_valid_per:
                    self.best_valid_per = current_per
                if current_f1 >= self.best_valid_f1:
                    self.best_valid_f1 = current_f1
                improved = True
        
        if improved:
            self.metric_no_improve_epochs = 0
        else:
            self.metric_no_improve_epochs += 1
        
        # Print progress every few epochs
        if current_epoch % 5 == 0:
            print(f"📊 Metric Convergence Check (Epoch {current_epoch}):")
            print(f"   Current PER: {current_per:.4f}, Current F1: {current_f1:.4f}")
            print(f"   Best PER: {self.best_valid_per:.4f}, Best F1: {self.best_valid_f1:.4f}")
            print(f"   No improve epochs: {self.metric_no_improve_epochs}/{self.metric_patience}")
            if self.per_threshold and self.f1_threshold:
                print(f"   PER threshold: {self.per_threshold:.4f}, F1 threshold: {self.f1_threshold:.4f}")
        
        # Check if we should freeze based on the selected metric
        should_freeze = False
        if self.freezing_metric == "PER" and self.per_threshold is not None:
            should_freeze = (current_per <= self.per_threshold and 
                           self.metric_no_improve_epochs >= self.metric_patience)
        elif self.freezing_metric == "F1" and self.f1_threshold is not None:
            should_freeze = (current_f1 >= self.f1_threshold and 
                           self.metric_no_improve_epochs >= self.metric_patience)
        elif self.freezing_metric == "both" and self.per_threshold is not None and self.f1_threshold is not None:
            should_freeze = ((current_per <= self.per_threshold or current_f1 >= self.f1_threshold) and 
                           self.metric_no_improve_epochs >= self.metric_patience)
        
        # Freeze if conditions are met
        if not self.encoder_frozen and should_freeze:
            print(f"\n🔒 Validation Metrics Converged! (No improvement for {self.metric_patience} epochs)")
            print(f"   Freezing based on metric: {self.freezing_metric}")
            print(f"   Current PER: {current_per:.4f}, Current F1: {current_f1:.4f}")
            print(f"   Freezing Encoder and SSL model...")
            
            self.freeze_encoder_and_ssl()
    
    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        predictions = self.compute_forward(batch, stage=stage)
        loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()
    
    def on_stage_start(self, stage, epoch):
        "Gets called when a stage (either training, va lidation, test) starts."
        self.ctc_metrics = self.hparams.ctc_stats()
        self.seq_metrics = self.hparams.seqlabel_stats()
        self.mispro_metrics = self.hparams.mispro_stats()
        self.mispro_metrics_cls = self.hparams.mispro_class_stats()
        self.conpco_metrics = self.hparams.conpco_stats()
        
        
        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "augmentation"):
                self.modules.perceived_ssl.model.config.apply_spec_augment = True

        if stage != sb.Stage.TRAIN:
            self.per_metrics = self.hparams.per_stats()
            self.per_metrics_seq = self.hparams.per_stats()
            self.mpd_metrics = MpdStats()
            self.mpd_metrics_seq = MpdStats()
   
    def compute_forward(self, batch, stage):
        """Forward pass supporting TRAIN, VALID, TEST, and INFERENCE stages.
        
        ╔════════════════════════════════════════════════════════════════════════════════╗
        ║                  TRAIN vs VALID vs TEST vs INFERENCE                          ║
        ╠════════════════════════════════════════════════════════════════════════════════╣
        ║                                                                                ║
        ║  TRAIN (sb.Stage.TRAIN):                                                     ║
        ║  ├─ Input: Audio + GT (canonical, perceived, mispro_label)                  ║
        ║  ├─ Use: Teacher forcing with GT targets                                    ║
        ║  ├─ Compute: All losses (CTC, seq2seq, mispro, GA, PCO)                     ║
        ║  └─ Purpose: Update model parameters via backprop                           ║
        ║                                                                                ║
        ║  VALID (sb.Stage.VALID):                                                     ║
        ║  ├─ Input: Audio + GT (canonical, perceived, mispro_label)                  ║
        ║  ├─ Use: Teacher forcing with GT targets for metric computation             ║
        ║  ├─ Compute: Losses + all metrics (PER, F1, MPD) using GT                   ║
        ║  └─ Purpose: Monitor training & select best model                           ║
        ║                                                                                ║
        ║  TEST (sb.Stage.TEST) - ✅ WITH Ground Truth:                                ║
        ║  ├─ Input: Audio + GT (if available in batch)                               ║
        ║  ├─ Use: Teacher forcing with GT targets for evaluation metrics              ║
        ║  ├─ Compute: Metrics (PER, F1, MPD) using GT                                ║
        ║  └─ Purpose: Evaluate on test set with GT available                         ║
        ║                                                                                ║
        ║  TEST (sb.Stage.TEST) - ❌ WITHOUT Ground Truth:                             ║
        ║  ├─ Input: ONLY audio (batch.id + batch.sig)                                ║
        ║  ├─ Use: Pure AR decoding (no teacher forcing)                              ║
        ║  ├─ Output: Only hypotheses (hyps, top_log_probs, top_lengths)              ║
        ║  └─ Purpose: Fallback to INFERENCE mode if GT not available                ║
        ║                                                                                ║
        ║  INFERENCE (hparams.inference_mode=True):                                    ║
        ║  ├─ Input: ONLY audio (batch.id + batch.sig)                                ║
        ║  ├─ NO ground truth available ❌                                             ║
        ║  ├─ Use: Encoder-only + pure AR decoding (no teacher forcing)               ║
        ║  ├─ Output: Only hypotheses (hyps, top_log_probs, top_lengths)              ║
        ║  └─ Purpose: Generate predictions for unlabeled data                        ║
        ║                                                                                ║
        ╚════════════════════════════════════════════════════════════════════════════════╝
        
        Summary:
        ┌────────────────────────────────────────────────────────────────────────────┐
        │ Stage       │ Has GT │ Teacher Forcing │ Compute Loss │ Compute Metrics   │
        ├────────────────────────────────────────────────────────────────────────────┤
        │ TRAIN       │ ✅    │ Yes             │ ✅          │ ❌                 │
        │ VALID       │ ✅    │ Yes             │ ✅          │ ✅                 │
        │ TEST (w/GT) │ ✅    │ Yes             │ ❌          │ ✅                 │
        │ TEST (no GT)│ ❌    │ No (AR only)    │ ❌          │ ❌                 │
        │ INFERENCE   │ ❌    │ No (AR only)    │ ❌          │ ❌                 │
        └────────────────────────────────────────────────────────────────────────────┘
        """
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        
        # Extract SSL features

        feats = self.modules.perceived_ssl(wavs)  # [B, T_s, ENC_DIM]
        if len(feats.shape) == 4: 
            feats = feats[self.hparams.preceived_ssl_emb_layer]
        
        current_epoch = self.hparams.epoch_counter.current


        # check why enc_out -> ctc get's lower performance than original ctc path
        # with torch.no_grad():
        feats_ssl_proj = self.modules.enc(feats)
        enc_out_ssl_proj, _ = self.modules.ConformerEncoder(feats_ssl_proj)
            
        
        # ==================== Initialize all output variables ====================
        p_ctc_logits = None
        p_seq_logits = None
        p_mispro_bin_logits = None
        p_mispro_cls_logits = None
        h_mispro_bin = None
        h_mispro_cls = None
        enc_out = None
        dec_out = None
        Cano_emb = None
        fuse_attn = None
        fuse_attn_dec = None
        hyps = None
        attn_map = None
        top_log_probs = None
        top_lengths = None
        
        # ==================== TRAINING Stage ====================
        if sb.Stage.TRAIN == stage:
            targets, target_lens = batch.phn_encoded_target
            targets_bos, target_lens_bos = batch.phn_encoded_target_bos
            targets_eos, target_lens_eos = batch.phn_encoded_target_eos
            canonicals, canonical_lens = batch.phn_encoded_canonical
            perceiveds_bos, perceived_lens_bos = batch.phn_encoded_perceived_bos
            perceiveds_eos, perceived_lens_eos = batch.phn_encoded_perceived_eos
            mispro_label, mispro_label_lens = batch.mispro_label
            
            # Create canonical phoneme embeddings
            Cano_emb = self.modules.phn_emb(canonicals)  # [B, T_c, D]
            
            if self.hparams.decoder_target == "perceived":
                targets_bos = perceiveds_bos
                target_lens_bos = perceived_lens_bos
            
            # Forward through TransASR encoder+decoder
            allow_ASR_hidden = getattr(self.hparams, "output_ASR_hidden_state", False)
            
            outs = self.modules.TransASR(
                    src=feats,
                    tgt=targets_bos,
                    wav_len=wav_lens,
                    pad_idx=self.hparams.blank_index,
            )

            # self.modules.TransASR.encoder.state_dict()['norm.norm.bias']
            # self.hparams.pretrainer.loadables['model'][1].state_dict()['norm.norm.bias']
            # pdb.set_trace()
            
            if allow_ASR_hidden:
                enc_out, hidden_outs, dec_out = outs
            else:
                enc_out, dec_out = outs
        


            # ==================== Train: Fuse canonical embeddings with audio features ====================
            if "enc" in self.hparams.fuse_enc_or_dec:
                memory = enc_out
                memory = self.modules.mem_proj(memory)  # [B, T_s, D]
                # project post encoder
                if self.hparams.post_encoder_reduction_factor >= 1:
                    # 使用Conv1d在时间维度降采样，保持D维度不变
                    import torch.nn.functional as F
                    # factor = self.hparams.post_encoder_reduction_factor
                    # B, T, D = memory.shape
    
                    memory = self.modules.mem_proj_cnn_post_enc(memory)
                
                fuse_feat, _,  fuse_attn = self.modules.fuse_net(
                    tgt=Cano_emb,
                    memory=memory,
                    tgt_key_padding_mask=make_pad_mask(Cano_emb.shape[1] * mispro_label_lens, maxlen=Cano_emb.shape[1]).to(self.device),
                    memory_key_padding_mask=make_pad_mask(memory.shape[1] * wav_lens, maxlen=memory.shape[1]).to(self.device),
                    pos_embs_tgt=RelPosEncXL(emb_dim=self.hparams.dnn_neurons)(Cano_emb).to(self.device),
                    pos_embs_src=RelPosEncXL(emb_dim=self.hparams.dnn_neurons)(memory).to(self.device)
                )# [B, T_c, D]
            if "dec" in self.hparams.fuse_enc_or_dec:
                memory = dec_out
                # LayerNorm and Positional Embedding
                memory = self.modules.mem_proj_dec(memory)  # [B, T_p, D]
                # tgt_causal_mask = get_lookahead_mask(Cano_emb)
                fuse_feat_dec, _,  fuse_attn_dec = self.modules.fuse_net_dec(
                    tgt=Cano_emb,
                    memory=memory,
                    tgt_key_padding_mask=make_pad_mask(Cano_emb.shape[1] * mispro_label_lens, maxlen=Cano_emb.shape[1]).to(self.device),
                    memory_key_padding_mask=make_pad_mask(memory.shape[1] * mispro_label_lens, maxlen=memory.shape[1]).to(self.device),
                    pos_embs_tgt=RelPosEncXL(emb_dim=self.hparams.dnn_neurons)(Cano_emb).to(self.device),
                    pos_embs_src=RelPosEncXL(emb_dim=self.hparams.dnn_neurons)(memory).to(self.device)
                )# [B, T_c, D]
            
            if "enc" in self.hparams.fuse_enc_or_dec and "dec" in self.hparams.fuse_enc_or_dec:
                # concat fuse_feat from enc and dec
                fuse_feat_ = torch.cat((fuse_feat, fuse_feat_dec), dim=-1)
                fuse_feat = self.modules.fuse_proj(fuse_feat_)  # [B, T, D]
                
                # tgt_mask=tgt_causal_mask,
                # print("Warning: No fuse net is used!")
            
            # If dec only, use dec's fuse_feat
            if self.hparams.fuse_enc_or_dec == "dec":
                fuse_feat = fuse_feat_dec
                
            h_mispro = self.hparams.mispro_head(fuse_feat.transpose(1, 2))
            # for binary detection, 
            h_mispro_bin = self.hparams.mispro_head_binary_out(h_mispro)
            h_mispro_bin = h_mispro_bin.transpose(1, 2)  # [B, T_c, 1]
            # for multi-class detection, 4 classes
            h_mispro_cls = self.hparams.mispro_head_class_out(h_mispro.transpose(1, 2)) #[B, T_c, 4]
            
            # p_mispro_logits = torch.nn.functional.log_softmax(h_mispro)  # Log probabilities for 4 Classes
            p_mispro_bin_logits = torch.nn.functional.sigmoid(h_mispro_bin)  # Log probabilities for binary
            p_mispro_cls_logits = torch.nn.functional.log_softmax(h_mispro_cls, dim=-1)

            # CTC head
            # from speechbrain.lobes.models.transformer.TransformerASR import make_transformer_src_tgt_masks
            # src_key_padding_mask, tgt_key_padding_mask, src_mask, tgt_mask,= make_transformer_src_tgt_masks(
            #     src=feats_ssl_proj,
            #     tgt=targets_bos,
            #     wav_len=wav_lens,
            #     causal=False,
            #     pad_idx=self.hparams.blank_index,
            # )
            # src_key_padding_mask_causal, tgt_key_padding_mask_causal, src_mask_causal, tgt_mask_causal,= make_transformer_src_tgt_masks(
            #     src=feats_ssl_proj,
            #     tgt=targets_bos,
            #     wav_len=wav_lens,
            #     causal=True,
            #     pad_idx=self.hparams.blank_index,
            # )
            
            # enc_out2 = self.hparams.TransASR.encoder(src=feats_ssl_proj,
            #                                          src_mask=src_mask_causal,
            #                                          src_key_padding_mask=src_key_padding_mask_causal,
            #                                          pos_embs=None
            #                                          )[0]
            # import matplotlib.pyplot as plt
            # plt.imshow(src_key_padding_mask.cpu().numpy(), aspect='auto')
            # plt.savefig('./src_key_padding_mask.png')
            # plt.imshow(src_mask.cpu().numpy(), aspect='auto')
            # plt.savefig('./src_mask.png')
            # plt.imshow(src_key_padding_mask_causal.cpu().numpy(), aspect='auto')
            # plt.savefig('./src_key_padding_mask_causal.png')
            # plt.imshow(src_mask_causal.cpu().numpy(), aspect='auto')
            # plt.savefig('./src_mask_causal.png')
            
            # import pdb; pdb.set_trace()
            # h_ctc_feat = self.modules.ctc_lin(enc_out_ssl_proj)  # [B, T_s, C]
            h_ctc_feat = self.modules.ctc_lin(enc_out)  # [B, T_s, C]
            p_ctc_logits = self.hparams.log_softmax(h_ctc_feat)  # Log probabilities
            # p_ctc_logits_2 = self.hparams.log_softmax(h_ctc_feat_2)  # Log probabilities
            # import pdb; pdb.set_trace()

            # seq2seq head
            h_seq_feat = self.modules.d_out(dec_out)  # [B, T_p+1, C]
            p_seq_logits = self.hparams.log_softmax(h_seq_feat)  # Log probabilities

        else:
            with torch.no_grad():
                # ==================== VALIDATION Stage: Full model with GT info ====================
                if stage == sb.Stage.VALID:
                    # Extract all ground truth information available during validation
                    targets_eos, target_lens_eos = batch.phn_encoded_target_eos
                    targets_bos, target_lens_bos = batch.phn_encoded_target_bos
                    canonicals, canonical_lens = batch.phn_encoded_canonical
                    perceiveds_bos, perceived_lens_bos = batch.phn_encoded_perceived_bos
                    perceiveds_eos, perceived_lens_eos = batch.phn_encoded_perceived_eos
                    mispro_label, mispro_label_lens = batch.mispro_label
                    
                    # Create canonical embeddings for validation
                    Cano_emb = self.modules.phn_emb(canonicals)  # [B, T_c, D]
                    
                    if self.hparams.decoder_target == "perceived":
                        targets_bos = perceiveds_bos
                        target_lens_bos = perceived_lens_bos
                    
                    # Forward pass with GT targets for teacher forcing
                    allow_ASR_hidden = getattr(self.hparams, "output_ASR_hidden_state", False)
                    outs = self.modules.TransASR(
                            src=feats,
                            tgt=targets_bos,
                            wav_len=wav_lens,
                            pad_idx=0,
                    )
                    
                    if allow_ASR_hidden:
                        enc_out, hidden_outs, dec_out = outs
                    else:
                        enc_out, dec_out = outs
                    
                    # ==================== Validation: Fuse canonical embeddings with audio features ====================
                    if "enc" in self.hparams.fuse_enc_or_dec:
                        memory = enc_out
                        memory = self.modules.mem_proj(memory)  # [B, T_s, D]
                        
                        if self.hparams.post_encoder_reduction_factor >= 1:
                            memory = self.modules.mem_proj_cnn_post_enc(memory)
                        
                        fuse_feat, _,  fuse_attn = self.modules.fuse_net(
                            tgt=Cano_emb,
                            memory=memory,
                            tgt_key_padding_mask=make_pad_mask(Cano_emb.shape[1] * mispro_label_lens, maxlen=Cano_emb.shape[1]).to(self.device),
                            memory_key_padding_mask=make_pad_mask(memory.shape[1] * wav_lens, maxlen=memory.shape[1]).to(self.device),
                            pos_embs_tgt=RelPosEncXL(emb_dim=self.hparams.dnn_neurons)(Cano_emb).to(self.device),
                            pos_embs_src=RelPosEncXL(emb_dim=self.hparams.dnn_neurons)(memory).to(self.device)
                        )# [B, T_p, D]

                    if "dec" in self.hparams.fuse_enc_or_dec:
                        memory = dec_out
                        memory = self.modules.mem_proj_dec(memory)  # [B, T_p, D]
                        fuse_feat_dec, _,  fuse_attn_dec = self.modules.fuse_net_dec(
                            tgt=Cano_emb,
                            memory=memory,
                            tgt_key_padding_mask=make_pad_mask(Cano_emb.shape[1] * mispro_label_lens, maxlen=Cano_emb.shape[1]).to(self.device),
                            memory_key_padding_mask=make_pad_mask(memory.shape[1] * mispro_label_lens, maxlen=memory.shape[1]).to(self.device),
                            pos_embs_tgt=RelPosEncXL(emb_dim=self.hparams.dnn_neurons)(Cano_emb).to(self.device),
                            pos_embs_src=RelPosEncXL(emb_dim=self.hparams.dnn_neurons)(memory).to(self.device)
                        ) # [B, T_p, D]               
                    
                    if "enc" in self.hparams.fuse_enc_or_dec and "dec" in self.hparams.fuse_enc_or_dec:
                        fuse_feat_ = torch.cat((fuse_feat, fuse_feat_dec), dim=-1)
                        fuse_feat = self.modules.fuse_proj(fuse_feat_)  # [B, T, D]
                    
                    if self.hparams.fuse_enc_or_dec == "dec":
                        fuse_feat = fuse_feat_dec
                    
                    # ==================== Validation: Mispronunciation head outputs ====================
                    h_mispro = self.hparams.mispro_head(fuse_feat.transpose(1, 2))
                    h_mispro_bin = self.hparams.mispro_head_binary_out(h_mispro)
                    h_mispro_bin = h_mispro_bin.transpose(1, 2)
                    h_mispro_cls = self.hparams.mispro_head_class_out(h_mispro.transpose(1, 2))
                    
                    p_mispro_bin_logits = torch.nn.functional.sigmoid(h_mispro_bin)
                    p_mispro_cls_logits = torch.nn.functional.log_softmax(h_mispro_cls, dim=-1)
                    
                    # ==================== Validation: CTC and Seq2Seq head outputs ====================
                    h_ctc_feat = self.modules.ctc_lin(enc_out)  # [B, T_s, C]
                    # h_ctc_feat = self.modules.ctc_lin(enc_out_ssl_proj)  # [B, T_s, C]
                    p_ctc_logits = self.hparams.log_softmax(h_ctc_feat)

                    h_seq_feat = self.modules.d_out(dec_out)  # [B, T_p+1, C]
                    p_seq_logits = self.hparams.log_softmax(h_seq_feat)

                    # ==================== Validation: Get greedy hypothesis ====================
                    if self.hparams.valid_search_interval > 0 and (current_epoch % self.hparams.valid_search_interval == 0) and self.hparams.valid_decode_mode == "AR":
                        # AR decoding
                        hyps, top_lengths, top_scores, top_log_probs = self.hparams.valid_search(
                                enc_out.detach(), wav_lens
                            )
                        # Teacher forcing decoding fallback
                        # hyps = p_seq_logits.argmax(dim=-1)  # [B, T_p+1]
                        # from speechbrain.utils.data_utils import undo_padding
                        # hyps = undo_padding(hyps, target_lens_bos)
                    elif self.hparams.valid_decode_mode == "teacher_forcing":
                        # fallback to teacher forcing decoding
                        hyps = p_seq_logits.argmax(dim=-1)  # [B, T_p+1]
                        from speechbrain.utils.data_utils import undo_padding
                        hyps = undo_padding(hyps, target_lens_bos)
                    # import pdb; pdb.set_trace()
                    attn_map = None
                    
                    # ==================== Validation: Optional attention visualization ====================
                    plot_interval = self.hparams.plot_attention_interval
                    if self.hparams.plot_attention and (current_epoch % plot_interval == 0):
                        import random
                        select_id = random.choice(range(len(batch.id)))
                        if fuse_attn is not None:
                            for index, (attn, c_id) in enumerate(zip(fuse_attn[-1], batch.id)):
                                if index != select_id:
                                    continue
                                from pathlib import Path
                                if len(attn.shape) == 2:
                                    attn = attn.unsqueeze(0)
                                c_id = "_".join(c_id.split("/")[-3:])
                                output_dir = Path(self.hparams.valid_attention_plot_dir) / f"{current_epoch:03d}"
                                plot_attention(attn.cpu(), self.hparams.nhead, c_id, output_dir)
                        
                        if fuse_attn_dec is not None:
                            for index, (attn, c_id) in enumerate(zip(fuse_attn_dec[-1], batch.id)):
                                if index != select_id:
                                    continue
                                from pathlib import Path
                                if len(attn.shape) == 2:
                                    attn = attn.unsqueeze(0)
                                c_id = "_".join(c_id.split("/")[-3:])
                                output_dir = Path(self.hparams.valid_attention_plot_dir) / f"{current_epoch:03d}_dec"
                                plot_attention(attn.cpu(), self.hparams.nhead, c_id, output_dir)

                # ==================== TEST Stage: Has GT for evaluation ====================
                elif stage == sb.Stage.TEST:
                    # ✅ TEST has ground truth available (batch contains canonical, mispro_label, etc.)
                    # Use them for evaluation/metrics, but NOT for generation
                    
                    # Try to extract GT info if available (for evaluation purposes)
                    try:
                        targets_eos, target_lens_eos = batch.phn_encoded_target_eos
                        targets_bos, target_lens_bos = batch.phn_encoded_target_bos
                        canonicals, canonical_lens = batch.phn_encoded_canonical
                        perceiveds_bos, perceived_lens_bos = batch.phn_encoded_perceived_bos
                        perceiveds_eos, perceived_lens_eos = batch.phn_encoded_perceived_eos
                        mispro_label, mispro_label_lens = batch.mispro_label
                        has_gt = True
                    except:
                        # If GT is not available, fall through to INFERENCE mode
                        has_gt = False
                    
                    if has_gt:
                        # ==================== TEST with GT: Use GT for evaluation ====================
                        # Compute SSL projection in test stage
                        feats_ssl_proj = self.modules.enc(feats)
                        enc_out_ssl_proj, _ = self.modules.ConformerEncoder.forward(feats_ssl_proj)
                        
                        # Create canonical embeddings for evaluation
                        Cano_emb = self.modules.phn_emb(canonicals)  # [B, T_c, D]
                        
                        if self.hparams.decoder_target == "perceived":
                            targets_bos = perceiveds_bos
                            target_lens_bos = perceived_lens_bos
                        
                        # Forward pass with GT targets (for computing fusion features needed for mispro head)
                        allow_ASR_hidden = getattr(self.hparams, "output_ASR_hidden_state", False)
                        outs = self.modules.TransASR(
                                src=feats,
                                tgt=targets_bos,
                                wav_len=wav_lens,
                                pad_idx=0,
                        )
                        
                        if allow_ASR_hidden:
                            enc_out, hidden_outs, dec_out = outs
                        else:
                            enc_out, dec_out = outs
                        
                        # ==================== TEST: Fuse canonical embeddings with audio features ====================
                        if "enc" in self.hparams.fuse_enc_or_dec:
                            memory = enc_out
                            memory = self.modules.mem_proj(memory)  # [B, T_s, D]
                            
                            if self.hparams.post_encoder_reduction_factor >= 1:
                                
                                # transpose applied inside mem_proj_cnn_post_enc
                                memory = self.modules.mem_proj_cnn_post_enc(memory)
                                
                            
                            fuse_feat, _,  fuse_attn = self.modules.fuse_net(
                                tgt=Cano_emb,
                                memory=memory,
                                tgt_key_padding_mask=make_pad_mask(Cano_emb.shape[1] * mispro_label_lens, maxlen=Cano_emb.shape[1]).to(self.device),
                                memory_key_padding_mask=make_pad_mask(memory.shape[1] * wav_lens, maxlen=memory.shape[1]).to(self.device),
                                pos_embs_tgt=RelPosEncXL(emb_dim=self.hparams.dnn_neurons)(Cano_emb).to(self.device),
                                pos_embs_src=RelPosEncXL(emb_dim=self.hparams.dnn_neurons)(memory).to(self.device)
                            )# [B, T_p, D]

                        if "dec" in self.hparams.fuse_enc_or_dec:
                            memory = dec_out
                            memory = self.modules.mem_proj_dec(memory)  # [B, T_p, D]
                            fuse_feat_dec, _,  fuse_attn_dec = self.modules.fuse_net_dec(
                                tgt=Cano_emb,
                                memory=memory,
                                tgt_key_padding_mask=make_pad_mask(Cano_emb.shape[1] * mispro_label_lens, maxlen=Cano_emb.shape[1]).to(self.device),
                                memory_key_padding_mask=make_pad_mask(memory.shape[1] * mispro_label_lens, maxlen=memory.shape[1]).to(self.device),
                                pos_embs_tgt=RelPosEncXL(emb_dim=self.hparams.dnn_neurons)(Cano_emb).to(self.device),
                                pos_embs_src=RelPosEncXL(emb_dim=self.hparams.dnn_neurons)(memory).to(self.device)
                            ) # [B, T_p, D]               
                        
                        if "enc" in self.hparams.fuse_enc_or_dec and "dec" in self.hparams.fuse_enc_or_dec:
                            fuse_feat_ = torch.cat((fuse_feat, fuse_feat_dec), dim=-1)
                            fuse_feat = self.modules.fuse_proj(fuse_feat_)  # [B, T, D]
                        
                        if self.hparams.fuse_enc_or_dec == "dec":
                            fuse_feat = fuse_feat_dec
                        
                        # ==================== TEST: Mispronunciation head outputs ====================
                        h_mispro = self.hparams.mispro_head(fuse_feat.transpose(1, 2))
                        h_mispro_bin = self.hparams.mispro_head_binary_out(h_mispro)
                        h_mispro_bin = h_mispro_bin.transpose(1, 2)
                        h_mispro_cls = self.hparams.mispro_head_class_out(h_mispro.transpose(1, 2))
                        
                        p_mispro_bin_logits = torch.nn.functional.sigmoid(h_mispro_bin)
                        p_mispro_cls_logits = torch.nn.functional.log_softmax(h_mispro_cls, dim=-1)
                        
                        # ==================== TEST: CTC and Seq2Seq head outputs ====================
                        h_ctc_feat = self.modules.ctc_lin(enc_out)  # [B, T_s, C]
                        # h_ctc_feat = self.modules.ctc_lin(enc_out_ssl_proj)  # [B, T_s, C]
                        p_ctc_logits = self.hparams.log_softmax(h_ctc_feat)

                        h_seq_feat = self.modules.d_out(dec_out)  # [B, T_p+1, C]
                        p_seq_logits = self.hparams.log_softmax(h_seq_feat)

                        # ==================== TEST: Get greedy hypothesis from logits ====================
                        # hyps = p_seq_logits.argmax(dim=-1)  # [B, T_p+1]
                        hyps, top_lengths, top_scores, top_log_probs = self.hparams.test_search(
                            enc_out.detach(), wav_lens
                        )
                        # from speechbrain.utils.data_utils import undo_padding
                        # hyps = undo_padding(hyps, target_lens_bos)
                        # attn_map = None
                        
                        top_log_probs = None
                        top_lengths = None
                    else:
                        # ==================== TEST without GT: Fall through to INFERENCE ====================
                        # Get encoder output only
                        enc_out = self.modules.TransASR.encode(feats)  # [B, T_s, D]
                        
                        if len(enc_out.shape) == 2:
                            enc_out = enc_out.unsqueeze(0)
                        
                        # Compute CTC logits for acoustic model output
                        h_ctc_feat = self.modules.ctc_lin(enc_out_ssl_proj)  # [B, T_s, C]
                        p_ctc_logits = self.hparams.log_softmax(h_ctc_feat)
                        
                        # Pure AR decoding
                        if self.hparams.test_decode_mode == "AR":
                            hyps, top_lengths, top_scores, top_log_probs = self.hparams.test_search(
                                enc_out.detach(), wav_lens
                            )
                        p_seq_logits = None
                        p_mispro_bin_logits = None
                        p_mispro_cls_logits = None
                        h_mispro_bin = None
                        h_mispro_cls = None
                        dec_out = None
                        Cano_emb = None
                        fuse_attn = None
                        fuse_attn_dec = None
                        attn_map = None

                # ==================== INFERENCE Stage: ❌ NO GT available ====================
                elif hasattr(self.hparams, 'inference_mode') and self.hparams.inference_mode:
                    # Pure inference mode - only audio input (batch.id + batch.sig)
                    # No ground truth available - cannot compute losses or evaluation metrics
                    
                    # Compute SSL projection in inference stage
                    feats_ssl_proj = self.modules.enc(feats)
                    enc_out_ssl_proj, _ = self.modules.ConformerEncoder.forward(feats_ssl_proj)
                    
                    # Get encoder output via TransASR.encode() - only requires feats and wav_lens
                    enc_out = self.modules.TransASR.encode(feats)  # [B, T_s, D]
                    
                    # Ensure enc_out is 3D [B, T, D]
                    if len(enc_out.shape) == 2:
                        enc_out = enc_out.unsqueeze(0)
                    
                    # ==================== INFERENCE: Pure AR decoding ====================
                    # Generate hypothesis using only encoder output (no teacher forcing, no GT)
                    if self.hparams.test_decode_mode == "AR":
                        hyps, top_lengths, top_scores, top_log_probs = self.hparams.test_search(
                            enc_out.detach(), wav_lens
                        )
                    
                    # Initialize other outputs to None for inference (no GT available)
                    p_ctc_logits = None
                    p_seq_logits = None
                    p_mispro_bin_logits = None
                    p_mispro_cls_logits = None
                    h_mispro_bin = None
                    h_mispro_cls = None
                    dec_out = None
                    Cano_emb = None
                    fuse_attn = None
                    fuse_attn_dec = None
                    attn_map = None

        # pdb.set_trace()
        return {
            "p_ctc_feat": p_ctc_logits,  # [B, T_s, C]
            "p_dec_out": p_seq_logits,  # [B, T_p+1, C]
            # "p_ctc_feat": h_ctc_feat,  # [B, T_s, C]
            # "p_dec_out": h_seq_feat,  # [B, T_p+1, C]
            # "feats": feats,  # [B, T_s, D]
            "attn_map": attn_map,  # [B, T_p+1, T_s] or similar
            "hyps": hyps,  # [B, T_p+1] or None if not applicable
            # "p_mispro_logits": p_mispro_logits,
            "p_mispro_bin_logits": p_mispro_bin_logits,
            "p_mispro_cls_logits": p_mispro_cls_logits,
            "h_mispro_bin": h_mispro_bin,
            "h_mispro_cls": h_mispro_cls,
            
            "fuse_attn": fuse_attn if 'fuse_attn' in locals() else None,
            "fuse_attn_dec": fuse_attn_dec if 'fuse_attn_dec' in locals() else None,
            "enc_out": enc_out,
            "dec_out": dec_out,
            "Cano_emb": Cano_emb,
            "top_log_probs": top_log_probs if 'top_log_probs' in locals() else None,
            "top_lengths": top_lengths if 'top_lengths' in locals() else None
        }
        
    def compute_objectives(self, predictions, batch, stage):
        "Computes the loss for the model."
        current_epoch = self.hparams.epoch_counter.current
        valid_search_interval = self.hparams.valid_search_interval
        
        p_ctc_feat = predictions["p_ctc_feat"] if "p_ctc_feat" in predictions else None
        p_dec_out = predictions["p_dec_out"] if "p_dec_out" in predictions else None
        # feats = predictions["feats"]
        attn_map = predictions["attn_map"] if "attn_map" in predictions else None
        # pdb.set_trace()
        hyps = predictions.get("hyps", [])  # [B, T_p+1] or None if not applicable
        # p_mispro_logits = predictions["p_mispro_logits"]  # [B, T_c, C]
        p_mispro_bin_logits = predictions["p_mispro_bin_logits"]  # [B, T_c, 1]
        p_mispro_cls_logits = predictions["p_mispro_cls_logits"]  # [B, T_c, 4]
        # p_mispro_logits = torch.cat((p_mispro_bin_logits, p_mispro_cls_logits), dim=-1)
        
        h_mispro_bin = predictions["h_mispro_bin"]  # [B, T_c, D]
        h_mispro_cls = predictions["h_mispro_cls"]
        # h_mispro = predictions["h_mispro"]  # [B, T_c, C]
        fuse_attn = predictions["fuse_attn"]  # [B, T_c, T_s] or similar
        fuse_attn_dec = predictions["fuse_attn_dec"]  # [B, T_c, T_p] or similar
        enc_out = predictions["enc_out"]  # [B, T_s, D]
        dec_out = predictions["dec_out"]  # [B, T_p+1, D]
        Cano_emb = predictions["Cano_emb"]  # [B, T_c, D]
        top_log_probs = predictions["top_log_probs"]
        top_lengths = predictions["top_lengths"]
        
        wavs, wav_lens = batch.sig
        targets, target_lens = batch.phn_encoded_target
        targets_bos, target_lens_bos = batch.phn_encoded_target_bos
        targets_eos, target_lens_eos = batch.phn_encoded_target_eos
        
        canonicals, canonical_lens = batch.phn_encoded_canonical
        canonicals_bos, canonical_lens_bos = batch.phn_encoded_canonical_bos
        canonicals_eos, canonical_lens_eos = batch.phn_encoded_canonical_eos
        perceiveds, perceived_lens = batch.phn_encoded_perceived
        perceiveds_bos, perceived_lens_bos = batch.phn_encoded_perceived_bos
        perceiveds_eos, perceived_lens_eos = batch.phn_encoded_perceived_eos
        ids = batch.id
        
        # assume mispro_label is 4 classes: 0-no, 1-sub, 2-del, 3-ins
        mispro_label, mispro_label_lens = batch.mispro_label
        mispro_label_bin = (mispro_label > 0).long()  # Convert to binary: 0 (no), 1 (any mispronunciation)
        

        # Caculate the loss for CTC and seq2seq outputs
        # loss_ctc = self.hparams.ctc_cost(p_ctc_feat, targets, wav_lens, target_lens
        if self.hparams.ctc_head_target == "perceived":
            loss_ctc = self.hparams.ctc_cost(p_ctc_feat, targets, wav_lens, target_lens)
        elif self.hparams.ctc_head_target == "canonical":
            loss_ctc = self.hparams.ctc_cost(p_ctc_feat, canonicals, wav_lens, canonical_lens)
        
        if self.hparams.decoder_target == "perceived":
            loss_dec_out = self.hparams.seq_cost(p_dec_out, perceiveds_eos, perceived_lens_eos)
        else:
            loss_dec_out = self.hparams.seq_cost(p_dec_out, targets_eos, target_lens_eos)
            
        loss_mispro = self.hparams.mispro_cost_bin(inputs=h_mispro_bin,
                                               targets=mispro_label_bin, 
                                               length=mispro_label_lens)
        
        loss_mispro_cls = self.hparams.mispro_cost_cls(p_mispro_cls_logits, mispro_label, mispro_label_lens)
        loss_mispro_all = loss_mispro + loss_mispro_cls
        
        # guided attention loss on ga
        # loss_ga_model = sb.nnet.loss.guidedattn_loss.GuidedAttentionLoss(sigma=0.2)
        # pdb.set_trace()
        loss_ga = 0
        # Use Last Layer's MHA 
        if "enc" in self.hparams.fuse_enc_or_dec:
            attn_last = fuse_attn[-1] 
            attn_ga = attn_last.mean(dim=1)  # [B, T_p]
            # FUSE_attn [B, T_p, T_s]
            loss_ga += self.hparams.ga_cost()(attn_ga, 
                                           target_lengths=(canonical_lens * Cano_emb.shape[1]).int(),
                                           input_lengths=((wav_lens * enc_out.shape[1]) / self.hparams.post_encoder_reduction_factor).int()
                                            )
        if "dec" in self.hparams.fuse_enc_or_dec:
            # FUSE_attn [B, T_p, T_p]
            attn_last_dec = fuse_attn_dec[-1]
            attn_ga = attn_last_dec.mean(dim=1)  # [B, T_p]
            loss_ga += self.hparams.ga_cost()(attn_ga, 
                                           target_lengths=(canonical_lens * Cano_emb.shape[1]).int(),
                                           input_lengths=(target_lens_eos * dec_out.shape[1]).int()
                                            )
        # FUSE_attn [B, T_p, T_p]
        
        # ConPCO loss
        # loss_phn_pco, loss_center_clap = loss_pco(phn_audio_feats, phn_text_feats, phn_label, phns)
        # use audio_feats: after SSL projector, after SSL Encoder, or after ASR Encoder
        # assume the duration must be 1:1 for cano phn and mispro_label
        # phn_audio_feats.shape = phn_text_feats.shape [B, T, D]
        # phn_label [B. T]
        # phns [B, T]
        # default ignore index = -1
        # audio = dec_out, text = Cano_emb, phn_label = mispro_label, phns = canonicals
        
        if self.hparams.decoder_target == "target":
            tgt_emb = self.modules.TransASR.custom_tgt_module(targets_eos)
        else:
            tgt_emb = self.modules.TransASR.custom_tgt_module(perceiveds_eos)
        # add tgt_pos_emb
        if (
            self.modules.TransASR.attention_type == "RelPosMHAXL"
            or self.modules.TransASR.attention_type == "RoPEMHA"
        ):
            tgt_emb = tgt_emb + self.modules.TransASR.positional_encoding_decoder(tgt_emb)
        elif (
            self.modules.TransASR.positional_encoding_type == "fixed_abs_sine"
            or self.modules.TransASR.attention_type == "hypermixing"
        ):
            tgt_emb = tgt_emb + self.modules.TransASR.positional_encoding(tgt_emb)

        tgt_emb_for_pco = self.modules.conpco_proj_perceived_phn_feat(tgt_emb)
        # apply l2 norm
        tgt_emb_for_pco = F.normalize(tgt_emb_for_pco, p=2, dim=-1)
        
        # if also apply decode positional embedding  proj on dec_out
        # pdb.set_trace()
        if getattr(self.hparams, "conpco_use_dec_pos_emb_on_audio_feat", False):
            # add dec_out pos emb
            if (
                self.modules.TransASR.attention_type == "RelPosMHAXL"
                or self.modules.TransASR.attention_type == "RoPEMHA"
            ):
                dec_out_proj = dec_out + self.modules.TransASR.positional_encoding_decoder(dec_out)
            elif (
                self.modules.TransASR.positional_encoding_type == "fixed_abs_sine"
                or self.modules.TransASR.attention_type == "hypermixing"
            ):
                dec_out_proj = dec_out + self.modules.TransASR.positional_encoding(dec_out)
        else:
            dec_out_proj = dec_out

        audio_feats_for_pco = self.modules.conpco_proj_audio_feat(dec_out_proj)
        # apply l2 norm
        audio_feats_for_pco = F.normalize(audio_feats_for_pco, p=2, dim=-1)
        
        # TODO: use real gt
        # current solution, as we are using target_eos as decoder input, assume they are all high score.
        # make tensor with same shape as targets_eos, all 1.0 and pad the relative positions with 0.0
        dummy_gt = torch.ones_like(targets_eos).float().to(self.device) * 2 # high score 2.0, mid score 1.0, low score 0.0
        dummy_gt = dummy_gt.masked_fill(targets_eos == 0, 0.0)
        # pdb.set_trace()
        # pdb.set_trace()
        loss_phn_pco, loss_center_clap = self.hparams.conpco_cost(
            features=audio_feats_for_pco.float(), 
            features_text=tgt_emb_for_pco.float(), 
            gt=dummy_gt, 
            phn_id=targets_eos,
        ).values()
        
        from utils.plot.plot_clap import plot_clap_clusters, plot_phone_cluster, plot_phoneme_centroids_with_instances
        # TODOS:
        '''
        1. Try / Use Dec_out's feature and Tgt Embedding for conpco learning.
        2. Check feature norm  expectally tgt_module, before PCO loss calculation.
        3. Better to ignore silence index in PCO loss calculation.
        '''
        # import pdb; pdb.set_trace()
        try:
            sil_index = self.label_encoder.lab2ind["sil"]
        except:
            try:
                sil_index = self.label_encoder.lab2ind["<sil>"]
            except:
                sil_index = None
                print("Silence index not found in label encoder.")
        
        if current_epoch % self.hparams.plot_conpco_interval == 0 and self.hparams.plot_conpco:
            from pathlib import Path
            
            if stage == sb.Stage.VALID:
                output_dir = Path(self.hparams.conpco_plot_dir) / "valid" / f"{current_epoch:03d}"
                # use the batch first sample to plot (keep batch dimension with unsqueeze)
                # fig, ax, fig_zoom, ax_zoom = plot_phoneme_centroids_with_instances(
                #     audio_feats=audio_feats_for_pco.detach().float().cpu()[0, :, :].unsqueeze(0),
                #     phoneme_feats=tgt_emb_for_pco.detach().float().cpu()[0, :, :].unsqueeze(0),
                #     phoneme_labels=targets_eos.detach().cpu()[0].unsqueeze(0),
                #     phone_scores=dummy_gt.detach().cpu()[0].unsqueeze(0),
                #     ignore_index=[self.hparams.blank_index, sil_index],
                #     max_phones=self.hparams.max_phones,
                #     show_audio_centroid=self.hparams.conPCO_plot_show_audio_centroid,
                #     show_audio_scatter=self.hparams.conPCO_plot_show_audio_scatter,
                #     show_phoneme_centroid=self.hparams.conPCO_plot_show_phoneme_centroid,
                #     show_phoneme_scatter=self.hparams.conPCO_plot_show_phoneme_scatter,
                # )
                # pdb.set_trace()
                fig, ax, fig_zoom, ax_zoom = plot_phoneme_centroids_with_instances(
                    audio_feats=audio_feats_for_pco.detach().float().cpu(), 
                    phoneme_feats=tgt_emb_for_pco.detach().float().cpu(), 
                    phoneme_labels=targets_eos.detach().cpu(), 
                    phone_scores=dummy_gt.detach().cpu(),
                    ignore_index=[self.hparams.blank_index, sil_index, self.hparams.bos_index, self.hparams.eos_index],
                    max_phones=self.hparams.max_phones,
                    show_audio_centroid=self.hparams.conPCO_plot_show_audio_centroid,
                    show_audio_scatter=self.hparams.conPCO_plot_show_audio_scatter,
                    show_phoneme_centroid=self.hparams.conPCO_plot_show_phoneme_centroid,
                    show_phoneme_scatter=self.hparams.conPCO_plot_show_phoneme_scatter,
                    # reduction_method="umap"
                )
                
                output_dir.mkdir(parents=True, exist_ok=True)
                # get current batch id
                # get batch_ids = [os.path.splitext(os.path.basename(id))[0] for id in ids]
                batch_ids = [os.path.splitext(os.path.basename(id))[0] for id in ids]
                representative_id = batch_ids[0]
                
                fig.savefig(output_dir / f"epoch{current_epoch}_{representative_id}_phoneme_centroids.png")
                fig_zoom.savefig(output_dir / f"epoch{current_epoch}_{representative_id}_phoneme_centroids_zoom.png")
            
        # no mispro loss nor guided attention loss, or ordinal, only PCO
        loss = (
            self.hparams.ctc_weight * loss_ctc
            + (1 - self.hparams.ctc_weight) * loss_dec_out
            + loss_mispro_all 
            + loss_ga * 10
            + loss_phn_pco
            + loss_center_clap
        )
        # loss = loss_ctc
        
            # + loss_phn_pco 

        if stage != sb.Stage.TRAIN:
            if current_epoch % valid_search_interval == 0 or (
                    stage == sb.Stage.TEST
                ):
                    # Record losses for posterit
                    # Traditional CTC greedy decoding
                
                sequence = sb.decoders.ctc_greedy_decode(
                    p_ctc_feat, wav_lens, blank_id=self.hparams.blank_index
                )

                sequence_decoder_out = hyps  # [B, T_p+1]

                if self.hparams.eval_with_silence == False:
                    sil_inx = self.label_encoder.lab2ind["sil"]
                    # sequence / sequence_decoder_out 可能是: list[list[int]] 或 np.ndarray 或 torch.Tensor
                    def _to_list_of_lists(obj):
                        import numpy as np
                        import torch
                        if isinstance(obj, list):
                            # 确认是否已经是 list of lists
                            if len(obj) > 0 and isinstance(obj[0], (list, tuple)):
                                return [list(x) for x in obj]
                            # 可能是单条序列 -> 包成 batch
                            return [list(obj)]
                        if isinstance(obj, np.ndarray):
                            if obj.ndim == 1:
                                return [obj.tolist()]
                            return [row.tolist() for row in obj]
                        if torch.is_tensor(obj):
                            if obj.dim() == 1:
                                return [obj.cpu().tolist()]
                            return [row.cpu().tolist() for row in obj]
                        # 其它直接尝试包装
                        return [list(obj)]

                    def _filter_sil(batch_seqs, sil_id):
                        filtered = []
                        orig_lens = []
                        new_lens = []
                        for hyp in batch_seqs:
                            orig_lens.append(len(hyp))
                            # 逐元素过滤 silence
                            kept = [tok for tok in hyp if tok != sil_id]
                            # 如果全部被滤掉，至少保留一个（避免空序列导致后续崩溃）
                            if len(kept) == 0:
                                kept = [sil_id]  # 或者可以放 <blank>
                            filtered.append(kept)
                            new_lens.append(len(kept))
                        return filtered, orig_lens, new_lens

                    seq_list = _to_list_of_lists(sequence)
                    dec_list = _to_list_of_lists(sequence_decoder_out)

                    seq_filtered, seq_orig_lens, seq_new_lens = _filter_sil(seq_list, sil_inx)
                    dec_filtered, dec_orig_lens, dec_new_lens = _filter_sil(dec_list, sil_inx)

                    # 可选：如果后续指标期望 list[list[int]] 形式就直接用；若需 tensor 可再 pad
                    sequence = seq_filtered
                    sequence_decoder_out = dec_filtered

                    # 调试少量打印（避免刷屏）
                    if torch.rand(1).item() < 0.001:
                        print(f"[SilenceFilter] Removed sil tokens: avg Δlen = "
                              f"{(sum(seq_orig_lens)-sum(seq_new_lens))/max(1,len(seq_new_lens)):.2f}")
                    
                # 先准备参考序列（若需要去除 sil）
                filtered_ctc_ref = None
                filtered_ctc_ref_lens = None
                filtered_seq_ref = None
                filtered_seq_ref_lens = None
                if self.hparams.eval_with_silence == False:
                    import pdb; pdb.set_trace()
                    sil_inx = self.label_encoder.lab2ind.get("sil", None)
                    if sil_inx is not None:
                        # CTC 参考
                        if self.hparams.ctc_head_target == "perceived":
                            filtered_ctc_ref, filtered_ctc_ref_lens, _ = self.filter_token_batch(
                                targets, target_lens, token_id=sil_inx, pad_id=0
                            )
                        elif self.hparams.ctc_head_target == "canonical":
                            filtered_ctc_ref, filtered_ctc_ref_lens, _ = self.filter_token_batch(
                                canonicals, canonical_lens, token_id=sil_inx, pad_id=0
                            )
                        # seq2seq 参考（不含 eos 的那份，用于 PER_seq）
                        if self.hparams.decoder_target == "perceived":
                            filtered_seq_ref, filtered_seq_ref_lens, _ = self.filter_token_batch(
                                perceiveds, perceived_lens, token_id=sil_inx, pad_id=0
                            )
                        else:
                            filtered_seq_ref, filtered_seq_ref_lens, _ = self.filter_token_batch(
                                targets, target_lens, token_id=sil_inx, pad_id=0
                            )

                # CTC metrics（仍使用原 logits + 可能过滤后的参考）
                if self.hparams.ctc_head_target == "perceived":
                    self.ctc_metrics.append(ids, p_ctc_feat, targets, wav_lens, target_lens)
                elif self.hparams.ctc_head_target == "canonical":
                    self.ctc_metrics.append(ids, p_ctc_feat, canonicals, wav_lens, canonical_lens)

                # Using Joint Decoder's confidence on each frame to do threshold-based masking
                if self.hparams.allow_confidence_thresholding:
                    mask = (top_log_probs < self.hparams.confidence_threshold).float()
                    # replace mask phn inx with err index
                    err_inx = self.label_encoder.lab2ind["err"]
                    # make where mask == 1's inx in sequence_decoder_out as err
                    sequence_decoder_out_confience_thre = torch.tensor(sequence_decoder_out, device=mask.device)
                    mask = mask[:, :sequence_decoder_out_confience_thre.shape[1]]
                    sequence_decoder_out_confience_thre[mask.bool()] = err_inx
                    sequence_decoder_out = sequence_decoder_out_confience_thre.cpu().numpy()
                
                # ---- 插入：使用 mispro_class 对 sequence_decoder_out (hyps) 做修改 ----
                # pdb.set_trace()
                if self.hparams.apply_mispro_to_hyps == True:
                    # pdb.set_trace()
                    sequence_decoder_out = self._apply_mispro_to_hyps(sequence_decoder_out, p_mispro_cls_logits, mode="default")
                    # sequence_decoder_out = self._apply_mispro_to_hyps(hyps, p_mispro_cls_logits, mode="ignore_sub")
                    # pdb.set_trace()
                    # pdb.set_trace()
                    # sequence_decoder_out = self._apply_mispro_to_hyps(canonicals, p_mispro_cls_logits, mode="default")
                # pdb.set_trace()
                # pdb.set_trace()
                # ------------------------------------------------------------------

                if self.hparams.decoder_target == "perceived":
                    self.seq_metrics.append(ids, log_probabilities=p_dec_out, targets=perceiveds_eos, length=perceived_lens_eos)
                else:
                    self.seq_metrics.append(ids, log_probabilities=p_dec_out, targets=targets_eos, length=target_lens_eos)
                self.mispro_metrics.append(ids, h_mispro_bin, mispro_label_bin, mispro_label_lens)
                self.mispro_metrics_cls.append(ids, p_mispro_cls_logits, mispro_label, mispro_label_lens)
                
                # self.ctc_metrics_fuse.append(ids, sequence_decoder_out, targets, wav_lens, target_lens)
                # CTC-only results
                # import pdb; pdb.set_trace()
                if self.hparams.ctc_head_target == "perceived":
                    if self.hparams.eval_with_silence == False and filtered_ctc_ref is not None:
                        self.per_metrics.append(
                            ids=ids,
                            predict=sequence,
                            target=filtered_ctc_ref,
                            predict_len=None,
                            target_len=filtered_ctc_ref_lens,
                            ind2lab=self.label_encoder.decode_ndim,
                        )
                    else:
                        self.per_metrics.append(
                            ids=ids,
                            predict=sequence,
                            target=targets,
                            predict_len=None,
                            target_len=target_lens,
                            ind2lab=self.label_encoder.decode_ndim,
                        )
                elif self.hparams.ctc_head_target == "canonical":
                    if self.hparams.eval_with_silence == False and filtered_ctc_ref is not None:
                        self.per_metrics.append(
                            ids=ids,
                            predict=sequence,
                            target=filtered_ctc_ref,
                            predict_len=None,
                            target_len=filtered_ctc_ref_lens,
                            ind2lab=self.label_encoder.decode_ndim,
                        )
                    else:
                        self.per_metrics.append(
                            ids=ids,
                            predict=sequence,
                            target=canonicals,
                            predict_len=None,
                            target_len=canonical_lens,
                            ind2lab=self.label_encoder.decode_ndim,
                        )
                    
                # seq2seq results
                # import pdb; pdb.set_trace()
                if self.hparams.decoder_target == "perceived":
                    if self.hparams.eval_with_silence == False and filtered_seq_ref is not None:
                        self.per_metrics_seq.append(
                            ids=ids,
                            predict=sequence_decoder_out,
                            target=filtered_seq_ref,
                            predict_len=None,
                            target_len=filtered_seq_ref_lens,
                            ind2lab=self.label_encoder.decode_ndim,
                        )
                    else:
                        self.per_metrics_seq.append(
                            ids=ids,
                            predict=sequence_decoder_out,
                            target=perceiveds,
                            predict_len=None,
                            target_len=perceived_lens_eos,
                            ind2lab=self.label_encoder.decode_ndim,
                        )
                else:
                    
                    if self.hparams.eval_with_silence == False and filtered_seq_ref is not None:
                        self.per_metrics_seq.append(
                            ids=ids,
                            predict=sequence_decoder_out,
                            target=filtered_seq_ref,
                            predict_len=None,
                            target_len=filtered_seq_ref_lens,
                            ind2lab=self.label_encoder.decode_ndim,
                        )
                    else:
                        # remove id 70 from sequence_decoder_out
                        sequence_decoder_out = [
                            [tok for tok in seq if tok != 70]
                            for seq in sequence_decoder_out
                        ]
                        self.per_metrics_seq.append(
                            ids=ids,
                            predict=sequence_decoder_out,
                            target=targets,
                            predict_len=None,
                            target_len=target_lens,
                            ind2lab=self.label_encoder.decode_ndim,
                        )
                # import pdb; pdb.set_trace()
                # MPD metrics
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
                
                self.mpd_metrics_seq.append(
                    ids=ids,
                    predict=sequence_decoder_out,
                    canonical=canonicals,
                    perceived=perceiveds,
                    predict_len=None,
                    canonical_len=canonical_lens,
                    perceived_len=perceived_lens,
                    ind2lab=self.label_encoder.decode_ndim,
                )
            
        # Log to wandb if available (VALID stage only)
        current_epoch = self.hparams.epoch_counter.current          
        valid_search_interval = self.hparams.valid_search_interval
        if current_epoch % valid_search_interval == 0 or (
            stage == sb.Stage.TEST
        ):
            try:
                import wandb
                wandb.log({
                    "loss": loss.item(),
                }, step=self.hparams.epoch_counter.current)
                # if loss_ga is not None:
                #     wandb.log({"loss_ga": loss_ga.item()}, step=self.hparams.epoch_counter.current)
                if loss_dec_out is not None:
                    wandb.log({"loss_dec_out": loss_dec_out.item()}, step=self.hparams.epoch_counter.current)
                if loss_ctc is not None:
                    wandb.log({"loss_ctc_head": loss_ctc.item()}, step=self.hparams.epoch_counter.current)
                if loss_mispro is not None:
                    wandb.log({"loss_mispro": loss_mispro.item()}, step=self.hparams.epoch_counter.current)
                if loss_mispro_cls is not None:
                    wandb.log({"loss_mispro_cls": loss_mispro_cls.item()}, step=self.hparams.epoch_counter.current)
                if loss_ga is not None:
                    wandb.log({"loss_ga": loss_ga.item()}, step=self.hparams.epoch_counter.current)
                if loss_phn_pco is not None:
                    wandb.log({"loss_conpco": loss_phn_pco.item()}, step=self.hparams.epoch_counter.current)
                if loss_center_clap is not None:
                    wandb.log({"loss_center_clap": loss_center_clap.item()}, step=self.hparams.epoch_counter.current)
            except Exception:
                pass
        return loss
    
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

        # Initialize optimizers after parameters are configured
        self.init_optimizers()

        # if self.checkpointer is not None:
        #     # TODO: support recover best on PER or mpd_f1 or averaged model of best PER and mpd_f1
        #     self.checkpointer.recover_if_possible(
        #         max_key="mpd_f1_seq",
        #         # max_key="mpd_f1",
        #         # importance_keys=[
        #         #     lambda ckpt: (-ckpt.meta.get("PER_seq", 1e6), ckpt.meta.get("mpd_f1_seq", 0), -ckpt.meta.get("PER", 1e6), ckpt.meta.get("mpd_f1", 0)),
        #         # ]
        #     )
        
        # For CTC Head init, usually means training from scratch.
        
        pretrainer = getattr(self.hparams, 'pretrainer', None)
        if pretrainer is not None and getattr(self.hparams, 'resume_from_folder', False):
            paths = pretrainer.collect_files(default_source=self.hparams.resume_from_folder)
            pretrainer.load_collected()
            # pdb.set_trace()
            # self.modules.perceived_ssl.model.state_dict()['encoder.layers.23.final_layer_norm.bias']== pretrainer.loadables['perceived_ssl'].state_dict()['model.encoder.layers.23.final_layer_norm.bias']
            # self.modules.enc.state_dict()["1.bias"] = pretrainer.loadables['model'][0].state_dict()["1.bias"]
            # self.modules.ConformerEncoder.state_dict()['layers.0.convolution_module.conv.weight'] == pretrainer.loadables['model'].state_dict()['1.layers.0.convolution_module.conv.weight']
            
        # Load pretrained components if specified
        # if getattr(self.hparams, 'load_pretrained_components', False):
        #     pretrained_path = getattr(self.hparams, 'pretrained_model_path', '')
        #     components = getattr(self.hparams, 'components_to_load', ['ssl', 'enc', "ctc_head"])
        #     freeze_loaded = getattr(self.hparams, 'freeze_loaded_components', True)
            
        #     if pretrained_path and os.path.exists(pretrained_path):
        #         try:
        #             self.load_pretrained_components(
        #                 checkpoint_path=pretrained_path,
        #                 components_to_load=components,
        #                 freeze_loaded=freeze_loaded
        #             )
        #         except Exception as e:
        #             print(f"❌ Failed to load pretrained components: {e}")
        #             print("   Continuing with random initialization...")
        #     else:
        #         print(f"⚠️  Pretrained model path not found: {pretrained_path}")
        #         print("   Continuing with random initialization...")
        # Load latest checkpoint to resume training if interrupted
        ## NOTE: make sure to use the "best" model to continual training
        ## so we set the `min_key` argument
        
        # TODO For resume training or VALID or TESTING use this head
        
        
        if self.checkpointer is not None:
            # TODO: support recover best on PER or mpd_f1 or averaged model of best PER and mpd_f1
            self.checkpointer.recover_if_possible(
                max_key="mpd_f1_seq",
                # max_key="mpd_f1",
                # importance_keys=[
                #     lambda ckpt: (-ckpt.meta.get("PER_seq", 1e6), ckpt.meta.get("mpd_f1_seq", 0), -ckpt.meta.get("PER", 1e6), ckpt.meta.get("mpd_f1", 0)),
                # ]
            )
            

    def on_stage_end(self, stage, stage_loss, epoch):
        current_stage = self.hparams.epoch_counter.current
        """Gets called at the end of a epoch."""
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss

        if stage == sb.Stage.VALID:
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval
            if (
                current_epoch % valid_search_interval == 0
            ):
                per = self.per_metrics.summarize("error_rate")
                mpd_f1 = self.mpd_metrics.summarize("mpd_f1")
                per_seq = self.per_metrics_seq.summarize("error_rate")
                mpd_f1_seq = self.mpd_metrics_seq.summarize("mpd_f1")

                mispro_loss = self.mispro_metrics.summarize("average")
                mispro_loss_cls = self.mispro_metrics_cls.summarize("average")
                
                # create a stat for conpco loss
                # from speechbrain.utils.metric_stats import MultiMetricStats
                
                # self.conpco_stats = MetricStats(metric_name="conpco_loss")
                # conpco_loss = self.conpco_stats.summarize("average")
                
                current_epoch = self.hparams.epoch_counter.current
                valid_search_interval = self.hparams.valid_search_interval
                # Log stats
                valid_stats = {
                    "loss": stage_loss,
                    "ctc_loss": self.ctc_metrics.summarize("average"),
                    "seq_loss": self.seq_metrics.summarize("average"),
                    "mispro_loss": self.mispro_metrics.summarize("average"),
                    "mispro_loss_cls": self.mispro_metrics_cls.summarize("average"),
                    "PER": per,
                    "mpd_f1": mpd_f1,
                    "PER_seq": per_seq,
                    "mpd_f1_seq": mpd_f1_seq,
                }
                
                # Check CTC convergence at the end of validation epoch
                current_ctc_loss = self.ctc_metrics.summarize("average")
                self.check_ctc_convergence(current_ctc_loss)
                
                # Check metric convergence at the end of validation epoch
                self.check_metric_convergence(per, mpd_f1)
            
                self.hparams.train_logger.log_stats(
                    stats_meta={
                        "epoch": epoch,
                        "lr_adam": self.adam_optimizer.param_groups[0]["lr"],
                        # "lr_pretrained": self.pretrained_opt_class.param_groups[0]["lr"],
                    },
                    train_stats={"loss": self.train_loss,
                                 "ctc_loss": self.ctc_metrics.summarize("average"),
                                 "mispro_loss": self.mispro_metrics.summarize("average"),
                                 "mispro_loss_cls": self.mispro_metrics_cls.summarize("average"),
                                 "seq_loss": self.seq_metrics.summarize("average"), 
                                 },
                    valid_stats=valid_stats,
                )                # Save best 3 models for each metric using simplified approach
                improved = False
                ckpt_name = f"{epoch:03d}_PER_{per:.4f}_PER_seq_{per_seq:.4f}_F1_{mpd_f1:.4f}_F1_seq_{mpd_f1_seq:.4f}.ckpt"
                self.checkpointer.save_and_keep_only(
                    meta={
                        "epoch": epoch,"PER": per,"mpd_f1": mpd_f1,"PER_seq": per_seq,"mpd_f1_seq": mpd_f1_seq},
                    name=ckpt_name,
                    num_to_keep=4,
                    importance_keys=[
                        lambda ckpt: (
                            -ckpt.meta.get("PER_seq", 1e6),
                            ckpt.meta.get("mpd_f1_seq", 0),
                            -ckpt.meta.get("PER", 1e6),
                            ckpt.meta.get("mpd_f1", 0)
                            ),
                    ]
                )
                
                # Early stopping logic: only track best valid loss, do not save checkpoint for valid loss
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

                # Logging
                wandb.log({
                    "epoch": epoch,
                    "train_loss": self.train_loss,
                    "valid_loss": stage_loss,
                    "ctc_loss": self.ctc_metrics.summarize("average"),
                    "PER": per,
                    "mpd_f1": mpd_f1,
                    "PER_seq": per_seq,
                    "mpd_f1_seq": mpd_f1_seq,
                    "encoder_frozen": self.encoder_frozen,
                    "ssl_frozen": self.ssl_frozen,
                    "best_ctc_loss": self.best_ctc_loss,
                    "ctc_no_improve_epochs": self.ctc_no_improve_epochs,
                    "best_valid_per": self.best_valid_per,
                    "best_valid_f1": self.best_valid_f1,
                    "metric_no_improve_epochs": self.metric_no_improve_epochs,
                    "freezing_metric": self.freezing_metric,
                }, step=epoch)
                # Early stop if patience exceeded
                if self.no_improve_epochs >= self.patience:
                    print(f"Early stopping at epoch {epoch} (no improvement for {self.patience} epochs)")
                    raise StopIteration

        if stage == sb.Stage.TEST:
            per = self.per_metrics.summarize("error_rate")
            mpd_f1 = self.mpd_metrics.summarize("mpd_f1")
            
            per_seq = self.per_metrics_seq.summarize("error_rate")
            mpd_f1_seq = self.mpd_metrics_seq.summarize("mpd_f1")
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats={"loss": stage_loss, "PER": per, "mpd_f1": mpd_f1, 
                            "PER_seq": per_seq, "mpd_f1_seq": mpd_f1_seq},
            )
            # 
            with open(self.hparams.per_file, "w") as w:
                w.write("CTC loss stats:\n")
                self.ctc_metrics.write_stats(w)
                w.write("\nPER stats:\n")
                self.per_metrics.write_stats(w)
                print(
                    "CTC and PER stats written to file",
                    self.hparams.per_file,
                )
            with open(self.hparams.mpd_file, "w") as m:
                m.write("MPD results and stats:\n")
                self.mpd_metrics.write_stats(m)
                print(
                    "MPD results and stats written to file",
                    self.hparams.mpd_file,
                )
            # pdb.set_trace()
            # if not files for joint decoding, create files

            if not hasattr(self.hparams, 'per_seq_file'):
                self.hparams.per_seq_file = self.hparams.per_file.replace(".txt", "_seq.txt")
            with open(self.hparams.per_seq_file, "w") as w:
                w.write("Joint CTC-Attention PER stats:\n")
                self.seq_metrics.write_stats(w)
                self.per_metrics_seq.write_stats(w)
                print(
                    "Joint CTC-Attention PER stats written to file",
                    self.hparams.per_seq_file,
                )
            if not hasattr(self.hparams, 'mpd_seq_file'):
                self.hparams.mpd_seq_file = self.hparams.mpd_file.replace(".txt", "_seq.txt")
            with open(self.hparams.mpd_seq_file, "w") as m:
                m.write("Joint CTC-Attention MPD results and stats:\n")
                self.mpd_metrics_seq.write_stats(m)
                print(
                    "Joint CTC-Attention MPD results and stats written to file",
                    self.hparams.mpd_seq_file,
                )

    def init_optimizers(self):

        self.adam_optimizer = self.hparams.adam_opt_class(
            # self.hparams.model.parameters(),
            self.hparams.trainable_model.parameters(),
        )
        
        # frz perceived SSL, TransASR's custom_src_module and encoder, as well as ctc head
        # if self.modules.perceived_ssl is not None:
        #     # if self.hparams.model.perceived_ssl.freeze:
        #     print("Freezing perceived SSL model parameters.")
        #     for p in self.modules.perceived_ssl.parameters():
        #         p.requires_grad = False
        # if self.modules.TransASR.custom_src_module is not None:
        #     print("Freezing TransASR custom_src_module parameters.")
        #     for p in self.modules.TransASR.custom_src_module.parameters():
        #         p.requires_grad = False
        # if self.modules.TransASR.encoder is not None:
        #     print("Freezing TransASR encoder parameters.")
        #     for p in self.modules.TransASR.encoder.parameters():
        #         p.requires_grad = False
        # if self.modules.ctc_lin is not None:
        #     print("Freezing TransASR ctc_head parameters.")
        #     for p in self.modules.ctc_lin.parameters():
        #         p.requires_grad = False
        # # 
        if self.checkpointer is not None:
            # if self.hparams.perceived_ssl is not None and not self.hparams.perceived_ssl.freeze:
            self.checkpointer.add_recoverable("adam_opt", self.adam_optimizer)
            # self.checkpointer.add_recoverable("pretrained_opt", self.pretrained_opt_class)
            # Add recoverable for freezing states
            self.checkpointer.add_recoverable("encoder_frozen", self)
            self.checkpointer.add_recoverable("ssl_frozen", self)
            self.checkpointer.add_recoverable("best_ctc_loss", self)
            self.checkpointer.add_recoverable("ctc_no_improve_epochs", self)
            self.checkpointer.add_recoverable("ctc_loss_history", self)
            self.checkpointer.add_recoverable("best_valid_per", self)
            self.checkpointer.add_recoverable("best_valid_f1", self)
            self.checkpointer.add_recoverable("metric_no_improve_epochs", self)
            self.checkpointer.add_recoverable("per_history", self)
            self.checkpointer.add_recoverable("f1_history", self)
            
    def on_evaluate_start(self, max_key=None, min_key=None):
        return super().on_evaluate_start(max_key, min_key)

    def on_fit_batch_end(self, batch, outputs, loss, should_step):
        """At the end of the optimizer step, apply noam annealing."""
        if should_step:
            self.hparams.noam_annealing(self.adam_optimizer)

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
        def check_gradients(loss):
            """Check if gradients are finite"""
            if not torch.isfinite(loss):
                print("Warning: loss is not finite, skipping step")
                return False
            return True
        
        if self.hparams.auto_mix_prec:
            # Use automatic mixed precision training
            self.adam_optimizer.zero_grad()
            # self.pretrained_opt_class.zero_grad()

            with torch.amp.autocast("cuda"):
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            # normalize the loss by gradient_accumulation and scale for mixed precision
            self.scaler.scale(loss / self.hparams.gradient_accumulation).backward()
            self.scaler.unscale_(self.adam_optimizer)
            # import pdb; pdb.set_trace()
            if check_gradients(loss):
                # Main optimizer always steps (includes decoder)
                self.scaler.step(self.adam_optimizer)
                
            self.scaler.update()

        else:
            
            outputs = self.compute_forward(batch, sb.Stage.TRAIN)
            
            loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            # normalize the loss by gradient_accumulation step
            (loss / self.hparams.gradient_accumulation).backward()

            if self.step % self.hparams.gradient_accumulation == 0:
                # gradient clipping & early stop if loss is not fini
                
                if check_gradients(loss):
                    # Only step optimizers for non-frozen parts
                    # if not self.ssl_frozen:
                    #     self.pretrained_opt_class.step()
                    # else:
                    #     print("SSL model frozen, skipping SSL optimizer step")
                    
                    # Main optimizer always steps (includes decoder)
                    self.adam_optimizer.step()

                # Always zero gradients
                # self.pretrained_opt_class.zero_grad()
                self.adam_optimizer.zero_grad()    

        return loss.detach().cpu()

    def _apply_mispro_to_hyps(self, sequence_decoder_out, p_mispro_cls_logits, mode):
        """Apply mispronunciation class predictions to sequence_decoder_out.

        Rules:
          - Substitution (1): replace token with err
          - Deletion (2): remove token
          - Insertion (3): insert an err token
        Accepts sequence_decoder_out as list/np.array/torch.tensor and returns list[list[int]].
        """
        # If no predictions provided, return input unchanged
        if p_mispro_cls_logits is None:
            return sequence_decoder_out

        err_inx = self.label_encoder.lab2ind.get("err", None)
        if err_inx is None:
            return sequence_decoder_out

        # helper: convert various types to list[list[int]]
        def _to_list_of_lists_local(obj):
            if isinstance(obj, list):
                if len(obj) > 0 and isinstance(obj[0], (list, tuple)):
                    return [list(x) for x in obj]
                return [list(obj)]
            if isinstance(obj, np.ndarray):
                if obj.ndim == 1:
                    return [obj.tolist()]
                return [row.tolist() for row in obj]
            if torch.is_tensor(obj):
                if obj.dim() == 1:
                    return [obj.cpu().tolist()]
                return [row.cpu().tolist() for row in obj]
            # fallback
            return [list(obj)]

        dec_list_local = _to_list_of_lists_local(sequence_decoder_out)
        mispro_pred = p_mispro_cls_logits.argmax(-1).cpu().tolist()  # [B, T_c]

        new_dec_out = []
        for b, seq_row in enumerate(dec_list_local):
            preds_b = mispro_pred[b] if b < len(mispro_pred) else []
            out_row = []
            min_len = min(len(seq_row), len(preds_b))
            # iterate aligned portion
            for i in range(min_len):
                cls = int(preds_b[i])
                if cls == 0:  # Correct
                    out_row.append(seq_row[i])
                elif cls == 1:  # Substitution -> replace with err
                    if mode == "default":
                        out_row.append(err_inx)
                    elif mode == "ignore_sub":
                        # Use original sub as diagnosis result
                        out_row.append(seq_row[i])
                elif cls == 2:  # Deletion -> skip this token
                    continue
                elif cls == 3:  # Insertion -> insert an err here
                    out_row.append(err_inx)
                else:
                    out_row.append(seq_row[i])
            # append any remaining tokens in seq_row beyond preds
            if len(seq_row) > min_len:
                out_row.extend(seq_row[min_len:])
            # if preds are longer than seq_row, handle insert-only preds
            if len(preds_b) > min_len:
                for j in range(min_len, len(preds_b)):
                    cls = int(preds_b[j])
                    if cls == 3:
                        out_row.append(err_inx)
            new_dec_out.append(out_row)

        return new_dec_out

    def inference_batch(self, batch):
        """Pure inference on a single batch - ONLY audio input, NO ground truth.
        
        This method is used for INFERENCE stage (hparams.inference_mode=True).
        Batch contains ONLY:
            - batch.id: utterance identifiers (strings)
            - batch.sig: (wavs [B, T], wav_lens [B])
        
        NO ground truth available - uses pure autoregressive decoding.
        
        Args:
            batch: Input batch containing only audio:
                - batch.id: list of utterance IDs
                - batch.sig: tuple of (wavs, wav_lens)
            
        Returns:
            dict: Contains predictions {
                "hyps": generated phoneme sequences [B, T_decoded],
                "top_log_probs": confidence scores [B, T_decoded],
                "top_lengths": actual sequence lengths [B],
                "ids": utterance identifiers
            }
        """
        batch = batch.to(self.device)
        
        with torch.no_grad():
            # Forward pass: encoder-only + AR decoding (no teacher forcing, no GT needed)
            self.hparams.inference_mode = True
            predictions = self.compute_forward(batch, stage=sb.Stage.TEST)
            self.hparams.inference_mode = False
        
        # Extract predictions (only outputs available in INFERENCE mode)
        return {
            "ids": batch.id,
            "hyps": predictions.get("hyps"),  # [B, T_decoded]
            "top_log_probs": predictions.get("top_log_probs"),  # [B, T_decoded] confidence scores
            "top_lengths": predictions.get("top_lengths"),  # [B] actual lengths
        }
    
    def inference(self, test_set, test_loader_kwargs=None, max_key=None, min_key=None, output_file=None):
        """Pure inference on entire dataset - ONLY audio input, NO ground truth.
        
        This method processes a test set containing ONLY audio (batch.id and batch.sig).
        Uses autoregressive decoding WITHOUT any ground truth phonemes.
        
        Process flow:
        1. Load best checkpoint if specified (max_key/min_key)
        2. For each batch: audio only → encoder → AR decoding → phoneme predictions
        3. Collect and optionally save results
        
        Args:
            test_set: Dataset for inference (batch must contain only batch.id and batch.sig)
            test_loader_kwargs (dict, optional): Kwargs for DataLoader creation
            max_key (str, optional): Key to maximize for best model selection
            min_key (str, optional): Key to minimize for best model selection
            output_file (str, optional): File path to save inference results
            
        Returns:
            dict: All inference results containing:
                - "ids": list of utterance IDs
                - "hyps": list of generated phoneme sequences
                - "top_log_probs": confidence scores for each frame
                - "top_lengths": actual lengths of generated sequences
        """
        # Load best checkpoint if specified
        if max_key is not None or min_key is not None:
            self.checkpointer.recover_if_possible(max_key=max_key, min_key=min_key)
        
        # Setup inference dataloader
        if test_loader_kwargs is None:
            test_loader_kwargs = {}
        
        test_dataloader = self.make_dataloader(test_set, stage=sb.Stage.TEST, **test_loader_kwargs)
        
        if not hasattr(self.hparams, 'inference_mode'):
            self.hparams.inference_mode = True
        else:
            self.hparams.inference_mode = True
        
        all_hyps = []
        all_ids = []
        all_lengths = []
        all_log_probs = []
        all_ctc_hyps = []  # Acoustic model (CTC) hypotheses
        
        self.modules.eval()
        
        from tqdm import tqdm
        
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Inferencing", dynamic_ncols=True):
                # Move batch to device
                batch = batch.to(self.device)
                
                # Get predictions
                predictions = self.compute_forward(batch, stage=sb.Stage.TEST)
                
                # Collect seq2seq results
                if predictions.get("hyps") is not None:
                    # Handle both list and tensor formats
                    hyps = predictions["hyps"]
                    if torch.is_tensor(hyps):
                        hyps = hyps.cpu().numpy()
                    if isinstance(hyps, np.ndarray):
                        hyps = hyps.tolist()
                    all_hyps.extend(hyps if isinstance(hyps[0], list) else [hyps])
                
                if hasattr(batch, 'id'):
                    ids = batch.id if isinstance(batch.id, list) else [batch.id]
                    all_ids.extend(ids)
                
                if predictions.get("top_lengths") is not None:
                    lengths = predictions["top_lengths"]
                    if torch.is_tensor(lengths):
                        lengths = lengths.cpu().numpy()
                    all_lengths.extend(lengths.tolist() if hasattr(lengths, 'tolist') else list(lengths))
                
                if predictions.get("top_log_probs") is not None:
                    log_probs = predictions["top_log_probs"]
                    if torch.is_tensor(log_probs):
                        log_probs = log_probs.cpu().numpy()
                    all_log_probs.extend(log_probs.tolist() if hasattr(log_probs, 'tolist') else list(log_probs))
                
                # Collect acoustic model (CTC) results
                if predictions.get("p_ctc_feat") is not None:
                    p_ctc_feat = predictions["p_ctc_feat"]
                    wav_lens = batch.sig[1]
                    # CTC greedy decoding
                    ctc_sequence = sb.decoders.ctc_greedy_decode(
                        p_ctc_feat, wav_lens, blank_id=self.hparams.blank_index
                    )
                    # Handle different formats
                    if torch.is_tensor(ctc_sequence):
                        ctc_sequence = ctc_sequence.cpu().numpy()
                    if isinstance(ctc_sequence, np.ndarray):
                        ctc_sequence = ctc_sequence.tolist()
                    all_ctc_hyps.extend(ctc_sequence if isinstance(ctc_sequence[0], list) else [ctc_sequence])
        
        # Compile results - decode phoneme indices to labels
        decoded_hyps = []
        for hyp in all_hyps:
            # Convert phoneme indices to labels using label encoder
            phn_list = self.label_encoder.decode_ndim(hyp)
            decoded_hyps.append(" ".join(phn_list))
        
        # Decode CTC results to labels
        decoded_ctc_hyps = []
        for ctc_hyp in all_ctc_hyps:
            phn_list = self.label_encoder.decode_ndim(ctc_hyp)
            decoded_ctc_hyps.append(" ".join(phn_list))
        
        # Extract stems from IDs (filename without path and extension)
        import os
        stem_ids = [os.path.splitext(os.path.basename(uid))[0] for uid in all_ids]
        
        results = {
            "ids": stem_ids,
            "labels": decoded_hyps,  # Seq2seq decoded phoneme sequences
            "labels_ctc": decoded_ctc_hyps,  # CTC (acoustic model) decoded phoneme sequences
            "hyps_indices": all_hyps,  # Seq2seq raw indices (for reference)
            "hyps_indices_ctc": all_ctc_hyps,  # CTC raw indices (for reference)
            "lengths": all_lengths,
            "log_probs": all_log_probs,
        }
        
        # Save results if output file is provided - Split into two separate CSV files
        if output_file is not None:
            import csv
            from pathlib import Path
            
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Generate separate file names for Seq2Seq and CTC
            # If output_file is "results.csv", create "results_seq2seq.csv" and "results_ctc.csv"
            output_stem = output_path.stem  # Get filename without extension
            output_dir = output_path.parent
            
            seq2seq_file = output_dir / f"{output_stem}_seq2seq.csv"
            ctc_file = output_dir / f"{output_stem}_ctc.csv"
            
            # Save Seq2Seq predictions to first file
            with open(seq2seq_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["id", "Labels"])  # Header
                for stem_id, label_seq2seq in zip(stem_ids, decoded_hyps):
                    writer.writerow([stem_id, label_seq2seq])
            
            print(f"✓ Seq2Seq results saved to: {seq2seq_file}")
            
            # Save CTC predictions to second file
            with open(ctc_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["id", "Labels"])  # Header
                for stem_id, label_ctc in zip(stem_ids, decoded_ctc_hyps):
                    writer.writerow([stem_id, label_ctc])
            
            print(f"✓ CTC results saved to: {ctc_file}")
        
        self.hparams.inference_mode = False
        return results