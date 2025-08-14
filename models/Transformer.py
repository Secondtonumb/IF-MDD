import os

import torch
import torch.nn

import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from mpd_eval_v4 import MpdStats


import wandb

from speechbrain.nnet.loss.guidedattn_loss import GuidedAttentionLoss
from speechbrain.lobes.models.dual_path import PyTorchPositionalEncoding
from speechbrain.nnet.attention import RelPosEncXL, RelPosMHAXL
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

from utils.layers.utils import make_pad_mask

# import torch

# # 当前 GPU 显存使用
# print("Allocated:", torch.cuda.memory_allocated() / 1024**2, "MB")
# print("Cached:", torch.cuda.memory_reserved() / 1024**2, "MB")

class TransformerMDD(sb.Brain):
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
        pdb.set_trace()
        print(f"\n🔄 Loading pretrained components from: {checkpoint_path}")
        print(f"   Components to load: {components_to_load}")
        
        # Load the checkpoint
        if os.path.isdir(checkpoint_path):
            # Find the checkpoint file in the directory
            ckpt_files = [f for f in os.listdir(checkpoint_path) if f.endswith('.ckpt')]
            if not ckpt_files:
                raise ValueError(f"No .ckpt files found in {checkpoint_path}")
            # Use the most recent checkpoint
            ckpt_files.sort()
            checkpoint_file = os.path.join(checkpoint_path, ckpt_files[-1])
            print(f"   Using checkpoint: {ckpt_files[-1]}")
            # pdb.set_trace()

        else:
            checkpoint_file = checkpoint_path
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_file, map_location=self.device)
        
        # Extract model state dict
        if 'model' in checkpoint:
            pretrained_state = checkpoint['model']
        else:
            pretrained_state = checkpoint
        
        # Get current model state
        current_state = self.modules.state_dict()
        
        # Component mapping
        component_mapping = {
            'ssl': ['perceived_ssl'],
            'encoder': ['TransASR.encoder', 'TransASR.custom_src_module'],
            'enc_projection': ['enc'],
            'ctc_head': ['ctc_lin'],
            'decoder': ['TransASR.decoder', 'd_out']
        }
        pdb.set_trace()
        # Load specified components
        loaded_components = []
        for component in components_to_load:
            if component not in component_mapping:
                print(f"   ⚠️  Warning: Unknown component '{component}', skipping...")
                continue
                
            module_prefixes = component_mapping[component]
            for prefix in module_prefixes:
                # Find matching keys
                matching_keys = [k for k in pretrained_state.keys() if k.startswith(prefix)]
                if not matching_keys:
                    print(f"   ⚠️  Warning: No parameters found for {prefix} in checkpoint")
                    continue
                
                # Load matching parameters
                loaded_count = 0
                for key in matching_keys:
                    if key in current_state:
                        try:
                            current_state[key] = pretrained_state[key]
                            loaded_count += 1
                        except Exception as e:
                            print(f"   ❌ Error loading {key}: {e}")
                    else:
                        print(f"   ⚠️  Key {key} not found in current model")
                
                if loaded_count > 0:
                    loaded_components.append(prefix)
                    print(f"   ✅ Loaded {loaded_count} parameters for {prefix}")
        pdb.set_trace()
        # Load the updated state dict
        self.modules.load_state_dict(current_state, strict=False)
        
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
                    
                elif component == 'enc_projection':
                    if hasattr(self.modules, 'enc'):
                        for param in self.modules.enc.parameters():
                            param.requires_grad = False
                        print("   🔒 Encoder projection frozen")
                        
                elif component == 'ctc_head':
                    for param in self.modules.ctc_lin.parameters():
                        param.requires_grad = False
                    print("   🔒 CTC head frozen")
        pdb.set_trace()
        print(f"   ✅ Successfully loaded components: {loaded_components}")
        return loaded_components
    
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
        if hasattr(self.hparams, "augmentation"):
            self.modules.perceived_ssl.model.config.apply_spec_augment = True

        if stage != sb.Stage.TRAIN:
            self.per_metrics = self.hparams.per_stats()
            self.per_metrics_seq = self.hparams.per_stats()
            self.mpd_metrics = MpdStats()
            self.mpd_metrics_seq = MpdStats()
   
    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
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
        
        feats = self.modules.perceived_ssl(wavs)  # [B, T_s, ENC_DIM] or [N_layers, B, T_s, ENC_DIM]
        if len(feats.shape) == 4:
            feats = feats[self.hparams.preceived_ssl_emb_layer] # [B, T_s, ENC_DIM]

        current_epoch = self.hparams.epoch_counter.current
        hyps = None
        attn_map = None

        if sb.Stage.TRAIN == stage:
            enc_out, hidden, dec_out = self.modules.TransASR(
                src=feats,
                tgt=targets_bos,
                wav_len=wav_lens,
                pad_idx=0, 
            )
            # CTC head
            h_ctc_feat = self.modules.ctc_lin(enc_out)  # [B, T_s, C]
            p_ctc_logits = self.hparams.log_softmax(h_ctc_feat)  # Log probabilities

            # seq2seq head
            h_seq_feat = self.modules.d_out(dec_out)  # [B, T_p+1, C]
            p_seq_logits = self.hparams.log_softmax(h_seq_feat)  # Log probabilities

        else:
            with torch.no_grad():
                enc_out, hidden, dec_out = self.modules.TransASR(
                    src=feats,
                    tgt=targets_bos,
                    wav_len=wav_lens,
                    pad_idx=0,  # Assuming 0 is the padding index
                )
                h_ctc_feat = self.modules.ctc_lin(enc_out)  # [B, T_s, C]
                p_ctc_logits = self.hparams.log_softmax(h_ctc_feat)  # Log probabilities

                # seq2seq head
                h_seq_feat = self.modules.d_out(dec_out)  # [B, T_p+1, C]
                p_seq_logits = self.hparams.log_softmax(h_seq_feat)  # Log probabilities
                
                hyps = None
                attn_map = None
        
            valid_search_interval = self.hparams.valid_search_interval
            if current_epoch % valid_search_interval == 0:
                hyps, top_lengths, top_scores, top_log_probs = self.hparams.valid_search(enc_out.detach(), wav_lens)
                attn_map = None
            if stage == sb.Stage.TEST:
                hyps, top_lengths, top_scores, top_log_probs = self.hparams.test_search(enc_out.detach(), wav_lens)
                attn_map = None
        return {
            "p_ctc_feat": p_ctc_logits,  # [B, T_s, C]
            "p_dec_out": p_seq_logits,  # [B, T_p+1, C]
            "feats": feats,  # [B, T_s, D]
            "attn_map": attn_map,  # [B, T_p+1, T_s] or similar
            "hyps": hyps,  # [B, T_p+1] or None if not applicable
        }
        
    def compute_objectives(self, predictions, batch, stage):
        "Computes the loss for the model."
        current_epoch = self.hparams.epoch_counter.current
        valid_search_interval = self.hparams.valid_search_interval
        
        p_ctc_feat = predictions["p_ctc_feat"]
        p_dec_out = predictions["p_dec_out"]
        feats = predictions["feats"]
        attn_map = predictions["attn_map"]
        hyps = predictions.get("hyps", [])  # [B, T_p+1] or None if not applicable

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
    
        # Caculate the loss for CTC and seq2seq outputs

        loss_ctc = self.hparams.ctc_cost(p_ctc_feat, targets, wav_lens, target_lens)
        loss_dec_out = self.hparams.seq_cost(p_dec_out, targets_eos, target_lens_eos)
        
        # if stage == sb.Stage.TRAIN:
        #     # # TODO: Better to Train CELoss without BOS / EOS
        #     # pdb.set_trace()
        #     # loss_dec_out = torch.tensor(0.0, device=self.device)

        #     loss_dec_out = sb.nnet.losses.kldiv_loss(
        #         p_dec_out,
        #         targets_eos,
        #         length=target_lens_eos,
        #         label_smoothing=0.1,
        #         reduction="batchmean",
        #     )

        # else: 
        #     # During inference, don't compute attention loss
        #     # loss_dec_out = torch.tensor(0.0, device=self.device)
        #     loss_dec_out = sb.nnet.losses.kldiv_loss(
        #         p_dec_out,
        #         targets_eos,
        #         length=target_lens_eos,
        #         label_smoothing=0.1,
        #         reduction="batchmean",
        #     )
        # ---- Guided Attention Loss ----

        loss = (
            self.hparams.ctc_weight * loss_ctc
            + (1 - self.hparams.ctc_weight) * loss_dec_out
        )
        
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

                self.ctc_metrics.append(ids, p_ctc_feat, targets, wav_lens, target_lens)
                self.seq_metrics.append(ids, log_probabilities=p_dec_out, targets=targets_eos, length=target_lens_eos)
                
                # self.ctc_metrics_fuse.append(ids, sequence_decoder_out, targets, wav_lens, target_lens)
                
                # CTC-only results
                self.per_metrics.append(
                    ids=ids,
                    predict=sequence,
                    target=targets,
                    predict_len=None,
                    target_len=target_lens,
                    ind2lab=self.label_encoder.decode_ndim,
                )
                    
                # seq2seq results
                self.per_metrics_seq.append(
                    ids=ids,
                    predict=sequence_decoder_out,
                    target=targets,
                    predict_len=None,
                    target_len=target_lens,
                    ind2lab=self.label_encoder.decode_ndim,
                )
                
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

        # Load latest checkpoint to resume training if interrupted
        ## NOTE: make sure to use the "best" model to continual training
        ## so we set the `min_key` argument
        if self.checkpointer is not None:
            # TODO: support recover best on PER or mpd_f1 or averaged model of best PER and mpd_f1
            self.checkpointer.recover_if_possible(
                min_key="PER",
                # max_key="mpd_f1",
            )
        
        # Load pretrained components if specified
        if getattr(self.hparams, 'load_pretrained_components', False):
            pretrained_path = getattr(self.hparams, 'pretrained_model_path', '')
            components = getattr(self.hparams, 'components_to_load', ['ssl', 'encoder'])
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
                    print("   Continuing with random initialization...")
            else:
                print(f"⚠️  Pretrained model path not found: {pretrained_path}")
                print("   Continuing with random initialization...")

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
                current_epoch = self.hparams.epoch_counter.current
                valid_search_interval = self.hparams.valid_search_interval
                # Log stats
                valid_stats = {
                    "loss": stage_loss,
                    "ctc_loss": self.ctc_metrics.summarize("average"),
                    "seq_loss": self.seq_metrics.summarize("average"),
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
                        "lr_pretrained": self.pretrained_opt_class.param_groups[0]["lr"],
                    },
                    train_stats={"loss": self.train_loss},
                    valid_stats=valid_stats,
                )                # Save best 3 models for each metric using simplified approach
                improved = False
                
                def save_best_model(metric_name, current_value, best_value, best_list, ckpt_prefix, 
                                meta_key, key_type, is_higher_better):
                    should_save = (current_value > best_value if is_higher_better else current_value < best_value) or len(best_list) < 3
                    
                    if should_save:
                        ckpt_name = f"{ckpt_prefix}_{epoch:03d}_{current_value:.4f}.ckpt"
                        meta = {"epoch": epoch, metric_name: current_value, meta_key: current_value}
                        if metric_name.endswith("_seq"):
                            meta.update({"PER_seq": per_seq, "mpd_f1_seq": mpd_f1_seq})
                        else:
                            meta.update({"PER": per, "mpd_f1": mpd_f1})
                        
                        self.checkpointer.save_and_keep_only(
                            meta=meta,
                            name=ckpt_name,
                            num_to_keep=self.hparams.max_save_models*4,
                            **{key_type: [meta_key]}
                        )
                        
                        best_list.append((current_value, epoch, ckpt_name))
                        best_list.sort(key=lambda x: -x[0] if is_higher_better else x[0])
                        best_list[:] = best_list[:3]
                        return best_list[0][0], True
                    return best_value, False
                    
                # Save models for each metric
                self.best_per, per_improved = save_best_model(
                    "per", per, self.best_per, self.best_per_list, 
                    "best_per", "best_PER", "min_keys", False)
                
                self.best_mpd_f1, mpd_improved = save_best_model(
                    "mpd_f1", mpd_f1, self.best_mpd_f1, self.best_mpd_f1_list,
                    "best_mpdf1", "best_mpd_f1", "max_keys", True)
                
                self.best_per_seq, per_seq_improved = save_best_model(
                    "per_seq", per_seq, self.best_per_seq, self.best_per_seq_list,
                    "best_per_seq", "best_PER_seq", "min_keys", False)
                
                self.best_mpd_f1_seq, mpd_seq_improved = save_best_model(
                    "mpd_f1_seq", mpd_f1_seq, self.best_mpd_f1_seq, self.best_mpd_f1_seq_list,
                    "best_mpd_f1_seq", "best_mpd_f1_seq", "max_keys", True)
                
                improved = per_improved or mpd_improved or per_seq_improved or mpd_seq_improved

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
                # write pure hypotheses at the end of the file 
                # id1 <hyps>
                # id2 <hyps>
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
                
            records = [x for x in self.mpd_metrics.scores]
            with open(self.hparams.output_folder + "/hyp", "w") as m:
                m.write("Hyp tokens:\n")
                for recs in records:
                    idx = recs['key']
                    ref = " ".join(recs['hypothesis'])
                    m.write(f"{idx} {ref}\n")
                print(
                    "Hypothesis tokens written to file",
                    self.hparams.output_folder + "/hyp",
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
                records = [x for x in self.mpd_metrics_seq.scores]
            with open(self.hparams.output_folder + "/ref", "w") as m:
                for recs in records:
                    idx = recs['key']
                    ref = " ".join(recs['canonical'])
                    m.write(f"{idx} {ref}\n")
                print(
                    "Canonical tokens written to file",
                    self.hparams.output_folder + "/ref",
                )
            with open(self.hparams.output_folder + "/human_seq", "w") as m:
                for recs in records:
                    idx = recs['key'] 
                    ref = " ".join(recs['perceived'])
                    m.write(f"{idx} {ref}\n")
                print(
                    "Perceived tokens written to file",
                    self.hparams.output_folder + "/human_seq",
                )
            with open(self.hparams.output_folder + "/hyp_joint", "w") as m:
                for recs in records:
                    idx = recs['key']
                    ref = " ".join(recs['hypothesis'])
                    m.write(f"{idx} {ref}\n")

    def init_optimizers(self):

        self.adam_optimizer = self.hparams.adam_opt_class(
            self.hparams.model.parameters(),
        )
        self.pretrained_opt_class = self.hparams.pretrained_opt_class(
            self.modules.perceived_ssl.parameters(), 
        )
        if self.checkpointer is not None:
            # if self.hparams.perceived_ssl is not None and not self.hparams.perceived_ssl.freeze:
            self.checkpointer.add_recoverable("adam_opt", self.adam_optimizer)
            self.checkpointer.add_recoverable("pretrained_opt", self.pretrained_opt_class)
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
            self.pretrained_opt_class.zero_grad()

            with torch.amp.autocast("cuda"):
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            # normalize the loss by gradient_accumulation and scale for mixed precision
            self.scaler.scale(loss / self.hparams.gradient_accumulation).backward()
            self.scaler.unscale_(self.adam_optimizer)
            if not self.ssl_frozen:
                self.scaler.unscale_(self.pretrained_opt_class)
            
            if check_gradients(loss):
                # Only step optimizers for non-frozen parts
                if not self.ssl_frozen:
                    self.scaler.step(self.pretrained_opt_class)
                else:
                    print("SSL model frozen, skipping SSL optimizer step")
                
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
                    if not self.ssl_frozen:
                        self.pretrained_opt_class.step()
                    else:
                        print("SSL model frozen, skipping SSL optimizer step")
                    
                    # Main optimizer always steps (includes decoder)
                    self.adam_optimizer.step()

                # Always zero gradients
                self.pretrained_opt_class.zero_grad()
                self.adam_optimizer.zero_grad()    

        return loss.detach().cpu()

class TransformerMDD_with_extra_loss(TransformerMDD):
    """TransformerMDD with extra loss for perceived phonemes"""
    
    def on_stage_start(self, stage, epoch):
        self.mispro_metrics = self.hparams.mispro_stats()
        return super().on_stage_start(stage, epoch)
    
    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
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

        feats = self.modules.perceived_ssl(wavs)  # [B, T_s, ENC_DIM]
        if len(feats.shape) == 4: 
            feats = feats[self.hparams.preceived_ssl_emb_layer]
            
        current_epoch = self.hparams.epoch_counter.current
        hyps = None
        attn_map = None

        if sb.Stage.TRAIN == stage:
            ## Todo apply mask-ctc decode
            enc_out, hidden, dec_out = self.modules.TransASR(
                src=feats,
                tgt=perceiveds_bos,
                wav_len=wav_lens,
                pad_idx=0, 
            )
            # CTC head
            h_ctc_feat = self.modules.ctc_lin(enc_out)  # [B, T_s, C]
            p_ctc_logits = self.hparams.log_softmax(h_ctc_feat)  # Log probabilities
            
            # seq2seq head
            h_seq_feat = self.modules.d_out(dec_out)  # [B, T_p+1, C]
            p_seq_logits = self.hparams.log_softmax(h_seq_feat)  # Log probabilities
            
            # d_out head for mispro detection
            h_dec_out = self.modules.d_out_mispro(p_seq_logits)  # [B, T_s, C]
            p_dec_out_logits = torch.nn.functional.sigmoid(h_dec_out)  # Sigmoid probabilities for mispro detection

        else:
            with torch.no_grad():
                enc_out, hidden, dec_out = self.modules.TransASR(
                    src=feats,
                    tgt=perceiveds_bos,
                    wav_len=wav_lens,
                    pad_idx=0,  # Assuming 0 is the padding index
                )
                h_ctc_feat = self.modules.ctc_lin(enc_out)  # [B, T_s, C]
                p_ctc_logits = self.hparams.log_softmax(h_ctc_feat)  # Log probabilities
                
                # seq2seq head
                h_seq_feat = self.modules.d_out(dec_out)  # [B, T_p+1, C]
                p_seq_logits = self.hparams.log_softmax(h_seq_feat)  # Log probabilities
                
                # d_out head for mispro detection
                # h_dec_out = self.modules.d_out_mispro(dec_out)  # [B, T_s, C]
                h_dec_out = self.modules.d_out_mispro(p_seq_logits)
                p_dec_out_logits = torch.nn.functional.sigmoid(h_dec_out)  # Sigmoid probabilities for mispro detection
                
                hyps = None
                attn_map = None
        
            valid_search_interval = self.hparams.valid_search_interval
            if current_epoch % valid_search_interval == 0:
                hyps, top_lengths, top_scores, top_log_probs = self.hparams.valid_search(enc_out.detach(), wav_lens)
                attn_map = None
            if stage == sb.Stage.TEST:
                hyps, top_lengths, top_scores, top_log_probs = self.hparams.test_search(enc_out.detach(), wav_lens)
                attn_map = None
        return {
            "p_ctc_feat": p_ctc_logits,  # [B, T_s, C]
            "p_dec_out": p_seq_logits,  # [B, T_p+1, C]
            "feats": feats,  # [B, T_s, D]
            "attn_map": attn_map,  # [B, T_p+1, T_s] or similar
            "hyps": hyps,  # [B, T_p+1] or None if not applicable
            "p_dec_out_logits": p_dec_out_logits,  # [B, T_s, C] for mispro detection
        }
    
    def compute_objectives(self, predictions, batch, stage):
        "Computes the loss for the model."
        current_epoch = self.hparams.epoch_counter.current
        valid_search_interval = self.hparams.valid_search_interval
        
        p_ctc_feat = predictions["p_ctc_feat"]
        p_dec_out = predictions["p_dec_out"]
        feats = predictions["feats"]
        attn_map = predictions["attn_map"]
        hyps = predictions.get("hyps", [])  # [B, T_p+1] or None if not applicable
        p_dec_out_logits = predictions["p_dec_out_logits"]  # [B, T_s, C] for mispro detection

        ids = batch.id
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
        
        # Mislabel rescaling for mispro detection
        mislabel_eos, mislabel_lens_eos = batch.mispro_label_eos # [B, T_p]
        
        # pdb.set_trace()
        # Caculate the loss for CTC and seq2seq outputs
        loss_ctc = self.hparams.ctc_cost(p_ctc_feat, targets, wav_lens, target_lens)
        # loss_ctc = self.hparams.ctc_cost(p_ctc_feat, targets_eos, wav_lens, target_lens_eos)
        loss_dec_out = self.hparams.seq_cost(p_dec_out, perceiveds_eos, length=perceived_lens_eos)
        loss_mispro = self.hparams.mispro_cost(p_dec_out_logits, mislabel_eos, mislabel_lens_eos)
        # pdb.set_trace()
        # def plot_mask(attn_map):
        #     """Plot attention map with mask."""
        #     import matplotlib.pyplot as plt
            
        #     # Convert to numpy for plotting
        #     attn_map_np = attn_map.cpu().numpy()
        #     if attn_map_np.ndim== 1:
        #         attn_map_np = attn_map_np[None, :]
            
        #     plt.figure(figsize=(10, 8))
        #     plt.imshow(attn_map_np, aspect='auto', cmap='viridis')
        #     plt.title("Attention Map")
        #     plt.xlabel("Target Length")
        #     plt.ylabel("Source Length")
            
        #     import time
        #     timestamp = time.strftime("%Y%m%d-%H%M%S")
        #     plt.colorbar()
        #     plt.tight_layout()
        #     plt.savefig(f"attention_map_{timestamp}.png")
        #     plt.close() 
        
        # plot_mask(mislabel)
        # plot_mask(scaled_mislabel)
        
        loss = (
            self.hparams.ctc_weight * loss_ctc 
            + (1 - self.hparams.ctc_weight) * loss_dec_out
            + self.hparams.mispro_weight * loss_mispro
            )     
        
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
                
                # print(f"GT: \n {self.label_encoder.decode_ndim(targets[-1])}")
                # print(f"CTC Greedy Decoding: \n {self.label_encoder.decode_ndim(sequence[-1])}")
                # print(f"Attention Decoder Greedy Decoding: \n {self.label_encoder.decode_ndim(sequence_decoder_out[-1])}")

                self.ctc_metrics.append(ids, p_ctc_feat, targets, wav_lens, target_lens)
                self.seq_metrics.append(ids, log_probabilities=p_dec_out, targets=perceiveds_eos, length=perceived_lens_eos)
                # pdb.set_trace()
                self.mispro_metrics.append(ids, p_dec_out_logits, mislabel_eos, mislabel_lens_eos)
                # self.ctc_metrics_fuse.append(ids, sequence_decoder_out, targets, wav_lens, target_lens)
                
                # CTC-only results
                self.per_metrics.append(
                    ids=ids,
                    predict=sequence,
                    target=targets,
                    predict_len=None,
                    target_len=target_lens,
                    ind2lab=self.label_encoder.decode_ndim,
                )
                    
                # seq2seq results
                # might need eos removal
                self.per_metrics_seq.append(
                    ids=ids,
                    predict=sequence_decoder_out,
                    target=targets,
                    predict_len=None,
                    target_len=target_lens,
                    ind2lab=self.label_encoder.decode_ndim,
                )
                
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
                    wandb.log({"test_loss_dec_out": loss_dec_out.item()}, step=self.hparams.epoch_counter.current)
                if loss_ctc is not None:
                    wandb.log({"test_loss_ctc_head": loss_ctc.item()}, step=self.hparams.epoch_counter.current)
                if loss_mispro is not None:
                    wandb.log({"test_loss_mispro": loss_mispro.item()}, step=self.hparams.epoch_counter.current)
            except Exception:
                pass
        return loss

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
                
                current_epoch = self.hparams.epoch_counter.current
                valid_search_interval = self.hparams.valid_search_interval
                # Log stats
                valid_stats = {
                    "loss": stage_loss,
                    "ctc_loss": self.ctc_metrics.summarize("average"),
                    "seq_loss": self.seq_metrics.summarize("average"),
                    "mispro_loss": self.mispro_metrics.summarize("average"),
                    "PER": per,
                    "mpd_f1": mpd_f1,
                    "PER_seq": per_seq,
                    "mpd_f1_seq": mpd_f1_seq,
                }
            
                self.hparams.train_logger.log_stats(
                    stats_meta={
                        "epoch": epoch,
                        "lr_adam": self.adam_optimizer.param_groups[0]["lr"],
                        "lr_pretrained": self.pretrained_opt_class.param_groups[0]["lr"],
                    },
                    train_stats={"loss": self.train_loss},
                    valid_stats=valid_stats,
                )                # Save best 3 models for each metric using simplified approach
                improved = False
                
                def save_best_model(metric_name, current_value, best_value, best_list, ckpt_prefix, 
                                meta_key, key_type, is_higher_better):
                    should_save = (current_value > best_value if is_higher_better else current_value < best_value) or len(best_list) < 3
                    
                    if should_save:
                        ckpt_name = f"{ckpt_prefix}_{epoch:03d}_{current_value:.4f}.ckpt"
                        meta = {"epoch": epoch, metric_name: current_value, meta_key: current_value}
                        if metric_name.endswith("_seq"):
                            meta.update({"PER_seq": per_seq, "mpd_f1_seq": mpd_f1_seq})
                        else:
                            meta.update({"PER": per, "mpd_f1": mpd_f1})
                        
                        self.checkpointer.save_and_keep_only(
                            meta=meta,
                            name=ckpt_name,
                            num_to_keep=self.hparams.max_save_models*4,
                            **{key_type: [meta_key]}
                        )
                        
                        best_list.append((current_value, epoch, ckpt_name))
                        best_list.sort(key=lambda x: -x[0] if is_higher_better else x[0])
                        best_list[:] = best_list[:self.hparams.max_save_models]
                        return best_list[0][0], True
                    return best_value, False
                    
                # Save models for each metric
                self.best_per, per_improved = save_best_model(
                    "per", per, self.best_per, self.best_per_list, 
                    "best_per", "best_PER", "min_keys", False)
                
                self.best_mpd_f1, mpd_improved = save_best_model(
                    "mpd_f1", mpd_f1, self.best_mpd_f1, self.best_mpd_f1_list,
                    "best_mpdf1", "best_mpd_f1", "max_keys", True)
                
                self.best_per_seq, per_seq_improved = save_best_model(
                    "per_seq", per_seq, self.best_per_seq, self.best_per_seq_list,
                    "best_per_seq", "best_PER_seq", "min_keys", False)
                
                self.best_mpd_f1_seq, mpd_seq_improved = save_best_model(
                    "mpd_f1_seq", mpd_f1_seq, self.best_mpd_f1_seq, self.best_mpd_f1_seq_list,
                    "best_mpd_f1_seq", "best_mpd_f1_seq", "max_keys", True)
                
                improved = per_improved or mpd_improved or per_seq_improved or mpd_seq_improved

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
                    "seq_loss": self.seq_metrics.summarize("average"),
                    "mispro_loss": self.mispro_metrics.summarize("average"),
                    "PER": per,
                    "mpd_f1": mpd_f1,
                    "PER_seq": per_seq,
                    "mpd_f1_seq": mpd_f1_seq,
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
            self.hparams.model.parameters(),
        )
        self.pretrained_opt_class = self.hparams.pretrained_opt_class(
            self.modules.perceived_ssl.parameters(), 
        )
        self.mispro_opt_class = self.hparams.mispro_opt_class(
            self.hparams.model_mispro.parameters(),
        )
        if self.checkpointer is not None:
            # if self.hparams.perceived_ssl is not None and not self.hparams.perceived_ssl.freeze:
            self.checkpointer.add_recoverable("adam_opt", self.adam_optimizer)
            self.checkpointer.add_recoverable("pretrained_opt", self.pretrained_opt_class)
            self.checkpointer.add_recoverable("mispro_opt", self.mispro_opt_class)
            
            
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
            self.pretrained_opt_class.zero_grad()
            self.mispro_opt_class.zero_grad()

            with torch.amp.autocast("cuda"):
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            # normalize the loss by gradient_accumulation and scale for mixed precision
            self.scaler.scale(loss / self.hparams.gradient_accumulation).backward()
            self.scaler.unscale_(self.adam_optimizer)
            self.scaler.unscale_(self.pretrained_opt_class)
            self.scaler.unscale_(self.mispro_opt_class)
            
            if check_gradients(loss):
                self.scaler.step(self.pretrained_opt_class)
                self.scaler.step(self.adam_optimizer)
                self.scaler.step(self.mispro_opt_class)
                
            self.scaler.update()

        else:
            
            outputs = self.compute_forward(batch, sb.Stage.TRAIN)
            
            loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            # normalize the loss by gradient_accumulation step
            (loss / self.hparams.gradient_accumulation).backward()

            if self.step % self.hparams.gradient_accumulation == 0:
                # gradient clipping & early stop if loss is not fini
                
                if check_gradients(loss):
                    self.pretrained_opt_class.step()
                    self.adam_optimizer.step()
                    self.mispro_opt_class.step()

                self.pretrained_opt_class.zero_grad()
                self.adam_optimizer.zero_grad()    
                self.mispro_opt_class.zero_grad()

        return loss.detach().cpu()

        return super().fit_batch(batch)

class TransformerMDD_dual_path(TransformerMDD):
    """TransformerMDD with dual path for perceived / canonical phonemes"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.        super().__init__(*args, **kwargs)

        self.cano_best_per_list = []  # List of (PER, epoch, ckpt_name)
        self.cano_best_mpd_f1_list = []  # List of (mpd_f1, epoch, ckpt_name)
        self.cano_best_per = float('inf')
        self.cano_best_mpd_f1 = float('-inf')
        
        self.cano_best_per_seq_list = []  # List of (PER_seq, epoch, ckpt_name)
        self.cano_best_mpd_f1_seq_list = []
        self.cano_best_per_seq = float('inf')
        self.cano_best_mpd_f1_seq = float('-inf')
            
        self.best_valid_loss = float('inf')
        self.best_valid_loss_list = []  # List of (valid_loss, epoch, ckpt_name)label_encoder.add_label("<eos>")
        
    def on_stage_start(self, stage, epoch):
        self.cano_ctc_metrics = self.hparams.ctc_stats()
        self.cano_seq_metrics = self.hparams.seqlabel_stats()
        if stage == sb.Stage.TRAIN:
            self.cano_per_metrics = self.hparams.per_stats()
            self.cano_per_metrics_seq = self.hparams.per_stats()
        return super().on_stage_start(stage, epoch)
    
    def compute_forward_cano(self, batch, stage):
        batch = batch.to(self.device)
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
        feats = self.modules.canonical_ssl(wavs)  # [B, T_s, ENC_DIM]
        if len(feats.shape) == 4: 
            feats = feats[self.hparams.canonical_ssl_emb_layer]
            
        current_epoch = self.hparams.epoch_counter.current
        hyps = None
        attn_map = None

        if sb.Stage.TRAIN == stage:
            enc_out, hidden, dec_out = self.modules.TransASR_cano(
                src=feats,
                tgt=canonicals_bos,
                wav_len=wav_lens,
                pad_idx=0, 
            )
            # CTC head
            h_ctc_feat = self.modules.ctc_lin_cano(enc_out)  # [B, T_s, C]
            p_ctc_logits = self.hparams.log_softmax(h_ctc_feat)  # Log probabilities

            # seq2seq head
            h_seq_feat = self.modules.d_out_cano(dec_out)  # [B, T_p+1, C]
            p_seq_logits = self.hparams.log_softmax(h_seq_feat)  # Log probabilities

        else:
            with torch.no_grad():
                enc_out, hidden, dec_out = self.modules.TransASR_cano(
                    src=feats,
                    tgt=canonicals_bos,
                    wav_len=wav_lens,
                    pad_idx=0,  # Assuming 0 is the padding index
                )
                h_ctc_feat = self.modules.ctc_lin_cano(enc_out)  # [B, T_s, C]
                p_ctc_logits = self.hparams.log_softmax(h_ctc_feat)  # Log probabilities

                # seq2seq head
                h_seq_feat = self.modules.d_out_cano(dec_out)  # [B, T_p+1, C]
                p_seq_logits = self.hparams.log_softmax(h_seq_feat)  # Log probabilities
                
                hyps = None
                attn_map = None
        
            valid_search_interval = self.hparams.valid_search_interval
            if current_epoch % valid_search_interval == 0:
                hyps, top_lengths, top_scores, top_log_probs = self.hparams.valid_search_cano(enc_out.detach(), wav_lens)
                attn_map = None
            if stage == sb.Stage.TEST:
                hyps, top_lengths, top_scores, top_log_probs = self.hparams.test_search_cano(enc_out.detach(), wav_lens)
                attn_map = None
        return {
            "p_ctc_feat": p_ctc_logits,  # [B, T_s, C]
            "p_dec_out": p_seq_logits,  # [B, T_p+1, C]
            "feats": feats,  # [B, T_s, D]
            "attn_map": attn_map,  # [B, T_p+1, T_s] or similar
            "hyps": hyps,  # [B, T_p+1] or None if not applicable
        }
        
    def compute_forward_perc(self, batch, stage):
        return super().compute_forward(batch, stage)
    
    def compute_forward(self, batch, stage):
        """Compute forward pass for both canonical and perceived phonemes."""
        loss_perc = self.compute_forward_perc(batch, stage)
        loss_cano = self.compute_forward_cano(batch, stage)

        return {
            "loss_perc": loss_perc,
            "loss_cano": loss_cano,
        }
    
    def compute_objectives_cano(self, predictions, batch, stage):
        
        """Computes the loss for the model."""
        current_epoch = self.hparams.epoch_counter.current
        valid_search_interval = self.hparams.valid_search_interval
        
        p_ctc_feat_cano = predictions["p_ctc_feat"]
        p_dec_out_cano = predictions["p_dec_out"]
        feats_cano = predictions["feats"]
        attn_map_cano = predictions["attn_map"]
        hyps_cano = predictions.get("hyps", [])

        ids = batch.id
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
    
        loss_ctc = self.hparams.ctc_cost(p_ctc_feat_cano, canonicals, wav_lens, canonical_lens)
        # pdb.set_trace()
        loss_dec_out = self.hparams.seq_cost(p_dec_out_cano, canonicals_eos, length=canonical_lens_eos)

        loss_cano = (
            self.hparams.ctc_weight * loss_ctc
            + (1 - self.hparams.ctc_weight) * loss_dec_out
        )
        
        if stage != sb.Stage.TRAIN:
            if current_epoch % valid_search_interval == 0 or (
                    stage == sb.Stage.TEST
                ):
                    # Record losses for posterit
                    # Traditional CTC greedy decoding
                sequence = sb.decoders.ctc_greedy_decode(
                    p_ctc_feat_cano, wav_lens, blank_id=self.hparams.blank_index
                )
                sequence_decoder_out = hyps_cano  # [B, T_p+1]

                self.cano_ctc_metrics.append(ids, p_ctc_feat_cano, canonicals, wav_lens, canonical_lens)
                self.cano_seq_metrics.append(ids, log_probabilities=p_dec_out_cano, targets=canonicals_eos, length=canonical_lens_eos)

                # self.ctc_metrics_fuse.append(ids, sequence_decoder_out, targets, wav_lens, target_lens)
                
                # CTC-only results
                self.cano_per_metrics.append(
                    ids=ids,
                    predict=sequence,
                    target=canonicals,
                    predict_len=None,
                    target_len=canonical_lens,
                    ind2lab=self.label_encoder.decode_ndim,
                )
                    
                # seq2seq results
                self.cano_per_metrics_seq.append(
                    ids=ids,
                    predict=sequence_decoder_out,
                    target=canonicals_eos,
                    predict_len=None,
                    target_len=canonical_lens_eos,
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
                    "loss_cano": loss_cano.item(),
                }, step=self.hparams.epoch_counter.current)
                # if loss_ga is not None:
                #     wandb.log({"loss_ga": loss_ga.item()}, step=self.hparams.epoch_counter.current)
                if loss_dec_out is not None:
                    wandb.log({"loss_dec_out_cano": loss_dec_out.item()}, step=self.hparams.epoch_counter.current)
                if loss_ctc is not None:
                    wandb.log({"loss_ctc_head_cano": loss_ctc.item()}, step=self.hparams.epoch_counter.current)
            except Exception:
                pass
        return loss_cano
    
    def compute_objectives_perc(self, predictions, batch, stage):
        # perceived path
        loss_perc = super().compute_objectives(predictions['loss_perc'], batch, stage)
        return loss_perc

    def compute_objectives(self, predictions, batch, stage):
        loss_perc = super().compute_objectives(predictions["loss_perc"], batch, stage)
        loss_cano = self.compute_objectives_cano(predictions["loss_cano"], batch, stage)
        loss = 0.5 *(loss_perc + loss_cano)

            
        return loss
    
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
                
                # Canonical metrics
                per_cano = self.cano_per_metrics.summarize("error_rate")
                per_seq_cano = self.cano_per_metrics_seq.summarize("error_rate")
        

                current_epoch = self.hparams.epoch_counter.current
                valid_search_interval = self.hparams.valid_search_interval
                # Log stats
                valid_stats = {
                    "loss": stage_loss,
                    "ctc_loss": self.ctc_metrics.summarize("average"),
                    "seq_loss": self.seq_metrics.summarize("average"),
                    "PER": per,
                    "mpd_f1": mpd_f1,
                    "PER_seq": per_seq,
                    
                    "ctc_loss_cano": self.cano_ctc_metrics.summarize("average"),
                    "seq_loss_cano": self.cano_seq_metrics.summarize("average"),
                    "mpd_f1_seq": mpd_f1_seq,
                    "PER_cano": per_cano,
                    "PER_seq_cano": per_seq_cano,
                }
            
                self.hparams.train_logger.log_stats(
                    stats_meta={
                        "epoch": epoch,
                        "lr_adam": self.adam_optimizer.param_groups[0]["lr"],
                        "lr_pretrained": self.pretrained_opt_class.param_groups[0]["lr"],
                    },
                    train_stats={"loss": self.train_loss},
                    valid_stats=valid_stats,
                )                # Save best 3 models for each metric using simplified approach
                improved = False
                
                def save_best_model(metric_name, current_value, best_value, best_list, ckpt_prefix, 
                                meta_key, key_type, is_higher_better):
                    should_save = (current_value > best_value if is_higher_better else current_value < best_value) or len(best_list) < 3
                    
                    if should_save:
                        ckpt_name = f"{ckpt_prefix}_{epoch:03d}_{current_value:.4f}.ckpt"
                        meta = {"epoch": epoch, metric_name: current_value, meta_key: current_value}
                        if metric_name.endswith("_seq"):
                            meta.update({"PER_seq": per_seq, "mpd_f1_seq": mpd_f1_seq})
                        else:
                            meta.update({"PER": per, "mpd_f1": mpd_f1})
                        
                        self.checkpointer.save_and_keep_only(
                            meta=meta,
                            name=ckpt_name,
                            num_to_keep=5,
                            **{key_type: [meta_key]}
                        )
                        
                        best_list.append((current_value, epoch, ckpt_name))
                        best_list.sort(key=lambda x: -x[0] if is_higher_better else x[0])
                        best_list[:] = best_list[:3]
                        return best_list[0][0], True
                    return best_value, False
                    
                # Save models for each metric
                self.best_per, per_improved = save_best_model(
                    "per", per, self.best_per, self.best_per_list, 
                    "best_per_joint", "best_PER", "min_keys", False)
                
                self.best_mpd_f1, mpd_improved = save_best_model(
                    "mpd_f1", mpd_f1, self.best_mpd_f1, self.best_mpd_f1_list,
                    "best_mpdf1_joint", "best_mpd_f1", "max_keys", True)
                
                self.best_per_seq, per_seq_improved = save_best_model(
                    "per_seq", per_seq, self.best_per_seq, self.best_per_seq_list,
                    "best_per_seq_joint", "best_PER_seq", "min_keys", False)
                
                self.best_mpd_f1_seq, mpd_seq_improved = save_best_model(
                    "mpd_f1_seq", mpd_f1_seq, self.best_mpd_f1_seq, self.best_mpd_f1_seq_list,
                    "best_mpd_f1_seq_joint", "best_mpd_f1_seq", "max_keys", True)
                
                improved = per_improved or mpd_improved or per_seq_improved or mpd_seq_improved

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
                    "seq_loss": self.seq_metrics.summarize("average"),
                    "PER": per,
                    "mpd_f1": mpd_f1,
                    "PER_seq": per_seq,
                    "mpd_f1_seq": mpd_f1_seq,
                    
                    "ctc_loss_cano": self.cano_ctc_metrics.summarize("average"),
                    "seq_loss_cano": self.cano_seq_metrics.summarize("average"),
                    "PER_cano": per_cano,
                    "PER_seq_cano": per_seq_cano,
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
                # TODO print result
                pdb.set_trace()
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
            # write canonical results
            with open(self.hparams.per_cano_file, "w") as w:
                w.write("Canonical CTC loss stats:\n")
                self.cano_ctc_metrics.write_stats(w)
                w.write("\nCanonical PER stats:\n")
                self.cano_per_metrics.write_stats(w)
                print(
                    "Canonical CTC and PER stats written to file",
                    self.hparams.per_cano_file,
                )
            if not hasattr(self.hparams, 'cano_per_seq_file'):
                self.hparams.cano_per_seq_file = self.hparams.per_cano_file.replace(".txt", "_seq.txt")
            with open(self.hparams.cano_per_seq_file, "w") as w:
                w.write("Canonical Joint CTC-Attention PER stats:\n")
                self.cano_per_metrics_seq.write_stats(w)
                print(
                    "Canonical Joint CTC-Attention PER stats written to file",
                    self.hparams.cano_per_seq_file,
                )
    
    def init_optimizers(self):

        self.adam_optimizer = self.hparams.adam_opt_class(
            self.hparams.model.parameters(),
        )
        self.adam_optimizer_cano = self.hparams.adam_opt_class(
            self.hparams.model_cano.parameters(),
        )
        self.pretrained_opt_class = self.hparams.pretrained_opt_class(
            self.modules.perceived_ssl.parameters(), 
        )
        self.pretrained_opt_class_cano = self.hparams.pretrained_opt_class(
            self.modules.canonical_ssl.parameters(), 
        )
        
        if self.checkpointer is not None:
            # if self.hparams.perceived_ssl is not None and not self.hparams.perceived_ssl.freeze:
            self.checkpointer.add_recoverable("adam_opt", self.adam_optimizer)
            self.checkpointer.add_recoverable("pretrained_opt", self.pretrained_opt_class)
            self.checkpointer.add_recoverable("adam_opt_cano", self.adam_optimizer_cano)
            self.checkpointer.add_recoverable("pretrained_opt_cano", self.pretrained_opt_class_cano)
    
    def on_fit_batch_end(self, batch, outputs, loss, should_step):
        """At the end of the optimizer step, apply noam annealing."""
        if should_step:
            self.hparams.noam_annealing(self.adam_optimizer)
            self.hparams.noam_annealing(self.adam_optimizer_cano)

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
            self.pretrained_opt_class.zero_grad()
            self.adam_optimizer_cano.zero_grad()
            self.pretrained_opt_class_cano.zero_grad()

            with torch.amp.autocast("cuda"):
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            # normalize the loss by gradient_accumulation and scale for mixed precision
            self.scaler.scale(loss / self.hparams.gradient_accumulation).backward()
            self.scaler.unscale_(self.adam_optimizer)
            self.scaler.unscale_(self.pretrained_opt_class)
            self.scaler.unscale_(self.adam_optimizer_cano)
            self.scaler.unscale_(self.pretrained_opt_class_cano)

            if check_gradients(loss):
                self.scaler.step(self.pretrained_opt_class)
                self.scaler.step(self.adam_optimizer)
                self.scaler.step(self.pretrained_opt_class_cano)
                self.scaler.step(self.adam_optimizer_cano)

            self.scaler.update()

        else:
            
            outputs = self.compute_forward(batch, sb.Stage.TRAIN)
            
            loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            # normalize the loss by gradient_accumulation step
            (loss / self.hparams.gradient_accumulation).backward()

            if self.step % self.hparams.gradient_accumulation == 0:
                # gradient clipping & early stop if loss is not fini
                
                if check_gradients(loss):
                    self.pretrained_opt_class.step()
                    self.adam_optimizer.step()
                    self.pretrained_opt_class_cano.step()
                    self.adam_optimizer_cano.step()

                self.pretrained_opt_class.zero_grad()
                self.adam_optimizer.zero_grad()
                self.pretrained_opt_class_cano.zero_grad()
                self.adam_optimizer_cano.zero_grad()

        return loss.detach().cpu()
class TransformerMDD_dual_ctc(TransformerMDD):
    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
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
    
        feats = self.modules.perceived_ssl(wavs)  # [B, T_s, ENC_DIM]
        
        before_enc_feats = self.modules.enc(feats)  # [B, T_s, D]
        # CTC detection before Transformer
        p_pre_enc_feats = self.modules.ctc_lin(before_enc_feats)  # [B, T_s, C]
        
        # pre_enc
        current_epoch = self.hparams.epoch_counter.current
        hyps = None
        attn_map = None

        if sb.Stage.TRAIN == stage:
            # pdb.set_trace()
            enc_out, hidden, dec_out = self.modules.TransASR(
                src=before_enc_feats,
                tgt=targets_bos,
                wav_len=wav_lens,
                pad_idx=0, 
            )
            # CTC head
            h_ctc_feat = self.modules.ctc_lin_after_enc(enc_out)  # [B, T_s, C]
            p_ctc_logits = self.hparams.log_softmax(h_ctc_feat)  # Log probabilities

            # seq2seq head
            h_seq_feat = self.modules.d_out(dec_out)  # [B, T_p+1, C]
            p_seq_logits = self.hparams.log_softmax(h_seq_feat)  # Log probabilities

        else:
            with torch.no_grad():
                enc_out, hidden, dec_out = self.modules.TransASR(
                    src=before_enc_feats,
                    tgt=targets_bos,
                    wav_len=wav_lens,
                    pad_idx=0,  # Assuming 0 is the padding index
                )
                h_ctc_feat = self.modules.ctc_lin_after_enc(enc_out)  # [B, T_s, C]
                p_ctc_logits = self.hparams.log_softmax(h_ctc_feat)  # Log probabilities

                # seq2seq head
                h_seq_feat = self.modules.d_out(dec_out)  # [B, T_p+1, C]
                p_seq_logits = self.hparams.log_softmax(h_seq_feat)  # Log probabilities
                
                hyps = None
                attn_map = None
        
            valid_search_interval = self.hparams.valid_search_interval
            if current_epoch % valid_search_interval == 0:
                hyps, top_lengths, top_scores, top_log_probs = self.hparams.valid_search(enc_out.detach(), wav_lens)
                attn_map = None
            if stage == sb.Stage.TEST:
                hyps, top_lengths, top_scores, top_log_probs = self.hparams.test_search(enc_out.detach(), wav_lens)
                attn_map = None
        return {
            "p_ctc_feat": p_ctc_logits,  # [B, T_s, C]
            "p_dec_out": p_seq_logits,  # [B, T_p+1, C]
            "feats": feats,  # [B, T_s, D]
            "attn_map": attn_map,  # [B, T_p+1, T_s] or similar
            "hyps": hyps,  # [B, T_p+1] or None if not applicable
            "p_pre_enc_feats": p_pre_enc_feats,  # [B, T_s, C]
        }
    def compute_objectives(self, predictions, batch, stage):
        "Computes the loss for the model."
        current_epoch = self.hparams.epoch_counter.current
        valid_search_interval = self.hparams.valid_search_interval
        
        p_ctc_feat = predictions["p_ctc_feat"]
        p_dec_out = predictions["p_dec_out"]
        feats = predictions["feats"]
        attn_map = predictions["attn_map"]
        hyps = predictions.get("hyps", [])  # [B, T_p+1] or None if not applicable
        p_pre_enc_feats = predictions["p_pre_enc_feats"]  # [B, T_s, C]

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
        
        # if sb.Stage.TRAIN == stage and hasattr(self.hparams, "wav_augment"):
        #     wavs, wav_lens = self.hparams.wav_augment(wavs, wav_lens)
        #     targets = self.hparams.wav_augment.replicate_labels(targets)
        #     target_lens = self.hparams.wav_augment.replicate_labels(target_lens)
        #     targets_bos = self.hparams.wav_augment.replicate_labels(targets_bos)
        #     target_lens_bos = self.hparams.wav_augment.replicate_labels(target_lens_bos)
        #     targets_eos = self.hparams.wav_augment.replicate_labels(targets_eos)
        #     target_lens_eos = self.hparams.wav_augment.replicate_labels(target_lens_eos)

        # Caculate the loss for CTC and seq2seq outputs
        # loss_ctc = self.hparams.ctc_cost(p_ctc_feat, targets, wav_lens, target_lens)
        loss_ctc = self.hparams.ctc_cost(p_ctc_feat, targets, wav_lens, target_lens)
        loss_dec_out = self.hparams.seq_cost(p_dec_out, targets_eos, target_lens_eos)
        loss_ctc_post_enc = self.hparams.ctc_cost(p_pre_enc_feats, targets, wav_lens, target_lens)
        
        # if stage == sb.Stage.TRAIN:
        #     # # TODO: Better to Train CELoss without BOS / EOS
        #     # pdb.set_trace()
        #     # loss_dec_out = torch.tensor(0.0, device=self.device)

        #     loss_dec_out = sb.nnet.losses.kldiv_loss(
        #         p_dec_out,
        #         targets_eos,
        #         length=target_lens_eos,
        #         label_smoothing=0.1,
        #         reduction="batchmean",
        #     )

        # else: 
        #     # During inference, don't compute attention loss
        #     # loss_dec_out = torch.tensor(0.0, device=self.device)
        #     loss_dec_out = sb.nnet.losses.kldiv_loss(
        #         p_dec_out,
        #         targets_eos,
        #         length=target_lens_eos,
        #         label_smoothing=0.1,
        #         reduction="batchmean",
        #     )
        ctc_loss_merge = self.hparams.dual_ctc_loss_weight * loss_ctc + (1 - self.hparams.dual_ctc_loss_weight) * loss_ctc_post_enc

        loss = (self.hparams.ctc_weight * ctc_loss_merge
            + (1 - self.hparams.ctc_weight) * loss_dec_out)
        
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
                
                # print(f"GT: \n {self.label_encoder.decode_ndim(targets[-1])}")
                # print(f"CTC Greedy Decoding: \n {self.label_encoder.decode_ndim(sequence[-1])}")
                # print(f"Attention Decoder Greedy Decoding: \n {self.label_encoder.decode_ndim(sequence_decoder_out[-1])}")

                self.ctc_metrics.append(ids, p_ctc_feat, targets, wav_lens, target_lens)
                self.seq_metrics.append(ids, log_probabilities=p_dec_out, targets=targets_eos, length=target_lens_eos)
                
                # self.ctc_metrics_fuse.append(ids, sequence_decoder_out, targets, wav_lens, target_lens)
                
                # CTC-only results
                self.per_metrics.append(
                    ids=ids,
                    predict=sequence,
                    target=targets,
                    predict_len=None,
                    target_len=target_lens,
                    ind2lab=self.label_encoder.decode_ndim,
                )
                    
                # seq2seq results
                self.per_metrics_seq.append(
                    ids=ids,
                    predict=sequence_decoder_out,
                    target=targets,
                    predict_len=None,
                    target_len=target_lens,
                    ind2lab=self.label_encoder.decode_ndim,
                )
                
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
            except Exception:
                pass
        return loss