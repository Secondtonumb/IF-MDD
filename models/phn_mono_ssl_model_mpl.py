"""
Momentum Pseudo Label (MPL) Training for Phoneme-level Mispronunciation Detection

This module extends PhnMonoSSLModel with semi-supervised learning capabilities
using a momentum-updated teacher model to generate pseudo labels for unlabeled data.

Key Features:
- Teacher model with exponential moving average (EMA) update
- Support for mixed labeled/unlabeled training
- Compatible with all encoder types (Conformer, Zipformer, RVQ, Linear)
- Compatible with all loss types (Vanilla, OTTC, CR-CTC, CR-OTTC)

Reference: Meta Pseudo Labels (Pham et al., 2021)
"""

import os
import sys
import math
import time
import copy
import itertools
import logging
import torch
import torch.nn as nn
import speechbrain as sb
from speechbrain.core import Stage
from speechbrain.utils.distributed import run_on_main
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from models.phn_mono_ssl_model_v3_refactored import PhnMonoSSLModel, TestResults, InferenceResult

logger = logging.getLogger(__name__)


class PhnMonoSSLModel_MPL(PhnMonoSSLModel):
    """
    Momentum Pseudo Label (MPL) variant of PhnMonoSSLModel.
    
    This model supports semi-supervised training with:
    - A student model (self.modules) that learns from both labeled and pseudo-labeled data
    - A teacher model (self.modules_teacher) that generates pseudo labels via EMA updates
    
    Usage:
        # In hparams yaml, define modules_teacher with same structure as modules
        asr_brain = PhnMonoSSLModel_MPL(...)
        asr_brain.fit_mpl(epoch_counter, train_data_l, train_data_u, valid_data)
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # MPL-specific attributes
        self.modules_teacher = None  # Will be initialized in on_fit_start
        self.momentum_factor = 0.999  # EMA momentum factor
        self.n_train_batch = 0
        
        # Track separate losses for labeled vs unlabeled
        self.labeled_loss_sum = 0.0
        self.labeled_loss_count = 0
        self.unlabeled_loss_sum = 0.0
        self.unlabeled_loss_count = 0
    
    def init_teacher_model(self):
        """
        Initialize teacher model as a copy of the student model.
        Teacher uses EMA updates and generates pseudo labels.
        """
        # import pdb; pdb.set_trace()
        if hasattr(self.hparams, 'modules_teacher'):
            # Use pre-defined teacher modules from hparams
            self.modules_teacher = nn.ModuleDict(self.hparams.modules_teacher).to(self.device)
            
            # Load weights from student model or checkpoint
            # import pdb; pdb.set_trace()
            if self.checkpointer is not None:
                chosen_ckpt = self.checkpointer.find_checkpoint(min_key="PER")
                if chosen_ckpt is not None:
                    # Load from checkpoint, this is for resuming training
                    for name, module in self.modules_teacher.items():
                        ckpt_path = chosen_ckpt.paramfiles.get(name)
                        if ckpt_path and os.path.exists(ckpt_path):
                            module.load_state_dict(
                                torch.load(ckpt_path, map_location=self.device)
                            )
                            logger.info(f"✅ Loaded teacher {name} from checkpoint")
                else:
                    # Copy from student, this is for fresh start
                    self._copy_student_to_teacher()
            else:

                self._copy_student_to_teacher()
        else:
            # Create teacher as deep copy of student modules
            logger.info("Creating teacher model as copy of student...")
            self.modules_teacher = nn.ModuleDict()
            for name, module in self.modules.items():
                self.modules_teacher[name] = copy.deepcopy(module)
            self.modules_teacher.to(self.device)
        
        # Freeze teacher (no gradient computation)
        for param in self.modules_teacher.parameters():
            param.requires_grad = False
        
        self.modules_teacher.eval()
        logger.info("✅ Teacher model initialized")
    
    def _copy_student_to_teacher(self):
        """Copy student weights to teacher"""
        with torch.no_grad():
            for name in self.modules_teacher.keys():
                if name in self.modules:
                    student_state = self.modules[name].state_dict()
                    self.modules_teacher[name].load_state_dict(student_state)
    
    def set_momentum_factor(self, n_train_batch, n_epochs):
        """
        Calculate momentum factor for EMA updates.
        
        The momentum factor is computed such that after all training steps,
        the teacher model contribution from the initial weights is `base_model_factor`.
        
        Args:
            n_train_batch: Number of training batches per epoch
            n_epochs: Number of training epochs
        """
        base_factor = getattr(self.hparams, 'base_model_factor', 0.01)
        gradient_accumulation = getattr(self.hparams, 'gradient_accumulation', 1)
        
        total_steps = float(n_train_batch * n_epochs // gradient_accumulation)
        if total_steps > 0:
            self.momentum_factor = math.exp((1 / total_steps) * math.log(base_factor))
        else:
            self.momentum_factor = 0.999
        
        logger.info(f"📊 MPL Momentum Factor: {self.momentum_factor:.6f}")
        logger.info(f"   Total steps: {total_steps}, Base factor: {base_factor}")
    
    def teacher_momentum_update(self):
        """
        Update teacher model weights using exponential moving average (EMA).
        
        teacher = momentum * teacher + (1 - momentum) * student
        """
        with torch.no_grad():
            student_state = self.modules.state_dict()
            teacher_state = self.modules_teacher.state_dict()
            
            for pname, param in teacher_state.items():
                if pname in student_state:
                    param.copy_(
                        self.momentum_factor * param + 
                        (1 - self.momentum_factor) * student_state[pname]
                    )
    
    def infer_batch_teacher(self, batch):
        """
        Generate pseudo labels using teacher model.
        
        Args:
            batch: Unlabeled batch (has sig but no phn_encoded_target)
            
        Returns:
            pseudo_labels: [B, max_len] tensor of pseudo label indices
            pseudo_label_lens: [B] tensor of relative lengths
        """
        # Ensure teacher is in eval mode with augmentation disabled
        self.modules_teacher.eval()
        
        # Disable spec augment for teacher if applicable
        if hasattr(self.modules_teacher, 'perceived_ssl'):
            ssl_model = self.modules_teacher.perceived_ssl
            if hasattr(ssl_model, 'model') and hasattr(ssl_model.model, 'config'):
                ssl_model.model.config.apply_spec_augment = False
        
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        
        with torch.no_grad():
            # Extract SSL features (teacher model)
            feats = self.modules_teacher.perceived_ssl(wavs)
            
            # Encode features - handle different encoder configurations
            x = feats
            
            # Linear projection (enc or ssl_proj)
            if hasattr(self.modules_teacher, 'enc') and self.modules_teacher.enc is not None:
                x = self.modules_teacher.enc(x)
            elif hasattr(self.modules_teacher, 'ssl_proj') and self.modules_teacher.ssl_proj is not None:
                x = self.modules_teacher.ssl_proj(x)
            
            # Apply Conformer if present
            if hasattr(self.modules_teacher, 'ConformerEncoder') and self.modules_teacher.ConformerEncoder is not None:
                conformer = self.modules_teacher.ConformerEncoder
                attention_type = getattr(conformer, 'attention_type', None)
                
                if attention_type == "RelPosEncXL":
                    # Some conformer implementations need positional embeddings
                    try:
                        from speechbrain.nnet.attention import RelPosEncXL
                        pos_emb = RelPosEncXL(emb_dim=x.shape[-1])(x).to(self.device)
                        x, _ = conformer(x, pos_embs=pos_emb)
                    except:
                        # Fallback: some conformer implementations don't need pos_embs
                        x, _ = conformer(x)
                else:
                    x, _ = conformer(x)
            
            # CTC output
            logits = self.modules_teacher.ctc_lin(x)
            p_ctc = self.hparams.log_softmax(logits)
            
            # Greedy decode to get pseudo labels
            pseudo_labels = sb.decoders.ctc_greedy_decode(
                p_ctc, wav_lens, blank_id=self.hparams.blank_index
            )
            
            # Convert to padded tensor
            max_len = max(len(seq) for seq in pseudo_labels) if pseudo_labels else 1
            max_len = max(max_len, 1)  # Ensure at least length 1
            
            pseudo_label_lens = torch.tensor(
                [float(len(seq) / max_len) for seq in pseudo_labels],
                device=self.device
            )
            
            pseudo_labels_padded = pad_sequence(
                [torch.tensor(seq, device=self.device) for seq in pseudo_labels],
                batch_first=True,
                padding_value=0
            )
        
        return pseudo_labels_padded, pseudo_label_lens
    
    def compute_objectives_unlabeled(self, predictions, batch, pseudo_labels, pseudo_label_lens):
        """
        Compute CTC loss for unlabeled data using pseudo labels.
        
        Uses CTCLossManager to support all loss types (vanilla, OTTC, CR-CTC, CR-OTTC)
        with pseudo labels. Unlike labeled data, we don't have canonical/perceived info,
        so we treat pseudo labels as the target.
        
        Args:
            predictions: Output from compute_forward with pseudo labels
            batch: Batch object
            pseudo_labels: Pseudo labels from teacher [B, max_len]
            pseudo_label_lens: Pseudo label lengths [B]
            
        Returns:
            loss: CTC loss on pseudo labels
        """
        # Parse predictions based on type (same as compute_objectives)
        if len(predictions) == 2:
            p_ctc, wav_lens = predictions
            extras = {}
        elif len(predictions) == 4:
            if isinstance(predictions[3], bool):  # CR-CTC
                p_ctc, wav_lens, time_mask, is_crctc_mode = predictions
                extras = {'time_mask': time_mask, 'is_crctc_mode': is_crctc_mode}
            else:  # RVQ
                p_ctc, wav_lens, commitment_loss, codebook_loss = predictions
                extras = {'commitment_loss': commitment_loss, 'codebook_loss': codebook_loss}
        elif len(predictions) == 5:  # OTTC or CR-OTTC with pseudo labels
            p_ctc, logits, weights_logits, weights_labels, wav_lens = predictions
            extras = {'logits': logits, 'weights_logits': weights_logits, 'weights_labels': weights_labels}
            if self.loss_manager.loss_type == 'crottc':
                extras['is_crottc_mode'] = True
        else:
            raise ValueError(f"Unexpected predictions format: {len(predictions)} elements")
        
        # Use CTCLossManager to compute loss（支持所有 loss type）
        loss, loss_dict = self.loss_manager.compute_loss(
            p_ctc, pseudo_labels, wav_lens, pseudo_label_lens, 
            stage=sb.Stage.TRAIN, 
            extras=extras
        )
        
        # Add RVQ losses if present
        if 'commitment_loss' in extras:
            loss = loss + extras['commitment_loss'] + extras['codebook_loss']
            loss_dict['commitment_loss'] = extras['commitment_loss'].detach()
            loss_dict['codebook_loss'] = extras['codebook_loss'].detach()
        
        return loss
    
    def fit_batch(self, batch):
        """
        Fit one batch - handles both labeled and unlabeled data.
        
        For labeled batches (has phn_encoded_target):
            - Standard supervised training
            
        For unlabeled batches (no phn_encoded_target):
            - Generate pseudo labels with teacher
            - Train student on pseudo labels
            - Update teacher with EMA
        """
        is_unlabeled = not hasattr(batch, 'phn_encoded_target')
        
        if self.hparams.auto_mix_prec:
            return self._fit_batch_amp(batch, is_unlabeled)
        else:
            return self._fit_batch_fp32(batch, is_unlabeled)
    
    def _fit_batch_amp(self, batch, is_unlabeled):
        """Mixed precision training"""
        self.pretrained_opt_class.zero_grad()
        self.adam_optimizer.zero_grad()
        
        with torch.amp.autocast("cuda"):
            if is_unlabeled:
                # Generate pseudo labels with teacher
                pls, pl_lens = self.infer_batch_teacher(batch)
                
                # Enable training mode and augmentation for student
                self.modules.train()
                if hasattr(self.hparams, "augmentation"):
                    self.modules.perceived_ssl.model.config.apply_spec_augment = True
                
                # Forward pass with pseudo labels（传递伪标签给 compute_forward）
                outputs = self.compute_forward(batch, sb.Stage.TRAIN, pseudo_labels=pls, pseudo_label_lens=pl_lens)
                
                # 用 CTCLossManager 计算 loss（支持所有 loss type）
                loss = self.compute_objectives_unlabeled(outputs, batch, pls, pl_lens)
                
                # Track unlabeled loss
                self.unlabeled_loss_sum += loss.item()
                self.unlabeled_loss_count += 1
            else:
                # Standard supervised training
                self.modules.train()
                if hasattr(self.hparams, "augmentation"):
                    self.modules.perceived_ssl.model.config.apply_spec_augment = True
                
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
                
                # Track labeled loss
                self.labeled_loss_sum += loss.item()
                self.labeled_loss_count += 1
        
        self.scaler.scale(loss / self.hparams.gradient_accumulation).backward()
        self.scaler.unscale_(self.pretrained_opt_class)
        self.scaler.unscale_(self.adam_optimizer)
        
        if self.check_gradients(loss):
            if any(p.requires_grad for p in self.pretrained_opt_class.param_groups[0]['params']):
                self.scaler.step(self.pretrained_opt_class)
            if any(p.requires_grad for p in self.adam_optimizer.param_groups[0]['params']):
                self.scaler.step(self.adam_optimizer)
        
        self.scaler.update()
        
        # Momentum update teacher
        if self.step % self.hparams.gradient_accumulation == 0:
            self.teacher_momentum_update()
        
        return loss.detach().cpu()
    
    def _fit_batch_fp32(self, batch, is_unlabeled):
        """Standard FP32 training"""
        if is_unlabeled:
            # Generate pseudo labels with teacher (eval mode, no augmentation)
            pls, pl_lens = self.infer_batch_teacher(batch)
            
            # Enable training mode and augmentation for student
            self.modules.train()
            if hasattr(self.hparams, "augmentation"):
                self.modules.perceived_ssl.model.config.apply_spec_augment = True
            
            # Forward pass with pseudo labels（传递伪标签给 compute_forward）
            outputs = self.compute_forward(batch, sb.Stage.TRAIN, pseudo_labels=pls, pseudo_label_lens=pl_lens)
            
            # 用 CTCLossManager 计算 loss（支持所有 loss type）
            loss = self.compute_objectives_unlabeled(outputs, batch, pls, pl_lens)
            
            # Track unlabeled loss
            self.unlabeled_loss_sum += loss.item()
            self.unlabeled_loss_count += 1
        else:
            # Standard supervised training
            self.modules.train()
            if hasattr(self.hparams, "augmentation"):
                self.modules.perceived_ssl.model.config.apply_spec_augment = True
            
            outputs = self.compute_forward(batch, sb.Stage.TRAIN)
            loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            
            # Track labeled loss
            self.labeled_loss_sum += loss.item()
            self.labeled_loss_count += 1
        
        # Backward pass
        (loss / self.hparams.gradient_accumulation).backward()
        
        if self.step % self.hparams.gradient_accumulation == 0:
            if self.check_gradients(loss):
                self.pretrained_opt_class.step()
                self.adam_optimizer.step()
            
            self.pretrained_opt_class.zero_grad()
            self.adam_optimizer.zero_grad()
            
            # Momentum update teacher
            self.teacher_momentum_update()
        
        return loss.detach().cpu()
    
    def on_fit_start(self):
        """Initialize optimizers and teacher model"""
        super().on_fit_start()
        
        # Initialize teacher model
        self.init_teacher_model()
        
        # Set start epoch for MPL (continue from pre-trained)
        mpl_start_epoch = getattr(self.hparams, 'mpl_start_epoch', None)
        if mpl_start_epoch is not None:
            self.hparams.epoch_counter.current = mpl_start_epoch
            logger.info(f"📊 MPL training starts from epoch {mpl_start_epoch}")
    
    def on_stage_start(self, stage, epoch):
        """Reset loss tracking at epoch start"""
        super().on_stage_start(stage, epoch)
        
        if stage == sb.Stage.TRAIN:
            self.labeled_loss_sum = 0.0
            self.labeled_loss_count = 0
            self.unlabeled_loss_sum = 0.0
            self.unlabeled_loss_count = 0
    
    def on_stage_end(self, stage, stage_loss, epoch):
        """Log MPL-specific metrics"""
        if stage == sb.Stage.TRAIN:
            # Compute average losses
            avg_labeled_loss = (self.labeled_loss_sum / max(1, self.labeled_loss_count)
                               if self.labeled_loss_count > 0 else 0.0)
            avg_unlabeled_loss = (self.unlabeled_loss_sum / max(1, self.unlabeled_loss_count)
                                 if self.unlabeled_loss_count > 0 else 0.0)
            
            self.avg_labeled_loss = avg_labeled_loss
            self.avg_unlabeled_loss = avg_unlabeled_loss
            
            logger.info(f"📊 Epoch {epoch} Training:")
            logger.info(f"   Labeled loss: {avg_labeled_loss:.4f} ({self.labeled_loss_count} batches)")
            logger.info(f"   Unlabeled loss: {avg_unlabeled_loss:.4f} ({self.unlabeled_loss_count} batches)")
        
        # Call parent's on_stage_end
        super().on_stage_end(stage, stage_loss, epoch)
        
        # Add MPL-specific wandb logging
        if stage == sb.Stage.VALID:
            import wandb
            wandb.log({
                "train_labeled_loss": getattr(self, 'avg_labeled_loss', 0.0),
                "train_unlabeled_loss": getattr(self, 'avg_unlabeled_loss', 0.0),
                "momentum_factor": self.momentum_factor,
            }, step=epoch)
    
    def fit_mpl(
        self,
        epoch_counter,
        train_data_l,
        train_data_u,
        valid_set=None,
        progressbar=None,
        train_loader_kwargs_l={},
        train_loader_kwargs_u={},
        valid_loader_kwargs={},
    ):
        """
        MPL training loop with mixed labeled and unlabeled data.
        
        Args:
            epoch_counter: Iterable returning epoch numbers
            train_data_l: Labeled training DataLoader
            train_data_u: Unlabeled training DataLoader
            valid_set: Validation DataLoader
            progressbar: Whether to show progress bar
            train_loader_kwargs_l: Kwargs for labeled data loader
            train_loader_kwargs_u: Kwargs for unlabeled data loader
            valid_loader_kwargs: Kwargs for validation data loader
        """
        # Calculate total training batches
        self.n_train_batch = len(train_data_l) + len(train_data_u)
        
        # Run fit start (initializes optimizers, teacher, etc.)
        self.on_fit_start()
        
        # Calculate momentum factor based on training schedule
        remaining_epochs = epoch_counter.limit - epoch_counter.current
        self.set_momentum_factor(self.n_train_batch, remaining_epochs)
        
        if progressbar is None:
            progressbar = not self.noprogressbar
        
        # Training loop
        for epoch in epoch_counter:
            # Chain labeled and unlabeled data loaders
            train_set = itertools.chain(train_data_l, train_data_u)
            
            # Training stage
            self.on_stage_start(Stage.TRAIN, epoch)
            self.modules.train()
            
            # Reset nonfinite count
            self.nonfinite_count = 0
            
            if self.train_sampler is not None and hasattr(self.train_sampler, "set_epoch"):
                self.train_sampler.set_epoch(epoch)
            
            # Time tracking for intra-epoch checkpoints
            last_ckpt_time = time.time()
            
            # Progress bar
            enable = progressbar and sb.utils.distributed.if_main_process()
            
            with tqdm(
                train_set,
                initial=self.step,
                total=self.n_train_batch,
                dynamic_ncols=True,
                disable=not enable,
                desc=f"Epoch {epoch}"
            ) as t:
                for batch in t:
                    self.step += 1
                    loss = self.fit_batch(batch)
                    self.avg_train_loss = self.update_average(loss, self.avg_train_loss)
                    
                    # Update progress bar
                    t.set_postfix(
                        train_loss=f"{self.avg_train_loss:.4f}",
                        L=f"{getattr(self, 'avg_labeled_loss', 0):.3f}" if hasattr(self, 'avg_labeled_loss') else "N/A",
                        U=f"{getattr(self, 'avg_unlabeled_loss', 0):.3f}" if hasattr(self, 'avg_unlabeled_loss') else "N/A"
                    )
                    
                    # Debug mode
                    if self.debug and self.step == self.debug_batches:
                        break
                    
                    # Intra-epoch checkpoint
                    if (self.checkpointer is not None and 
                        self.ckpt_interval_minutes > 0 and
                        time.time() - last_ckpt_time >= self.ckpt_interval_minutes * 60.0):
                        if sb.utils.distributed.if_main_process():
                            self._save_intra_epoch_ckpt()
                        last_ckpt_time = time.time()
            
            # End of training epoch
            self.on_stage_end(Stage.TRAIN, self.avg_train_loss, epoch)
            self.avg_train_loss = 0.0
            self.step = 0
            
            # Validation stage
            if valid_set is not None:
                self.on_stage_start(Stage.VALID, epoch)
                self.modules.eval()
                avg_valid_loss = 0.0
                
                with torch.no_grad():
                    for batch in tqdm(valid_set, dynamic_ncols=True, disable=not enable, desc="Validation"):
                        self.step += 1
                        loss = self.evaluate_batch(batch, stage=Stage.VALID)
                        avg_valid_loss = self.update_average(loss, avg_valid_loss)
                        
                        if self.debug and self.step == self.debug_batches:
                            break
                    
                    self.step = 0
                    run_on_main(
                        self.on_stage_end,
                        args=[Stage.VALID, avg_valid_loss, epoch],
                    )
            
            # Debug mode - only few epochs
            if self.debug and epoch == self.debug_epochs:
                break
        
        logger.info("✅ MPL Training completed!")
        return self


class PhnMonoSSLModel_MPL_Interleaved(PhnMonoSSLModel_MPL):
    """
    Interleaved MPL variant that alternates between labeled and unlabeled batches.
    
    Instead of chaining all labeled then unlabeled, this version interleaves them
    for potentially better training dynamics.
    """
    
    def fit_mpl(
        self,
        epoch_counter,
        train_data_l,
        train_data_u,
        valid_set=None,
        progressbar=None,
        **kwargs
    ):
        """
        Interleaved MPL training with alternating labeled/unlabeled batches.
        """
        self.n_train_batch = len(train_data_l) + len(train_data_u)
        
        self.on_fit_start()
        
        remaining_epochs = epoch_counter.limit - epoch_counter.current
        self.set_momentum_factor(self.n_train_batch, remaining_epochs)
        
        if progressbar is None:
            progressbar = not self.noprogressbar
        
        for epoch in epoch_counter:
            # Create iterators
            iter_l = iter(train_data_l)
            iter_u = iter(train_data_u)
            
            # Interleave labeled and unlabeled
            train_set = self._interleave_dataloaders(iter_l, iter_u)
            
            self.on_stage_start(Stage.TRAIN, epoch)
            self.modules.train()
            self.nonfinite_count = 0
            
            enable = progressbar and sb.utils.distributed.if_main_process()
            
            with tqdm(
                train_set,
                initial=self.step,
                total=self.n_train_batch,
                dynamic_ncols=True,
                disable=not enable,
                desc=f"Epoch {epoch}"
            ) as t:
                for batch in t:
                    self.step += 1
                    loss = self.fit_batch(batch)
                    self.avg_train_loss = self.update_average(loss, self.avg_train_loss)
                    t.set_postfix(train_loss=f"{self.avg_train_loss:.4f}")
                    
                    if self.debug and self.step == self.debug_batches:
                        break
            
            self.on_stage_end(Stage.TRAIN, self.avg_train_loss, epoch)
            self.avg_train_loss = 0.0
            self.step = 0
            
            # Validation
            if valid_set is not None:
                self.on_stage_start(Stage.VALID, epoch)
                self.modules.eval()
                avg_valid_loss = 0.0
                
                with torch.no_grad():
                    for batch in tqdm(valid_set, dynamic_ncols=True, disable=not enable):
                        self.step += 1
                        loss = self.evaluate_batch(batch, stage=Stage.VALID)
                        avg_valid_loss = self.update_average(loss, avg_valid_loss)
                    
                    self.step = 0
                    run_on_main(self.on_stage_end, args=[Stage.VALID, avg_valid_loss, epoch])
        
        return self
    
    def _interleave_dataloaders(self, iter_l, iter_u):
        """Interleave two iterators, yielding from both alternately"""
        while True:
            try:
                yield next(iter_l)
            except StopIteration:
                iter_l = None
            
            try:
                yield next(iter_u)
            except StopIteration:
                iter_u = None
            
            if iter_l is None and iter_u is None:
                break
            elif iter_l is None:
                # Exhaust remaining unlabeled
                while True:
                    try:
                        yield next(iter_u)
                    except StopIteration:
                        break
                break
            elif iter_u is None:
                # Exhaust remaining labeled
                while True:
                    try:
                        yield next(iter_l)
                    except StopIteration:
                        break
                break


# ============================================================================
# Helper Functions for MPL Training Setup
# ============================================================================

def create_teacher_modules(hparams, device):
    """
    Create teacher module dict from hparams.
    
    Usage in yaml:
        modules_teacher:
            perceived_ssl: !ref <perceived_ssl>
            enc: !ref <enc>
            ctc_lin: !ref <ctc_lin>
            ConformerEncoder: !ref <ConformerEncoder>  # optional
    """
    modules_teacher = {}
    
    # SSL model
    if hasattr(hparams, 'perceived_ssl_teacher'):
        modules_teacher['perceived_ssl'] = hparams.perceived_ssl_teacher
    elif hasattr(hparams, 'perceived_ssl'):
        modules_teacher['perceived_ssl'] = copy.deepcopy(hparams.perceived_ssl)
    
    # Encoder
    if hasattr(hparams, 'enc_teacher'):
        modules_teacher['enc'] = hparams.enc_teacher
    elif hasattr(hparams, 'enc'):
        modules_teacher['enc'] = copy.deepcopy(hparams.enc)
    
    # CTC head
    if hasattr(hparams, 'ctc_lin_teacher'):
        modules_teacher['ctc_lin'] = hparams.ctc_lin_teacher
    elif hasattr(hparams, 'ctc_lin'):
        modules_teacher['ctc_lin'] = copy.deepcopy(hparams.ctc_lin)
    
    # Optional: Conformer
    if hasattr(hparams, 'ConformerEncoder_teacher'):
        modules_teacher['ConformerEncoder'] = hparams.ConformerEncoder_teacher
    elif hasattr(hparams, 'ConformerEncoder'):
        modules_teacher['ConformerEncoder'] = copy.deepcopy(hparams.ConformerEncoder)
    
    return nn.ModuleDict(modules_teacher).to(device)


def setup_mpl_dataloaders(hparams, train_data_l, train_data_u, valid_data, test_data):
    """
    Setup data loaders for MPL training.
    
    Args:
        hparams: Hyperparameters dict
        train_data_l: Labeled training dataset
        train_data_u: Unlabeled training dataset
        valid_data: Validation dataset
        test_data: Test dataset
        
    Returns:
        Tuple of DataLoaders
    """
    from torch.utils.data import DataLoader
    from speechbrain.dataio.batch import PaddedBatch
    
    batch_size_l = hparams.get('batch_size_labeled', hparams.get('batch_size', 8))
    batch_size_u = hparams.get('batch_size_unlabeled', batch_size_l)
    num_workers = hparams.get('num_workers', 4)
    
    train_loader_l = DataLoader(
        train_data_l,
        batch_size=batch_size_l,
        drop_last=False,
        shuffle=True,
        collate_fn=PaddedBatch,
        num_workers=num_workers
    )
    
    train_loader_u = DataLoader(
        train_data_u,
        batch_size=batch_size_u,
        drop_last=False,
        shuffle=True,
        collate_fn=PaddedBatch,
        num_workers=num_workers
    )
    
    valid_loader = DataLoader(
        valid_data,
        batch_size=batch_size_l,
        drop_last=False,
        shuffle=False,
        collate_fn=PaddedBatch,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_data,
        batch_size=1,
        drop_last=False,
        shuffle=False,
        collate_fn=PaddedBatch,
        num_workers=1
    )
    
    return train_loader_l, train_loader_u, valid_loader, test_loader
