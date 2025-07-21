import os
import sys
import torch
import logging
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from mpd_eval_v3 import MpdStats
import librosa
import json
import wandb
import time
import torchaudio
from speechbrain.inference.text import GraphemeToPhoneme

import re
            
# Define training procedure
class PhnDualSSLModel(sb.Brain):
    def __init__(self, *args, patience=30, **kwargs):
        super().__init__(*args, **kwargs)
        self.patience = patience
        self.no_improve_epochs = 0
        self.best_per_list = []  # List of (PER, epoch, ckpt_name)
        self.best_mpd_f1_list = []  # List of (mpd_f1, epoch, ckpt_name)
        self.best_per = float('inf')
        self.best_mpd_f1 = float('-inf')
        self.last_improved_epoch = 0
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.best_per_cano = float("inf")
        self.best_per_cano_list = []
        if self.modules.perceived_ssl is not None:
            self.modules.perceived_ssl.to(self.device)

        if self.modules.enc is not None:
            self.modules.enc.to(self.device)
        if self.modules.ctc_lin is not None:
            self.modules.ctc_lin.to(self.device)
        if self.modules.canonical_ssl is not None:
            self.modules.canonical_ssl.to(self.device)
        if self.modules.enc_can is not None:
            self.modules.enc_can.to(self.device)
        if self.modules.ctc_lin_can is not None:
            self.modules.ctc_lin_can.to(self.device)

    def check_gradients(self, loss):
        """Check if gradients are finite"""
        if not torch.isfinite(loss):
            print("Warning: loss is not finite, skipping step")
            return False
        return True
    
    def compute_forward(self, batch, stage):
        "Given an input batch it computes the phoneme probabilities."
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        # phns_bos, _ = batch.phn_encoded_bos

        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs)
        feats_per = self.modules.perceived_ssl(wavs)
        feats_can = self.modules.canonical_ssl(wavs)
        
        x_per = self.modules.enc(feats_per)
        x_can = self.modules.enc_can(feats_can)

        # output layer for ctc log-probabilities
        logits_per = self.modules.ctc_lin(x_per)
        logits_can = self.modules.ctc_lin_can(x_can)
        
        p_ctc_can = self.hparams.log_softmax(logits_can)
        p_ctc_per = self.hparams.log_softmax(logits_per)
        
        # residual 
        
        
        # 这里也可以改一下

        p_ctc = self.hparams.blend_alpha * p_ctc_can + (1 - self.hparams.blend_alpha) * p_ctc_per
        
        return p_ctc, p_ctc_can, p_ctc_per, wav_lens

    def compute_objectives(self, predictions, batch, stage):
        "Given the network predictions and targets computed the NLL loss."

        p_ctc, p_ctc_can, p_ctc_per, wav_lens = predictions

        ids = batch.id
        targets, target_lens = batch.phn_encoded_target
        canonicals, canonical_lens = batch.phn_encoded_canonical 
        perceiveds, perceived_lens = batch.phn_encoded_perceived 
        # len(perceiveds) == len(canonicals) != len(targets) 
        
        # if stage != sb.Stage.TRAIN:
        #     canonicals, canonical_lens = batch.phn_encoded_canonical

        # loss_ctc = self.hparams.ctc_cost(p_ctc, targets, wav_lens, target_lens)
        loss_ctc_can = self.hparams.ctc_cost(p_ctc_can, canonicals, wav_lens, canonical_lens)
        loss_ctc_per_with_silence = self.hparams.ctc_cost(p_ctc_per, perceiveds, wav_lens, perceived_lens)
        loss_ctc_per = self.hparams.ctc_cost(p_ctc_per, targets, wav_lens, target_lens)
        
        loss = self.hparams.blend_alpha * loss_ctc_can + (1 - self.hparams.blend_alpha) * loss_ctc_per
        # Log both CTC losses to wandb

        # Record losses for posterity
        if stage != sb.Stage.TRAIN:
            # Note: sb.decoders.ctc_greedy_decode will also remove padded tokens
            # that is, it return a list of list with different lengths
            sequence_per = sb.decoders.ctc_greedy_decode(
                p_ctc_per, wav_lens, blank_id=self.hparams.blank_index
            )
            sequence_can = sb.decoders.ctc_greedy_decode(
                p_ctc_can, canonical_lens, blank_id=self.hparams.blank_index
            )
            # self.ctc_metrics.append(ids, p_ctc, targets, wav_lens, target_lens)
            self.ctc_metrics_can.append(ids, p_ctc_can, canonicals, wav_lens, canonical_lens)
            self.ctc_metrics_per.append(ids, p_ctc_per, targets, wav_lens, target_lens)

            self.per_metrics_per.append(
                ids=ids,
                predict=sequence_per,
                target=targets,
                predict_len=None,
                target_len=target_lens,
                ind2lab=self.label_encoder.decode_ndim,
            )
            self.per_metrics_can.append(
                ids=ids,
                predict=sequence_can,
                target=canonicals,
                predict_len=None,
                target_len=canonical_lens,
                ind2lab=self.label_encoder.decode_ndim,
            )
            
            self.mpd_metrics.append(
                ids=ids,
                predict=sequence_per,
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

    def on_stage_start(self, stage, epoch):
        "Gets called when a stage (either training, validation, test) starts."
        self.ctc_metrics_per = self.hparams.ctc_stats()
        self.ctc_metrics_can = self.hparams.ctc_stats_can()
        
        if hasattr(self.hparams, "augmentation"):
            self.modules.perceived_ssl.model.config.apply_spec_augment = True
            self.modules.canonical_ssl.model.config.apply_spec_augment = True
        
        if stage != sb.Stage.TRAIN:
            self.per_metrics_per = self.hparams.per_stats()
            self.per_metrics_can = self.hparams.per_stats_can()
            self.mpd_metrics = MpdStats()


    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
            
        else:
            per_can = self.per_metrics_can.summarize("error_rate")
            per_per = self.per_metrics_per.summarize("error_rate")
            mpd_f1 = self.mpd_metrics.summarize("mpd_f1")

        if stage == sb.Stage.VALID:

            self.hparams.train_logger.log_stats(
                stats_meta={
                    "epoch": epoch,
                    "lr_adam": self.adam_optimizer.param_groups[0]["lr"],
                    "lr_pretrained": self.pretrained_opt_class.param_groups[0]["lr"],
                    "lr_mispro": self.warm_up_opt_class.param_groups[0]["lr"],
                },
                train_stats={"loss": self.train_loss},
                valid_stats={
                    "loss": stage_loss,
                    "ctc_loss_can": self.ctc_metrics_can.summarize("average"),
                    "ctc_loss_per": self.ctc_metrics_per.summarize("average"),
                    "PER_can": per_can,
                    "PER_per": per_per,
                    "mpd_f1": mpd_f1
                },
            )
       # Save best 3 Models
            improved = False
            # Save best 3 PER_per models (lower is better)
            if per_per < self.best_per or len(self.best_per_list) < 3:
                ckpt_name = f"best_per_{epoch:03d}_{per_per:.4f}.ckpt"
                self.checkpointer.save_and_keep_only(
                    meta={"PER_per": per_per, "mpd_f1": mpd_f1, "epoch": epoch},
                    name=ckpt_name,
                    num_to_keep=3,
                    min_keys=["PER_per"]
                )
                self.best_per_list.append((per_per, epoch, ckpt_name))
                self.best_per_list = sorted(self.best_per_list, key=lambda x: x[0])[:3]
                self.best_per = self.best_per_list[0][0]
                improved = True
                # Remove extra checkpoints
                if len(self.best_per_list) > 3:
                    to_remove = self.best_per_list[3:]
                    for _, _, name in to_remove:
                        self.checkpointer.delete_checkpoint(name)
                    self.best_per_list = self.best_per_list[:3]
            # Save best 3 mpd_f1 models (higher is better)
            if per_can < self.best_per_cano or len(self.best_per_cano_list) < 3:
                ckpt_name = f"best_per_cano_{epoch:03d}_{per_can:.4f}.ckpt"
                self.checkpointer.save_and_keep_only(
                    meta={"PER_can": per_can, "mpd_f1": mpd_f1, "epoch": epoch},
                    name=ckpt_name,
                    num_to_keep=3,
                    min_keys=["PER_can"]
                )
                self.best_per_cano_list.append((per_can, epoch, ckpt_name))
                self.best_per_cano_list = sorted(self.best_per_cano_list, key=lambda x: x[0])[:3]
                self.best_per_cano = self.best_per_cano_list[0][0]
                improved = True
                # Remove extra checkpoints
                if len(self.best_per_cano_list) > 3:
                    to_remove = self.best_per_cano_list[3:]
                    for _, _, name in to_remove:
                        self.checkpointer.delete_checkpoint(name)
                    self.best_per_cano_list = self.best_per_cano_list[:3]
            if mpd_f1 > self.best_mpd_f1 or len(self.best_mpd_f1_list) < 3:
                ckpt_name = f"best_mpdf1_{epoch:03d}_{mpd_f1:.4f}.ckpt"
                self.checkpointer.save_and_keep_only(
                    meta={"PER_can": per_can, "PER_per": per_per, "mpd_f1": mpd_f1, "epoch": epoch},
                    name=ckpt_name,
                    num_to_keep=3,
                    max_keys=["mpd_f1"]
                )
                self.best_mpd_f1_list.append((mpd_f1, epoch, ckpt_name))
                self.best_mpd_f1_list = sorted(self.best_mpd_f1_list, key=lambda x: -x[0])[:3]
                self.best_mpd_f1 = self.best_mpd_f1_list[0][0]
                improved = True
                # Remove extra checkpoints
                if len(self.best_mpd_f1_list) > 3:
                    to_remove = self.best_mpd_f1_list[3:]
                    for _, _, name in to_remove:
                        self.checkpointer.delete_checkpoint(name)
                    self.best_mpd_f1_list = self.best_mpd_f1_list[:3]
            # Early stopping logic
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
                "ctc_loss_per": self.ctc_metrics_per.summarize("average"),
                "ctc_loss_can": self.ctc_metrics_can.summarize("average"),
                "PER_per": per_per,
                "PER_can": per_can,
                "mpd_f1": mpd_f1,
                
            }, step=epoch)
            # Early stop if patience exceeded
            if self.no_improve_epochs >= self.patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {self.patience} epochs)")
                raise StopIteration

            # # Save best model based on MPD-F1 (higher is better)
            # # We'll use a separate checkpoint name to avoid conflicts
            # self.checkpointer.save_checkpoint(
            #     meta={"PER": per, "mpd_f1": mpd_f1, "epoch": epoch},
            #     name="best_mpd_f1_{}.ckpt".format(epoch),
            # )

        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats={"loss": stage_loss, "PER": per_per, "mpd_f1": mpd_f1},
            )
            with open(self.hparams.per_file, "w") as w:
                w.write("CTC loss stats on perceived_phonemes :\n")
                self.ctc_metrics_per.write_stats(w)
                w.write("\nPER on stats:\n")
                self.per_metrics_per.write_stats(w)
                print(
                    "CTC and PER stats on perceived phonemes written to file",
                    self.hparams.per_file,
                )
                
            with open(self.hparams.per_can_file, "w") as p:
                p.write("CTC loss stats on canonical phonemes:\n")
                self.ctc_metrics_can.write_stats(p)
                p.write("PER on canonical phonemes:\n")
                self.per_metrics_can.write_stats(p)
                print(
                    "PER on canonical phonemes stats written to file",
                    self.hparams.per_can_file,
                )
                
            with open(self.hparams.mpd_file, "w") as m:
                m.write("MPD results and stats:\n")
                self.mpd_metrics.write_stats(m)
                print(
                    "MPD results and stats written to file",
                    self.hparams.mpd_file,
                )

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
        # import pdb; pdb.set_trace()
        if self.hparams.auto_mix_prec:
            self.pretrained_opt_class.zero_grad() # perceived_ssl
            self.pretrained_opt_cano_class.zero_grad() # canonical_ssl
            self.adam_optimizer.zero_grad()
            self.adam_optimizer_cano.zero_grad()

            with torch.amp.autocast("cuda"):
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)

            # normalize the loss by gradient_accumulation and scale for mixed precision
            self.scaler.scale(loss / self.hparams.gradient_accumulation).backward()
            self.scaler.unscale_(self.pretrained_opt_class)
            self.scaler.unscale_(self.pretrained_opt_cano_class)
            self.scaler.unscale_(self.adam_optimizer)

            if self.check_gradients(loss):
                self.scaler.step(self.pretrained_opt_class)
                self.scaler.step(self.pretrained_opt_cano_class)
                self.scaler.step(self.adam_optimizer)
                self.scaler.step(self.adam_optimizer_cano)

            self.scaler.update()

        else:
            outputs = self.compute_forward(batch, sb.Stage.TRAIN)
            
            loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            # normalize the loss by gradient_accumulation step
            (loss / self.hparams.gradient_accumulation).backward()

            if self.step % self.hparams.gradient_accumulation == 0:
                # gradient clipping & early stop if loss is not fini
                
                if self.check_gradients(loss):
                    self.pretrained_opt_class.step()
                    self.pretrained_opt_cano_class.step()
                    self.adam_optimizer.step()
                    self.adam_optimizer_cano.step()

                self.pretrained_opt_class.zero_grad()
                self.pretrained_opt_cano_class.zero_grad()
                self.adam_optimizer.zero_grad()
                self.adam_optimizer_cano.zero_grad()

        return loss.detach().cpu()

    def init_optimizers(self):
        self.adam_optimizer = self.hparams.adam_opt_class(
            self.hparams.model.parameters(), 
        )
        self.adam_optimizer_cano = self.hparams.adam_opt_class_cano(
            self.hparams.model_cano.parameters(), 
        )
        self.pretrained_opt_class = self.hparams.pretrained_opt_class(
            self.modules.perceived_ssl.parameters(), 
        )
        self.pretrained_opt_cano_class = self.hparams.pretrained_opt_cano_class(
            self.modules.canonical_ssl.parameters(), 
        )
        self.warm_up_opt_class = self.hparams.warm_up_opt_class(
            self.modules.residual_lin.parameters(),
        )

        if self.checkpointer is not None:
            # if self.hparams.perceived_ssl is not None and not self.hparams.perceived_ssl.freeze:
            self.checkpointer.add_recoverable("adam_opt", self.adam_optimizer)
            self.checkpointer.add_recoverable("adam_opt_cano", self.adam_optimizer_cano)
            self.checkpointer.add_recoverable("pretrained_opt", self.pretrained_opt_class)
            self.checkpointer.add_recoverable("pretrained_opt_cano", self.pretrained_opt_cano_class)
            self.checkpointer.add_recoverable("warm_up_opt", self.warm_up_opt_class)

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
            self.checkpointer.recover_if_possible(
                min_key="PER"
            )
            
class PhnDualSSLModel_with_SimpleResidual(PhnDualSSLModel):
    def __init__(self, *args, patience=30, **kwargs):
        super().__init__(*args, patience=patience, **kwargs)
        if self.modules.residual_lin is not None:
            self.modules.residual_lin.to(self.device)

    def compute_forward(self, batch, stage):
        "Given an input batch it computes the phoneme probabilities."
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        # phns_bos, _ = batch.phn_encoded_bos

        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs)
        feats_per = self.modules.perceived_ssl(wavs)
        feats_can = self.modules.canonical_ssl(wavs)
        
        x_per = self.modules.enc(feats_per)
        x_can = self.modules.enc_can(feats_can)

        # output layer for ctc log-probabilities
        logits_per = self.modules.ctc_lin(x_per)
        logits_can = self.modules.ctc_lin_can(x_can)
        
        p_ctc_can = self.hparams.log_softmax(logits_can)
        p_ctc_per = self.hparams.log_softmax(logits_per)
        
        # residual 
        # x_can, x_per: [B, T, D]  —— encoder 输出

        delta = x_per - x_can     # [B, T, D]
        logits_diag = self.modules.residual_lin(delta)
        p_diag = self.hparams.log_softmax(logits_diag)  # [B, T, 2]
        # 这里也可以改一下
        p_ctc = self.hparams.blend_alpha * p_ctc_can + (1 - self.hparams.blend_alpha) * p_ctc_per

        return p_ctc, p_ctc_can, p_ctc_per, p_diag, wav_lens
    
    def compute_objectives(self, predictions, batch, stage):
        "Given the network predictions and targets computed the NLL loss."

        p_ctc, p_ctc_can, p_ctc_per, p_diag, wav_lens = predictions

        ids = batch.id
        targets, target_lens = batch.phn_encoded_target
        canonicals, canonical_lens = batch.phn_encoded_canonical 
        perceiveds, perceived_lens = batch.phn_encoded_perceived 
        # len(perceiveds) == len(canonicals) != len(targets) 
        
        # if stage != sb.Stage.TRAIN:
        #     canonicals, canonical_lens = batch.phn_encoded_canonical

        # loss_ctc = self.hparams.ctc_cost(p_ctc, targets, wav_lens, target_lens)
        loss_ctc_can = self.hparams.ctc_cost(p_ctc_can, canonicals, wav_lens, canonical_lens)
        loss_ctc_per_with_silence = self.hparams.ctc_cost(p_ctc_per, perceiveds, wav_lens, perceived_lens)
        loss_ctc_per = self.hparams.ctc_cost(p_ctc_per, targets, wav_lens, target_lens)
        
        # canonical
        try:
            mispronunciation_label = torch.Tensor(canonicals != perceiveds).long()
        except:
            mispronunciation_label = torch.Tensor(canonicals[:,: -1] != perceiveds).long()
        # [true=1, false=-1, 0=ignore]
        loss_ctc_mispro = self.hparams.ctc_cost(p_diag, mispronunciation_label, wav_lens, canonical_lens) # [B, 2]

        loss = self.hparams.blend_alpha * loss_ctc_can + (1 - self.hparams.blend_alpha) * loss_ctc_per \
               + self.hparams.mispronunciation_weight * loss_ctc_mispro
        # Log both CTC losses to wandb

        # Record losses for posterity
        if stage != sb.Stage.TRAIN:
            # Note: sb.decoders.ctc_greedy_decode will also remove padded tokens
            # that is, it return a list of list with different lengths
            sequence_per = sb.decoders.ctc_greedy_decode(
                p_ctc_per, wav_lens, blank_id=self.hparams.blank_index
            )
            sequence_can = sb.decoders.ctc_greedy_decode(
                p_ctc_can, canonical_lens, blank_id=self.hparams.blank_index
            )
            sequeuce_diag = sb.decoders.ctc_greedy_decode(
                p_diag, canonical_lens, blank_id=-1
            )
            # self.ctc_metrics.append(ids, p_ctc, targets, wav_lens, target_lens)
            self.ctc_metrics_can.append(ids, p_ctc_can, canonicals, wav_lens, canonical_lens)
            self.ctc_metrics_per.append(ids, p_ctc_per, targets, wav_lens, target_lens)
            self.ctc_metrics_diag.append(ids, p_diag, perceiveds, wav_lens, perceived_lens)

            # print(f"sequence_per: {sequence_per}")
            predict_canonicals = [self.label_encoder.decode_ndim(seq) for seq in sequence_can]
            predict_perceiveds = [self.label_encoder.decode_ndim(seq) for seq in sequence_per]
            predict_diag_labels = [seq for seq in sequeuce_diag]
            # gt canonicals
            gt_canonicals = [self.label_encoder.decode_ndim(seq) for seq in canonicals]
            # gt perceiveds
            gt_perceiveds = [self.label_encoder.decode_ndim(seq) for seq in perceiveds]
            # gt mispronunciation labels
            gt_mispronunciation_label = [seq for seq in mispronunciation_label]
            
            print(f"perceiveds: {gt_perceiveds[0]}")
            print(f"predict_perceiveds: {predict_perceiveds[0]}")
            print(f"canonicals: {gt_canonicals[0]}")
            print(f"predict_canonicals: {predict_canonicals[0]}")
            print(f"mispronunciation_label: {gt_mispronunciation_label[0]}")
            print(f"predict_diag_labels: {predict_diag_labels[0]}")
            
            self.per_metrics_per.append(
                ids=ids,
                predict=sequence_per,
                target=targets,
                predict_len=None,
                target_len=target_lens,
                ind2lab=self.label_encoder.decode_ndim,
            )
            self.per_metrics_can.append(
                ids=ids,
                predict=sequence_can,
                target=canonicals,
                predict_len=None,
                target_len=canonical_lens,
                ind2lab=self.label_encoder.decode_ndim,
            )
            
            self.per_metrics_diag.append(
                ids=ids,
                predict=sequeuce_diag,
                target=mispronunciation_label,
                predict_len=None,
                target_len=canonical_lens,
                ind2lab=self.label_encoder.decode_ndim,
            )
            
            self.mpd_metrics.append(
                ids=ids,
                predict=sequence_per,
                canonical=canonicals,
                perceived=perceiveds,
                predict_len=None,
                canonical_len=canonical_lens,
                perceived_len=perceived_lens,
                ind2lab=self.label_encoder.decode_ndim,
            )

        return loss
    
    def on_stage_start(self, stage, epoch):
        "Gets called when a stage (either training, validation, test) starts."
        self.ctc_metrics_per = self.hparams.ctc_stats()
        self.ctc_metrics_can = self.hparams.ctc_stats_can()
        self.ctc_metrics_diag = self.hparams.ctc_stats()
        
        if hasattr(self.hparams, "augmentation"):
            self.modules.perceived_ssl.model.config.apply_spec_augment = True
            self.modules.canonical_ssl.model.config.apply_spec_augment = True
        
        if stage != sb.Stage.TRAIN:
            self.per_metrics_per = self.hparams.per_stats()
            self.per_metrics_can = self.hparams.per_stats_can()
            self.per_metrics_diag = self.hparams.per_stats()
            self.mpd_metrics = MpdStats()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
            
        else:
            per_can = self.per_metrics_can.summarize("error_rate")
            per_per = self.per_metrics_per.summarize("error_rate")
            mpd_f1 = self.mpd_metrics.summarize("mpd_f1")

        if stage == sb.Stage.VALID:

            self.hparams.train_logger.log_stats(
                stats_meta={
                    "epoch": epoch,
                    "lr_adam": self.adam_optimizer.param_groups[0]["lr"],
                    "lr_pretrained": self.pretrained_opt_class.param_groups[0]["lr"],
                },
                train_stats={"loss": self.train_loss},
                valid_stats={
                    "loss": stage_loss,
                    "ctc_loss_can": self.ctc_metrics_can.summarize("average"),
                    "ctc_loss_per": self.ctc_metrics_per.summarize("average"),
                    "diag_loss": self.ctc_metrics_diag.summarize("average"),
                    "PER_can": per_can,
                    "PER_per": per_per,
                    "mpd_f1": mpd_f1
                },
            )
       # Save best 3 Models
            improved = False
            # Save best 3 PER_per models (lower is better)
            if per_per < self.best_per or len(self.best_per_list) < 3:
                ckpt_name = f"best_per_{epoch:03d}_{per_per:.4f}.ckpt"
                self.checkpointer.save_and_keep_only(
                    meta={"PER_per": per_per, "mpd_f1": mpd_f1, "epoch": epoch},
                    name=ckpt_name,
                    num_to_keep=3,
                    min_keys=["PER_per"]
                )
                self.best_per_list.append((per_per, epoch, ckpt_name))
                self.best_per_list = sorted(self.best_per_list, key=lambda x: x[0])[:3]
                self.best_per = self.best_per_list[0][0]
                improved = True
                # Remove extra checkpoints
                if len(self.best_per_list) > 3:
                    to_remove = self.best_per_list[3:]
                    for _, _, name in to_remove:
                        self.checkpointer.delete_checkpoint(name)
                    self.best_per_list = self.best_per_list[:3]
            # # Save best 3 mpd_f1 models (higher is better)
            # if per_can < self.best_per_cano or len(self.best_per_cano_list) < 3:
            #     ckpt_name = f"best_per_cano_{epoch:03d}_{per_can:.4f}.ckpt"
            #     self.checkpointer.save_and_keep_only(
            #         meta={"PER_can": per_can, "mpd_f1": mpd_f1, "epoch": epoch},
            #         name=ckpt_name,
            #         num_to_keep=3,
            #         min_keys=["PER_can"]
            #     )
            #     self.best_per_cano_list.append((per_can, epoch, ckpt_name))
            #     self.best_per_cano_list = sorted(self.best_per_cano_list, key=lambda x: x[0])[:3]
            #     self.best_per_cano = self.best_per_cano_list[0][0]
            #     improved = True
            #     # Remove extra checkpoints
            #     if len(self.best_per_cano_list) > 3:
            #         to_remove = self.best_per_cano_list[3:]
            #         for _, _, name in to_remove:
            #             self.checkpointer.delete_checkpoint(name)
            #         self.best_per_cano_list = self.best_per_cano_list[:3]
            if mpd_f1 > self.best_mpd_f1 or len(self.best_mpd_f1_list) < 3:
                ckpt_name = f"best_mpdf1_{epoch:03d}_{mpd_f1:.4f}.ckpt"
                self.checkpointer.save_and_keep_only(
                    meta={"PER_can": per_can, "PER_per": per_per, "mpd_f1": mpd_f1, "epoch": epoch},
                    name=ckpt_name,
                    num_to_keep=3,
                    max_keys=["mpd_f1"]
                )
                self.best_mpd_f1_list.append((mpd_f1, epoch, ckpt_name))
                self.best_mpd_f1_list = sorted(self.best_mpd_f1_list, key=lambda x: -x[0])[:3]
                self.best_mpd_f1 = self.best_mpd_f1_list[0][0]
                improved = True
                # Remove extra checkpoints
                if len(self.best_mpd_f1_list) > 3:
                    to_remove = self.best_mpd_f1_list[3:]
                    for _, _, name in to_remove:
                        self.checkpointer.delete_checkpoint(name)
                    self.best_mpd_f1_list = self.best_mpd_f1_list[:3]
            # Early stopping logic
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
                "ctc_loss_per": self.ctc_metrics_per.summarize("average"),
                "ctc_loss_can": self.ctc_metrics_can.summarize("average"),
                "PER_per": per_per,
                "PER_can": per_can,
                "mpd_f1": mpd_f1,
                
            }, step=epoch)
            # Early stop if patience exceeded
            if self.no_improve_epochs >= self.patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {self.patience} epochs)")
                raise StopIteration

            # # Save best model based on MPD-F1 (higher is better)
            # # We'll use a separate checkpoint name to avoid conflicts
            # self.checkpointer.save_checkpoint(
            #     meta={"PER": per, "mpd_f1": mpd_f1, "epoch": epoch},
            #     name="best_mpd_f1_{}.ckpt".format(epoch),
            # )

        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats={"loss": stage_loss, "PER": per_per, "mpd_f1": mpd_f1},
            )
            with open(self.hparams.per_file, "w") as w:
                w.write("CTC loss stats on perceived_phonemes :\n")
                self.ctc_metrics_per.write_stats(w)
                w.write("\nPER on stats:\n")
                self.per_metrics_per.write_stats(w)
                print(
                    "CTC and PER stats on perceived phonemes written to file",
                    self.hparams.per_file,
                )
                
            with open(self.hparams.per_can_file, "w") as p:
                p.write("CTC loss stats on canonical phonemes:\n")
                self.ctc_metrics_can.write_stats(p)
                p.write("PER on canonical phonemes:\n")
                self.per_metrics_can.write_stats(p)
                print(
                    "PER on canonical phonemes stats written to file",
                    self.hparams.per_can_file,
                )
                
            with open(self.hparams.mpd_file, "w") as m:
                m.write("MPD results and stats:\n")
                self.mpd_metrics.write_stats(m)
                print(
                    "MPD results and stats written to file",
                    self.hparams.mpd_file,
                )
        # append the p_diag metrics