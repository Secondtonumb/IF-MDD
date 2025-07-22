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

# Placeholder: gather CTC-aligned representations for each token
def gather_ctc_aligned_reps(encoded, targets, target_lens):
    # Placeholder: ideally, implement alignment logic here
    # For now, simply average pool the encoded features into L segments
    B, T, D = encoded.size()
    L = targets.size(1)
    avg_len = T // L
    reps = []
    for i in range(L):
        reps.append(encoded[:, i * avg_len : (i + 1) * avg_len].mean(dim=1))
    return torch.stack(reps, dim=1)  # [B, L, D]

def simplify_phoneme(p):
    return re.sub(r"\d", "", p).lower()  # e.g., 'AA1' -> 'aa'

class PhnMonoSSLModel(sb.Brain):
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
        if self.modules.perceived_ssl is not None:
            self.modules.perceived_ssl.to(self.device)
        if self.modules.canonical_ssl is not None:
            self.modules.canonical_ssl.to(self.device)
        if self.modules.enc is not None:
            self.modules.enc.to(self.device)
        if self.modules.ctc_lin is not None:
            self.modules.ctc_lin.to(self.device)

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

        # some wav2vec models (e.g. large-lv60) needs attention_mask
        # if self.modules.perceived_ssl.feature_extractor.return_attention_mask:
        #     attn_mask = make_attn_mask(wavs, wav_lens)
        # else:
        #     attn_mask = None

        feats = self.modules.perceived_ssl(wavs)
        x = self.modules.enc(feats)

        # output layer for ctc log-probabilities
        logits = self.modules.ctc_lin(x)
        p_ctc = self.hparams.log_softmax(logits) # (B, T, C)
        return p_ctc, wav_lens

    def compute_objectives(self, predictions, batch, stage):
        "Given the network predictions and targets computed the NLL loss."
        p_ctc, wav_lens = predictions
        ids = batch.id
        targets, target_lens = batch.phn_encoded_target
        # Additional: BCE loss on binary mispronunciation prediction
        if stage != sb.Stage.TRAIN:
            canonicals, canonical_lens = batch.phn_encoded_canonical
            perceiveds, perceived_lens = batch.phn_encoded_perceived

        loss_ctc = self.hparams.ctc_cost(p_ctc, targets, wav_lens, target_lens)
        loss = loss_ctc

        # Record losses for posterity
        if stage != sb.Stage.TRAIN:
            # Note: sb.decoders.ctc_greedy_decode will also remove padded tokens
            # that is, it return a list of list with different lengths
            sequence = sb.decoders.ctc_greedy_decode(
                p_ctc, wav_lens, blank_id=self.hparams.blank_index
            )
            self.ctc_metrics.append(ids, p_ctc, targets, wav_lens, target_lens)

            self.per_metrics.append(
                ids=ids,
                predict=sequence,
                target=targets,
                predict_len=None,
                target_len=target_lens,
                ind2lab=self.label_encoder.decode_ndim,
            )
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

    def on_stage_start(self, stage, epoch):
        "Gets called when a stage (either training, validation, test) starts."
        self.ctc_metrics = self.hparams.ctc_stats()
        if hasattr(self.hparams, "augmentation"):
            self.modules.perceived_ssl.model.config.apply_spec_augment = True

        if stage != sb.Stage.TRAIN:
  
            self.per_metrics = self.hparams.per_stats()
            self.mpd_metrics = MpdStats()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        else:
            per = self.per_metrics.summarize("error_rate")
            mpd_f1 = self.mpd_metrics.summarize("mpd_f1")

        if stage == sb.Stage.VALID:
            # Log stats
            self.hparams.train_logger.log_stats(
                stats_meta={
                    "epoch": epoch,
                    "lr_adam": self.adam_optimizer.param_groups[0]["lr"],
                    "lr_pretrained": self.pretrained_opt_class.param_groups[0]["lr"],
                },
                train_stats={"loss": self.train_loss},
                valid_stats={
                    "loss": stage_loss,
                    "ctc_loss": self.ctc_metrics.summarize("average"),
                    "PER": per,
                    "mpd_f1": mpd_f1
                },
            )
            # Save best 3 Models
            improved = False
            # Save best 3 PER models (lower is better)
            if per < self.best_per or len(self.best_per_list) < 3:
                ckpt_name = f"best_per_{epoch:03d}_{per:.4f}.ckpt"
                self.checkpointer.save_and_keep_only(
                    meta={"PER": per, "mpd_f1": mpd_f1, "epoch": epoch},
                    name=ckpt_name,
                    num_to_keep=3,
                    min_keys=["PER"]
                )
                self.best_per_list.append((per, epoch, ckpt_name))
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
            if mpd_f1 > self.best_mpd_f1 or len(self.best_mpd_f1_list) < 3:
                ckpt_name = f"best_mpdf1_{epoch:03d}_{mpd_f1:.4f}.ckpt"
                self.checkpointer.save_and_keep_only(
                    meta={"PER": per, "mpd_f1": mpd_f1, "epoch": epoch},
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
                "ctc_loss": self.ctc_metrics.summarize("average"),
                "PER": per,
                "mpd_f1": mpd_f1,
                
            }, step=epoch)
            # Early stop if patience exceeded
            if self.no_improve_epochs >= self.patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {self.patience} epochs)")
                raise StopIteration

        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats={"loss": stage_loss, "PER": per, "mpd_f1": mpd_f1},
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

        if self.hparams.auto_mix_prec:
            self.pretrained_opt_class.zero_grad()
            self.adam_optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)

            # normalize the loss by gradient_accumulation and scale for mixed precision
            self.scaler.scale(loss / self.hparams.gradient_accumulation).backward()
            self.scaler.unscale_(self.pretrained_opt_class)
            self.scaler.unscale_(self.adam_optimizer)

            if self.check_gradients(loss):
                self.scaler.step(self.pretrained_opt_class)
                self.scaler.step(self.adam_optimizer)

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
                    self.adam_optimizer.step()

                self.pretrained_opt_class.zero_grad()
                self.adam_optimizer.zero_grad()    

        return loss.detach().cpu()

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
                #max_key="mpd_f1",
            )

class PhnMonoSSLModel_misproBCE(PhnMonoSSLModel):
    def __init__(self, *args, patience=30, **kwargs):
        super().__init__(*args, patience=patience, **kwargs)
        self_best_mispro_ce = float('inf')
        # if self.modules.BCEloss is not None:
        #     self.modules.BCEloss.to(self.device)
            
    def compute_objectives(self, predictions, batch, stage):
        "Given the network predictions and targets computed the NLL loss."
        loss = 0
        p_ctc, wav_lens = predictions
        ids = batch.id
        targets, target_lens = batch.phn_encoded_target
        mis_pro_labels, mis_pro_lens = batch.mispro_label
        canonicals, canonical_lens = batch.phn_encoded_canonical
        perceiveds, perceived_lens = batch.phn_encoded_perceived

        sequence = sb.decoders.ctc_greedy_decode(
            p_ctc, wav_lens, blank_id=self.hparams.blank_index
        )
        
        
        loss_ctc = self.hparams.ctc_cost(p_ctc, targets, wav_lens, target_lens)
        loss_mirpro_BCElike_ctc = self.hparams.ctc_cost_mispro(
            p_ctc, mis_pro_labels, wav_lens, mis_pro_lens
        )
        alpha = 0.8
        loss += (alpha * loss_ctc + (1-alpha)*loss_mirpro_BCElike_ctc)
        # loss += loss_ctc
        
        # Compute BCE loss for mispronunciation detection
        
        # Record losses for posterity
        if stage != sb.Stage.TRAIN:
            # Note: sb.decoders.ctc_greedy_decode will also remove padded tokens
            # that is, it return a list of list with different lengths
            sequence = sb.decoders.ctc_greedy_decode(
                p_ctc, wav_lens, blank_id=self.hparams.blank_index
            )
            self.ctc_metrics.append(ids, p_ctc, targets, wav_lens, target_lens)

            self.per_metrics.append(
                ids=ids,
                predict=sequence,
                target=targets,
                predict_len=None,
                target_len=target_lens,
                ind2lab=self.label_encoder.decode_ndim,
            )
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