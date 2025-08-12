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

from speechbrain.decoders.utils import (
    _update_mem,
    inflate_tensor,
    mask_by_condition,
)

import torch.nn as nn
import torch.nn.functional as F

from utils.layers.utils import make_pad_mask

class CTC_Transformer_dec_only(sb.Brain):
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
        
        feats = self.modules.perceived_ssl(wavs)  # [B, T_s, ENC_DIM]
        feats_enc = self.modules.enc(feats)  # [B, T_s, D]
        
        h_ctc_feat = self.module.ctc_lin(feats_enc)
        p_ctc_logits = self.hparams.log_softmax(h_ctc_feat)  # Log probabilities

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
        if self.checkpointer is not None:
            # if self.hparams.perceived_ssl is not None and not self.hparams.perceived_ssl.freeze:
            self.checkpointer.add_recoverable("adam_opt", self.adam_optimizer)
            self.checkpointer.add_recoverable("pretrained_opt", self.pretrained_opt_class)
            # Add recoverable for freezing states
            # self.checkpointer.add_recoverable("encoder_frozen", self)
            # self.checkpointer.add_recoverable("ssl_frozen", self)
            # self.checkpointer.add_recoverable("best_ctc_loss", self)
            # self.checkpointer.add_recoverable("ctc_no_improve_epochs", self)
            # self.checkpointer.add_recoverable("ctc_loss_history", self)
            # self.checkpointer.add_recoverable("best_valid_per", self)
            # self.checkpointer.add_recoverable("best_valid_f1", self)
            # self.checkpointer.add_recoverable("metric_no_improve_epochs", self)
            # self.checkpointer.add_recoverable("per_history", self)
            # self.checkpointer.add_recoverable("f1_history", self)
            
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
        ctc_loss_merge = self.hparams.dual_ctc_loss * loss_ctc + (1 - self.hparams.dual_ctc_loss) * loss_ctc_post_enc
        
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