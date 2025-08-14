import os

import torch
import torch.nn

import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from mpd_eval_v3 import MpdStats


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

class TransducerMDDConformerEnc(sb.Brain):
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
        self.best_valid_loss_list = []  # List of (valid_loss, epoch, ckpt_name)label_encoder.add_label("<eos>")
    
    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        predictions = self.compute_forward(batch, stage=stage)
        loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()
    
    def on_stage_start(self, stage, epoch):
        "Gets called when a stage (either training, va lidation, test) starts."
        self.ctc_metrics = self.hparams.ctc_stats()
        self.transducer_metrics = self.hparams.transducer_stats()
        # self.seq_metrics = self.hparams.seqlabel_stats()
        if hasattr(self.hparams, "augmentation"):
            self.modules.perceived_ssl.model.config.apply_spec_augment = True

        if stage != sb.Stage.TRAIN:
            self.per_metrics = self.hparams.per_stats()
            # self.per_metrics_seq = self.hparams.per_stats()
            self.mpd_metrics = MpdStats()
            # self.mpd_metrics_seq = MpdStats()
   
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
        from speechbrain.lobes.models.transformer.Transformer import PositionalEncoding
        from speechbrain.nnet.attention import RelPosEncXL
        
        x = self.modules.enc(feats)  # [B, T_s, ENC_DIM]
        pos_emb = RelPosEncXL(x.shape[-1]).make_pe(seq_len=x.shape[1]).to(self.device) # [1, 2*T_s-1, ENC_DIM]
        
        x, hs = self.modules.Confenc(x, pos_embs=pos_emb)  # [B, T_s, ENC_DIM]
        # PosEnc = PositionalEncoding(x.shape[-1]).to(self.device)

        x = self.modules.enc_lin(x)  # [B, T_s, ENC_DIM]
        
        e_in = self.modules.emb(targets_bos)  # [B, T_p+1, ENC_DIM]
        h, _ = self.modules.dec(e_in)
        h = self.modules.dec_lin(h)  # [B, T_p+1, ENC_DIM]
        
        # Joint network
        joint = self.modules.Tjoint(x.unsqueeze(2), h.unsqueeze(1)) 
        logits = self.modules.output(joint)  # [B, T_p+1, T_s, C]
        
        current_epoch = self.hparams.epoch_counter.current
        hyps = None
        attn_map = None

        if sb.Stage.TRAIN == stage:
            hyps, top_lengths, top_scores, top_log_probs = self.hparams.Greedysearcher(x)
            
        elif sb.Stage.VALID == stage:
            # pdb.set_trace()
            valid_search_interval = self.hparams.valid_search_interval
            if current_epoch % valid_search_interval == 0:
                hyps, top_lengths, top_scores, top_log_probs = self.hparams.Greedysearcher(x)
            return {"p_out": logits,
                    "wav_lens": wav_lens,  # [B]
                    "hyps": hyps,
            }
        elif stage == sb.Stage.TEST:
            hyps, top_lengths, top_scores, top_log_probs = self.hparams.BeamSearcher(x)
            return {"p_out": logits,
                    "wav_lens": wav_lens,  # [B]
                    "hyps": hyps,
            }
        return {
            "p_out": logits,
            "wav_lens": wav_lens,  # [B]
        }
        
    def compute_objectives(self, predictions, batch, stage):
        "Computes the loss for the model."
        current_epoch = self.hparams.epoch_counter.current
        valid_search_interval = self.hparams.valid_search_interval
        
        p_out = predictions["p_out"]
        
        wav_lens = predictions["wav_lens"]
        if len(predictions) == 3:
            hyps = predictions["hyps"]  # [B, T_p+1] or None if not applicable

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
        
        loss = self.hparams.transducer_cost(p_out, targets, wav_lens, target_lens)
        self.transducer_metrics.append(ids, p_out, targets, wav_lens, target_lens)

        if stage != sb.Stage.TRAIN:
            if current_epoch % valid_search_interval == 0 or (
                    stage == sb.Stage.TEST
                ):
                    # Record losses for posterit
                    # Traditional CTC greedy decoding

                sequence_decoder_out = hyps  # [B, T_p+1]
                # CTC-only results
                self.per_metrics.append(
                    ids=ids,
                    predict=hyps,
                    target=targets,
                    predict_len=None,
                    target_len=target_lens,
                    ind2lab=self.label_encoder.decode_ndim,
                )
                    
                # MPD metrics
                self.mpd_metrics.append(
                    ids=ids,
                    predict=hyps,
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
                # min_key="PER",
                max_key="mpd_f1",
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
                
                current_epoch = self.hparams.epoch_counter.current
                valid_search_interval = self.hparams.valid_search_interval
                # Log stats
                valid_stats = {
                    "loss": stage_loss,
                    "PER": per,
                    "mpd_f1": mpd_f1,
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
                
                improved = per_improved 

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
                    "PER": per,
                    "mpd_f1": mpd_f1,
                }, step=epoch)
                # Early stop if patience exceeded
                if self.no_improve_epochs >= self.patience:
                    print(f"Early stopping at epoch {epoch} (no improvement for {self.patience} epochs)")
                    raise StopIteration

        if stage == sb.Stage.TEST:
            per = self.per_metrics.summarize("error_rate")
            mpd_f1 = self.mpd_metrics.summarize("mpd_f1")
            

            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats={"loss": stage_loss, "PER": per, "mpd_f1": mpd_f1,} 
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
            self.scaler.unscale_(self.pretrained_opt_class)
            
            if check_gradients(loss):
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
                
                if check_gradients(loss):
                    self.pretrained_opt_class.step()
                    self.adam_optimizer.step()

                self.pretrained_opt_class.zero_grad()
                self.adam_optimizer.zero_grad()    

        return loss.detach().cpu()

