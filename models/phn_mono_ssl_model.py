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
from torch.nn.functional import kl_div
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
    def __init__(self, *args, patience=25, **kwargs):
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
            
        self.best_valid_loss = float('inf')
        self.best_valid_loss_list = []  # List of (valid_loss, epoch, ckpt_name)

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
            # # Save best 3 PER models (lower is better)
            # if per < self.best_per or len(self.best_per_list) < 3:
            #     ckpt_name = f"best_per_{epoch:03d}_{per:.4f}.ckpt"
            #     self.checkpointer.save_and_keep_only(
            #         meta={"PER": per, "mpd_f1": mpd_f1, "epoch": epoch},
            #         name=ckpt_name,
            #         num_to_keep=3,
            #         min_keys=["PER"]
            #     )
            #     self.best_per_list.append((per, epoch, ckpt_name))
            #     self.best_per_list = sorted(self.best_per_list, key=lambda x: x[0])[:3]
            #     self.best_per = self.best_per_list[0][0]
            #     improved = True
            #     # Remove extra checkpoints
            #     if len(self.best_per_list) > 3:
            #         to_remove = self.best_per_list[3:]
            #         for _, _, name in to_remove:
            #             self.checkpointer.delete_checkpoint(name)
            #         self.best_per_list = self.best_per_list[:3]
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
    def __init__(self, *args, patience=25, **kwargs):
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

class PhnMonoSSLModel_withcanoPhnEmb_Hybrid_CTC_Attention(PhnMonoSSLModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Canonical Phoneme Embedding, LSTM + Linear, LSTM out as key, Linear out as value
        self.modules.CanonicalPhonemeEmbedding = torch.nn.Embedding(
            num_embeddings= 42,
            embedding_dim=384  # Use the same dimension as the encoder output
        ).to(self.device)

        self.modules.CanonicalPhonemeLSTM = torch.nn.LSTM(
            input_size=384,
            hidden_size=384 // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        ).to(self.device)

        self.modules.CanonicalPhonemeLinear = torch.nn.Linear(
            in_features=384,
            out_features=384
        ).to(self.device)
        
        # Cross-attention mechanism
        # # Assuming x is the query and canonical_out is the key-value pair
        self.modules.cross_attention = sb.nnet.attention.MultiheadAttention(
            nhead=4,
            d_model=384,
            dropout=0.0
        )

        self.modules.attn_proj = sb.nnet.linear.Linear(
            input_size=384,  # Assuming the output dimension of cross-attention
            n_neurons=42
        )
        self.modules.out_sequence = sb.nnet.linear.Linear(
            input_size=42 *2,  # Concatenation of CTC and attention outputs
            n_neurons=42
        )
        
        # wrap all the added modules in a ModuleList
        self.modules.canoPhnEmb_Hybrid_CTC_Attention = torch.nn.ModuleList([
            self.modules.CanonicalPhonemeEmbedding,
            self.modules.CanonicalPhonemeLSTM,
            self.modules.CanonicalPhonemeLinear,
            self.modules.cross_attention,
            self.modules.attn_proj,
            self.modules.out_sequence
        ]).to(self.device)
        
    def compute_forward(self, batch, stage):
        "Given an input batch it computes the phoneme probabilities."
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        canonicals, canonical_lens = batch.phn_encoded_canonical

        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs)

        feats = self.modules.perceived_ssl(wavs)
        x = self.modules.enc(feats)

        # output layer for ctc log-probabilities
        logits = self.modules.ctc_lin(x)  # [B, T, 42]
        p_ctc = self.hparams.log_softmax(logits)  # [B, T, 42]

        # Get canonical phoneme embeddings
        canonical_embeds = self.modules.CanonicalPhonemeEmbedding(canonicals)  # [B, T, D]
        # Pass through LSTM
        canonical_lstm_out, _ = self.modules.CanonicalPhonemeLSTM(canonical_embeds)  # [B, T, D]
        # Apply linear transformation
        canonical_out = self.modules.CanonicalPhonemeLinear(canonical_lstm_out)  # [B, T, D]

        # Cross-attention mechanism
        # Q: Acoustic x
        # K: Canonical LSTM output
        # V: Canonical Linear output
        # relpostion embedding
        pos_emb = torch.randn(1,
                              x.size(1)*2-1,
                              x.size(2)).to(self.device)
        attn_output, attn_map = self.modules.cross_attention(
            query=x,
            key=canonical_lstm_out,
            value=canonical_out,
        )  # [B, T, D]
        attn_output_logits = self.modules.attn_proj(attn_output)  # [B, T, 42]
        p_attn = self.hparams.log_softmax(attn_output_logits)  # [B, T, 42]

        # Concatenate CTC and attention outputs (both [B, T, 42]) along last dim
        concat_hidden = torch.cat((logits, attn_output_logits), dim=-1)  # [B, T, 768]
        out_logits = self.modules.out_sequence(concat_hidden)  # [B, T, 42]
        p_out = self.hparams.log_softmax(out_logits)  # [B, T, 42]

        return p_out, p_ctc, p_attn, wav_lens, attn_map
        
        
    def compute_objectives(self, predictions, batch, stage):
        "Given the network predictions and targets computed the NLL loss."

        p_out, p_ctc, p_attn, wav_lens, attn_map = predictions

        ids = batch.id
        targets, target_lens = batch.phn_encoded_target
        canonicals, canonical_lens = batch.phn_encoded_canonical 
        perceiveds, perceived_lens = batch.phn_encoded_perceived 
        # len(perceiveds) == len(canonicals) != len(targets) 
        
        # if stage != sb.Stage.TRAIN:
        #     canonicals, canonical_lens = batch.phn_encoded_canonical
        # loss_ctc = self.hparams.ctc_cost(p_ctc, targets, wav_lens, target_lens)
        loss_ctc_ssl_ctc = self.hparams.ctc_cost(p_ctc, targets, wav_lens, target_lens)
        loss_ctc_attn = self.hparams.ctc_cost(p_attn, targets, wav_lens, target_lens)
        loss_ctc_out = self.hparams.ctc_cost(p_out, targets, wav_lens, target_lens)
        
        # Tobe fix
        loss = loss_ctc_attn
        # Log both CTC losses to wandb

        # Record losses for posterity
        if stage != sb.Stage.TRAIN:

            # Note: sb.decoders.ctc_greedy_decode will also remove padded tokens
            # that is, it return a list of list with different lengths
            sequence_ssl_ctc  = sb.decoders.ctc_greedy_decode(
                p_ctc, wav_lens, blank_id=self.hparams.blank_index
            )
            sequence_attn = sb.decoders.ctc_greedy_decode(
                p_attn, wav_lens, blank_id=self.hparams.blank_index
            )
            sequence_out = sb.decoders.ctc_greedy_decode(
                p_out, wav_lens, blank_id=self.hparams.blank_index
            )
            # self.ctc_metrics.append(ids, p_ctc, targets, wav_lens, target_lens)
            # dump first attention map to file
            import matplotlib.pyplot as plt
            # output_dir = self.hparams.output_dir
            # create attn dir
            attn_dir = os.path.join(self.hparams.output_folder, "attention_maps")
            if not os.path.exists(attn_dir):
                os.makedirs(attn_dir, exist_ok=True)
            # get current epoch
            epoch = self.hparams.epoch_counter.current
            if epoch %5 == 0 or epoch==1:
                for attn_id, attn in enumerate(attn_map[0: 1]):
                    plt.figure(figsize=(5, 5))
                    # apply log
                    plt.imshow(torch.log(attn).cpu().detach().numpy(), aspect='auto', origin='lower')
                    
                    plt.title(f"Attention Map for ID {ids[attn_id]}")
                    plt.xlabel("Canonical Phoneme Index")
                    plt.ylabel("Acoustic Feature Index")
                    plt.tight_layout()
                    attn_file = os.path.join(attn_dir, f"{ids[attn_id].split('/')[-1]}_epoch{epoch}.png")
                    plt.savefig(attn_file)
                    plt.close()
                    print(f"Saved attention map to {attn_file}")
                
            self.ctc_metrics.append(ids, p_out, targets, wav_lens, target_lens)

            self.per_metrics.append(
                ids=ids,
                predict=sequence_out,
                target=targets,
                predict_len=None,
                target_len=target_lens,
                ind2lab=self.label_encoder.decode_ndim,
            )
            
            self.mpd_metrics.append(
                ids=ids,
                predict=sequence_out,
                canonical=canonicals,
                perceived=perceiveds,
                predict_len=None,
                canonical_len=canonical_lens,
                perceived_len=perceived_lens,
                ind2lab=self.label_encoder.decode_ndim,
            )

        return loss

    def init_optimizers(self):
        # merge two models parameters
        modules_to_be_trained = torch.nn.ModuleList([
            self.hparams.model,
            self.modules.canoPhnEmb_Hybrid_CTC_Attention
        ])
        
        self.adam_optimizer = self.hparams.adam_opt_class(
            modules_to_be_trained.parameters()
        )
        self.pretrained_opt_class = self.hparams.pretrained_opt_class(
            self.modules.perceived_ssl.parameters(), 
        )

        if self.checkpointer is not None:
            # if self.hparams.perceived_ssl is not None and not self.hparams.perceived_ssl.freeze:
            self.checkpointer.add_recoverable("adam_opt", self.adam_optimizer)
            self.checkpointer.add_recoverable("pretrained_opt", self.pretrained_opt_class)
  
      
    def compute_forward(self, batch, stage):
        "Given an input batch it computes the phoneme probabilities."
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        canonicals, canonical_lens = batch.phn_encoded_canonical

        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs)

        feats = self.modules.perceived_ssl(wavs)
        x = self.modules.enc(feats)

        # output layer for ctc log-probabilities
        logits = self.modules.ctc_lin(x)  # [B, T, 42]
        p_ctc = self.hparams.log_softmax(logits)  # [B, T, 42]

        # Get canonical phoneme embeddings
        canonical_embeds = self.modules.CanonicalPhonemeEmbedding(canonicals)  # [B, T, D]
        # Pass through LSTM
        canonical_lstm_out, _ = self.modules.CanonicalPhonemeLSTM(canonical_embeds)  # [B, T, D]
        # Apply linear transformation
        canonical_out = self.modules.CanonicalPhonemeLinear(canonical_lstm_out)  # [B, T, D]

        # Cross-attention mechanism
        # Q: Acoustic x
        # K: Canonical LSTM output
        # V: Canonical Linear output
        # relpostion embedding
        attn_output, attn_map = self.modules.cross_attention(
            query=x,
            key=canonical_lstm_out,
            value=canonical_out,
        )  # [B, T, D]
        
        # attn_output_logits = self.modules.attn_proj(attn_output)  # [B, T, 42]
        # Concatenate CTC and attention outputs (both [B, T, 42]) along last dim
        concat_hidden = torch.cat((feats, attn_output), dim=-1)  # [B, T, 768 + 384]
        # Convoulutional layer to project to 42 phonemes [786 + 384] -> 42
        concat_hidden = self.modules.cnn(concat_hidden.transpose(1, 2)).transpose(1, 2) # [B, T', 42]

        out_logits = self.modules.out_sequence(concat_hidden)  # [B, T', 42]
        p_ctc = self.hparams.log_softmax(out_logits)  # [B, T', 42]

        return p_ctc, wav_lens, attn_map
    
    def compute_objectives(self, predictions, batch, stage):
        "Given the network predictions and targets computed the NLL loss."

        p_ctc, wav_lens, attn_map = predictions

        ids = batch.id
        targets, target_lens = batch.phn_encoded_target
        canonicals, canonical_lens = batch.phn_encoded_canonical 
        perceiveds, perceived_lens = batch.phn_encoded_perceived 
        # len(perceiveds) == len(canonicals) != len(targets) 
        
        # if stage != sb.Stage.TRAIN:
        #     canonicals, canonical_lens = batch.phn_encoded_canonical
        # loss_ctc = self.hparams.ctc_cost(p_ctc, targets, wav_lens, target_lens)
        loss_ctc = self.hparams.ctc_cost(p_ctc, targets, wav_lens, target_lens)
        # Tobe fix
        loss = loss_ctc
        # Log both CTC losses to wandb

        # Record losses for posterity
        if stage != sb.Stage.TRAIN:

            # Note: sb.decoders.ctc_greedy_decode will also remove padded tokens
            # that is, it return a list of list with different lengths
            sequence = sb.decoders.ctc_greedy_decode(
                p_ctc, wav_lens, blank_id=self.hparams.blank_index
            )
    
            # self.ctc_metrics.append(ids, p_ctc, targets, wav_lens, target_lens)
            # dump first attention map to file
            import matplotlib.pyplot as plt
            # output_dir = self.hparams.output_dir
            # create attn dir
            attn_dir = os.path.join(self.hparams.output_folder, "attention_maps")
            if not os.path.exists(attn_dir):
                os.makedirs(attn_dir, exist_ok=True)
            # get current epoch
            epoch = self.hparams.epoch_counter.current
            if epoch %5 == 0 or epoch==1:
                for attn_id, attn in enumerate(attn_map[0: 1]):
                    plt.figure(figsize=(5, 5))
                    # apply log
                    plt.imshow(attn.cpu().detach().numpy(), aspect='auto', origin='lower')
                    plt.title(f"Attention Map for ID {ids[attn_id]}")
                    plt.xlabel("Canonical Phoneme Index")
                    plt.ylabel("Acoustic Feature Index")
                    plt.tight_layout()
                    attn_file = os.path.join(attn_dir, f"{ids[attn_id].split('/')[-1]}_epoch{epoch}.png")
                    plt.savefig(attn_file)
                    plt.close()
                    print(f"Saved attention map to {attn_file}")
                
                        
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

class PhnMonoSSLModel_withcanoPhnEmb_Hybrid_CTC_Attention_Ver2(PhnMonoSSLModel_withcanoPhnEmb_Hybrid_CTC_Attention):
    """PhnMonoSSLModel_withcanoPhnEmb_Hybrid_CTC_Attention_Ver2
    "https://arxiv.org/abs/2110.07274"
    Args:
        [Attn, SSL] -> Linear -> CTC 
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Canonical Phoneme Embedding, LSTM + Linear, LSTM out as key, Linear out as value
        self.modules.CanonicalPhonemeEmbedding = torch.nn.Embedding(
            num_embeddings=self.hparams.output_neurons,
            embedding_dim=self.hparams.dnn_neurons
        ).to(self.device)

        self.modules.CanonicalPhonemeLSTM = torch.nn.LSTM(
            input_size=self.hparams.dnn_neurons,
            hidden_size=self.hparams.dnn_neurons // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.modules.CanonicalPhonemeLinear = torch.nn.Linear(
            in_features=self.hparams.dnn_neurons,
            out_features=self.hparams.dnn_neurons
        )

        # Cross-attention mechanism: use TransformerEncoderLayer as cross-attention
        self.modules.cross_attention = sb.nnet.attention.MultiheadAttention(
            nhead=4,
            d_model=self.hparams.dnn_neurons,
            dropout=0.1
        )

        self.modules.attn_proj = sb.nnet.linear.Linear(
            input_size=self.hparams.dnn_neurons,
            n_neurons=self.hparams.output_neurons
        )

        self.modules.cnn = torch.nn.Conv1d(
            in_channels=self.hparams.ENCODER_DIM + self.hparams.dnn_neurons,
            out_channels=self.hparams.output_neurons,
            kernel_size=9,
            padding=4
        )
        # Project SSL features to attention dimension for query
        self.modules.q_proj = torch.nn.Linear(
            in_features=self.hparams.ENCODER_DIM,
            out_features=self.hparams.dnn_neurons
        )
        # TransformerDecoder for attention output， want the output time length align with target sequence time length
        
        self.modules.out_sequence = sb.nnet.linear.Linear(
            input_size=self.hparams.output_neurons,
            n_neurons=self.hparams.output_neurons
        )
        # wrap all the added modules in a ModuleList
        self.modules.canoPhnEmb_Hybrid_CTC_Attention = torch.nn.ModuleList([
            self.modules.CanonicalPhonemeEmbedding,
            self.modules.CanonicalPhonemeLSTM,
            self.modules.CanonicalPhonemeLinear,
            self.modules.cross_attention,
            self.modules.attn_proj,
            self.modules.out_sequence,
            self.modules.cnn,
            self.modules.q_proj,
        ]).to(self.device)
        
    def compute_forward(self, batch, stage):
        "Given an input batch it computes the phoneme probabilities."
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        canonicals, canonical_lens = batch.phn_encoded_canonical # [B, T_c]
        perceiveds, perceived_lens = batch.phn_encoded_perceived # [B, T_p]

        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs)

        feats = self.modules.perceived_ssl(wavs) # [B, T, ENC_DIM]
        x = self.modules.enc(feats) # [B, T, d_model]

        # output layer for ctc log-probabilities
        logits = self.modules.ctc_lin(x) # [B, T, D]
        p_ctc = self.hparams.log_softmax(logits) # [B, T, D]

        # Get canonical phoneme embeddings
        canonical_embeds = self.modules.CanonicalPhonemeEmbedding(canonicals) # [B, T_c, d_model]
        # Pass through LSTM
        canonical_lstm_out, _ = self.modules.CanonicalPhonemeLSTM(canonical_embeds) # [B, T_c, d_model]
        # Apply linear transformation
        canonical_out = self.modules.CanonicalPhonemeLinear(canonical_lstm_out) # [B, T_c, d_model]

        # ---- Build key padding mask for canonical sequence (True = pad) ----
        device = canonical_lstm_out.device
        # SpeechBrain length tensors are ratios in [0,1]; convert to absolute token lengths
        max_S = canonical_lstm_out.size(1)
        canon_token_lens = (canonical_lens.to(device).float() * max_S).round().clamp(max=max_S).long()  # [B]
        key_padding_mask = torch.arange(max_S, device=device).unsqueeze(0) >= canon_token_lens.unsqueeze(1)  # [B, S], True = pad

        # ---- Query padding mask (True = pad) ----
        device = feats.device
        T = feats.size(1)
        q_tok_lens = (wav_lens.to(device).float() * T).round().clamp(max=T).long()  # [B]
        query_pad_mask = torch.arange(T, device=device).unsqueeze(0) >= q_tok_lens.unsqueeze(1)  # [B, T]

        # # ---- Build query from SSL feats (pre-CTC encoder) ----
        # q_raw = feats  # [B, T, ENC_DIM], keep richer temporal structure
        # q_proj = self.modules.q_proj(q_raw)  # [B, T, d_model]

        # # zero-out padded query positions
        # q_masked = q_proj.masked_fill(query_pad_mask.unsqueeze(-1), 0.0)
        # 零掉 query 的 pad 位置
        x_masked = x.masked_fill(query_pad_mask.unsqueeze(-1), 0.0)

        # ---- Cross-attention with padding mask ----
        attn_output, attn_map = self.modules.cross_attention(
            query=x_masked,
            key=canonical_lstm_out,
            value=canonical_out,
            attn_mask=None,
            key_padding_mask=key_padding_mask,
        )  # [B, T, d_model]
        # Concatenate CTC and attention outputs
        concat_hidden = torch.cat((feats, attn_output), dim=-1)
        concat_hidden = self.modules.cnn(concat_hidden.transpose(1, 2)).transpose(1, 2)
        # 
        out_logits = self.modules.out_sequence(concat_hidden)
        
        p_ctc = self.hparams.log_softmax(out_logits)
        # attn_output --> TransformerDecoder, 
        p_attn = self.hparams.log_softmax(self.modules.attn_proj(attn_output))
        
        return p_ctc, p_attn, wav_lens, attn_map
    
    def compute_objectives(self, predictions, batch, stage):
        "Given the network predictions and targets computed the NLL loss."

        p_ctc, p_attn, wav_lens, attn_map = predictions

        ids = batch.id
        targets, target_lens = batch.phn_encoded_target
        canonicals, canonical_lens = batch.phn_encoded_canonical 
        perceiveds, perceived_lens = batch.phn_encoded_perceived 
        # len(perceiveds) == len(canonicals) != len(targets) 

        # main CTC loss
        loss_ctc = self.hparams.ctc_cost(p_ctc, targets, wav_lens, target_lens)
        # attention CTC loss for canonical phoneme
        loss_attn_ctc = self.hparams.ctc_cost(p_attn, targets, wav_lens, target_lens)
    
        
        if hasattr(self.hparams, "loss_lambda"):
            lam = self.hparams.loss_lambda
        else:
            lam = 0.5  # default value if not specified
        loss = lam * loss_ctc + (1 - lam) * loss_attn_ctc
        # loss = loss_attn_ctc  # use attention CTC loss as main loss

        # Record losses for posterity
        if stage != sb.Stage.TRAIN:

            # Note: sb.decoders.ctc_greedy_decode will also remove padded tokens
            # that is, it return a list of list with different lengths
            sequence = sb.decoders.ctc_greedy_decode(
                p_ctc, wav_lens, blank_id=self.hparams.blank_index
            )
            # self.ctc_metrics.append(ids, p_ctc, targets, wav_lens, target_lens)
            # dump first attention map to file
            import matplotlib.pyplot as plt
            # output_dir = self.hparams.output_dir
            # create attn dir
            attn_dir = os.path.join(self.hparams.output_folder, "attention_maps")
            if not os.path.exists(attn_dir):
                os.makedirs(attn_dir, exist_ok=True)
            # get current epoch
            epoch = self.hparams.epoch_counter.current
            if epoch %5 == 0 or epoch==1:
                for attn_id, attn in enumerate(attn_map[0: 1]):
                    plt.figure(figsize=(5, 5))
                    # apply log
                    plt.imshow(attn.cpu().detach().numpy(), aspect='auto', origin='lower')
                    plt.title(f"Attention Map for ID {ids[attn_id]}")
                    plt.xlabel("Canonical Phoneme Index")
                    plt.ylabel("Acoustic Feature Index")
                    plt.tight_layout()
                    attn_file = os.path.join(attn_dir, f"{ids[attn_id].split('/')[-1]}_epoch{epoch}.png")
                    plt.savefig(attn_file)
                    plt.close()
                    print(f"Saved attention map to {attn_file}")
                
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

        # Log to wandb if available (VALID stage only)
        if stage == sb.Stage.VALID:
            try:
                import wandb
                wandb.log({
                    "loss_ctc_head": loss_ctc.item(),
                    "loss_ctc_attention": loss_attn_ctc.item(),
                }, step=self.hparams.epoch_counter.current)
            except Exception:
                pass

        return loss

class PhnMonoSSLModel_withcanoPhnEmb_Hybrid_CTC_Attention_Ver3(PhnMonoSSLModel_withcanoPhnEmb_Hybrid_CTC_Attention):
    """PhnMonoSSLModel_withcanoPhnEmb_Hybrid_CTC_Attention_Ver2
    "https://arxiv.org/abs/2110.07274"
    Args:
        [Attn, SSL] -> Linear -> CTC 
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Canonical Phoneme Embedding, LSTM + Linear, LSTM out as key, Linear out as value
        self.modules.CanonicalPhonemeEmbedding = torch.nn.Embedding(
            num_embeddings=self.hparams.output_neurons,
            embedding_dim=self.hparams.dnn_neurons
        ).to(self.device)

        self.modules.CanonicalPhonemeLSTM = torch.nn.LSTM(
            input_size=self.hparams.dnn_neurons,
            hidden_size=self.hparams.dnn_neurons // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.modules.CanonicalPhonemeLinear = torch.nn.Linear(
            in_features=self.hparams.dnn_neurons,
            out_features=self.hparams.dnn_neurons
        )

        # Cross-attention mechanism: use TransformerEncoderLayer as cross-attention
        self.modules.cross_attention = sb.nnet.attention.MultiheadAttention(
            nhead=4,
            d_model=self.hparams.dnn_neurons,
            dropout=0.1
        )

        self.modules.attn_proj = sb.nnet.linear.Linear(
            input_size=self.hparams.dnn_neurons,
            n_neurons=self.hparams.output_neurons
        )

        self.modules.cnn = torch.nn.Conv1d(
            in_channels=self.hparams.ENCODER_DIM + self.hparams.dnn_neurons,
            out_channels=self.hparams.output_neurons,
            kernel_size=9,
            padding=4
        )
        # TransformerDecoder for attention output， want the output time length align with target sequence time length

        self.modules.TransformerDecoder = torch.nn.TransformerDecoder(
            decoder_layer=torch.nn.TransformerDecoderLayer(
                d_model=self.hparams.dnn_neurons,
                nhead=4,
                dim_feedforward=self.hparams.dnn_neurons * 4,
                dropout=0.1
            ),
            num_layers=1,
            norm=None
        )
        
        self.modules.out_sequence = sb.nnet.linear.Linear(
            input_size=self.hparams.output_neurons,
            n_neurons=self.hparams.output_neurons
        )
        # wrap all the added modules in a ModuleList
        self.modules.canoPhnEmb_Hybrid_CTC_Attention = torch.nn.ModuleList([
            self.modules.CanonicalPhonemeEmbedding,
            self.modules.CanonicalPhonemeLSTM,
            self.modules.CanonicalPhonemeLinear,
            self.modules.cross_attention,
            self.modules.attn_proj,
            self.modules.out_sequence,
            self.modules.cnn,
            self.modules.TransformerDecoder,
        ]).to(self.device)
        
    def compute_forward(self, batch, stage):
        "Given an input batch it computes the phoneme probabilities."
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        canonicals, canonical_lens = batch.phn_encoded_canonical

        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs)

        feats = self.modules.perceived_ssl(wavs)
        x = self.modules.enc(feats)

        # output layer for ctc log-probabilities
        logits = self.modules.ctc_lin(x)
        p_ctc = self.hparams.log_softmax(logits)

        # Get canonical phoneme embeddings
        canonical_embeds = self.modules.CanonicalPhonemeEmbedding(canonicals)
        # Pass through LSTM
        canonical_lstm_out, _ = self.modules.CanonicalPhonemeLSTM(canonical_embeds)
        # Apply linear transformation
        canonical_out = self.modules.CanonicalPhonemeLinear(canonical_lstm_out)

        # Prepare key padding mask for canonical_lstm_out

        # Use TransformerEncoderLayer as cross-attention
        # Input x: [B, T, D], src_key_padding_mask: [B, S]
        attn_output, attn_map = self.modules.cross_attention(
            query=x,
            key=canonical_lstm_out,
            value=canonical_out,
        )  # [B, T, D]

        # Concatenate CTC and attention outputs
        concat_hidden = torch.cat((feats, attn_output), dim=-1)
        concat_hidden = self.modules.cnn(concat_hidden.transpose(1, 2)).transpose(1, 2)
        # 
        
        out_logits = self.modules.out_sequence(concat_hidden)
        
        p_ctc = self.hparams.log_softmax(out_logits)
        
        # attn_output --> TransformerDecoder, 
        p_attn = self.hparams.log_softmax(self.modules.attn_proj(attn_output))
        
        return p_ctc, p_attn, wav_lens, attn_map
    
    def compute_objectives(self, predictions, batch, stage):
        "Given the network predictions and targets computed the NLL loss."

        p_ctc, p_attn, wav_lens, attn_map = predictions

        ids = batch.id
        targets, target_lens = batch.phn_encoded_target
        canonicals, canonical_lens = batch.phn_encoded_canonical 
        perceiveds, perceived_lens = batch.phn_encoded_perceived 
        # len(perceiveds) == len(canonicals) != len(targets) 

        eps = 1e-8
        # main CTC loss
        loss_ctc = self.hparams.ctc_cost(p_ctc, targets, wav_lens, target_lens)

        # ---- Build CTC posterior gamma over *canonical* tokens so it matches attn_map key length ----
        gamma = ctc_posteriors_batch(
            log_probs=p_ctc,                    # [B,T,C] log-softmax already
            targets=canonicals,                 # use canonical as target for alignment
            in_lens=wav_lens,
            tgt_lens=canonical_lens,
            blank_id=self.hparams.blank_index,
        )  # [B,T,Lk]

        # attn_map: [B,H,T_q,T_k]; average heads
        # attn_map → [B,H,T,L] 或 [B,T,L]
        if attn_map.dim() == 4:
            A = attn_map.mean(dim=1)        # 平均各 head → [B,T,L]
        elif attn_map.dim() == 3:
            A = attn_map                     # 已经是 [B,T,L]
        else:
            raise ValueError(f"Unexpected attn_map shape {attn_map.shape}")

        B, T, Lk = A.shape
        t_mask = (torch.arange(T, device=A.device)[None, :] >= wav_lens[:, None])  # [B,T]
        k_mask = (torch.arange(Lk, device=A.device)[None, :] >= canonical_lens[:, None])  # [B,Lk]

        # zero-out masked positions then renorm row-wise
        A = A.masked_fill(t_mask.unsqueeze(-1), 0.0)
        A = A.masked_fill(k_mask.unsqueeze(1), 0.0)
        A = A / (A.sum(dim=-1, keepdim=True) + eps)

        gamma = gamma.masked_fill(t_mask.unsqueeze(-1), 0.0)
        gamma = gamma.masked_fill(k_mask.unsqueeze(1), 0.0)
        gamma = gamma / (gamma.sum(dim=-1, keepdim=True) + eps)

        log_A = (A + eps).log()
        # import pdb; pdb.set_trace()
        loss_attn_align = F.kl_div(log_A, gamma, reduction='batchmean')

        alpha = 0.5
        loss = alpha * loss_ctc + (1 - alpha) * loss_attn_align

        # Record losses for posterity
        if stage != sb.Stage.TRAIN:
            # Note: sb.decoders.ctc_greedy_decode will also remove padded tokens
            # that is, it return a list of list with different lengths
            sequence = sb.decoders.ctc_greedy_decode(
                p_ctc, wav_lens, blank_id=self.hparams.blank_index
            )
            # self.ctc_metrics.append(ids, p_ctc, targets, wav_lens, target_lens)
            # dump first attention map to file
            import matplotlib.pyplot as plt
            # output_dir = self.hparams.output_dir
            # create attn dir
            attn_dir = os.path.join(self.hparams.output_folder, "attention_maps")
            if not os.path.exists(attn_dir):
                os.makedirs(attn_dir, exist_ok=True)
            # get current epoch
            epoch = self.hparams.epoch_counter.current
            if epoch %5 == 0 or epoch==1:
                for attn_id, attn in enumerate(attn_map[0: 1]):
                    plt.figure(figsize=(5, 5))
                    # apply log
                    plt.imshow(attn.cpu().detach().numpy(), aspect='auto', origin='lower')
                    plt.title(f"Attention Map for ID {ids[attn_id]}")
                    plt.xlabel("Canonical Phoneme Index")
                    plt.ylabel("Acoustic Feature Index")
                    plt.tight_layout()
                    attn_file = os.path.join(attn_dir, f"{ids[attn_id].split('/')[-1]}_epoch{epoch}.png")
                    plt.savefig(attn_file)
                    plt.close()
                    print(f"Saved attention map to {attn_file}")
                
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

        # Log to wandb if available (VALID stage only)
        if stage == sb.Stage.VALID:
            try:
                import wandb
                wandb.log({
                    "loss_ctc_head": loss_ctc.item(),
                    "loss_attn_align": loss_attn_align.item(),
                }, step=self.hparams.epoch_counter.current)
            except Exception:
                pass

        return loss
    
class PhnMonoSSLModel_Transformer_Hybrid_CTC_Attention(PhnMonoSSLModel_withcanoPhnEmb_Hybrid_CTC_Attention):
    """
    Real Hybrid CTC + Attention model with canonical phoneme embedding. 
    Args:
        PhnMonoSSLModel_withcanoPhnEmb_Hybrid_CTC_Attention (_type_): _description_
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Canonical Phoneme Embedding, LSTM + Linear, LSTM out as key, Linear out as value

        self.modules.CanonicalPhonemeLinear = torch.nn.Linear(
            in_features=self.hparams.dnn_neurons,
            out_features=self.hparams.dnn_neurons
        )

        self.modules.TransformerEncoder = torch.nn.TransformerEncoder(
            encoder_layer=torch.nn.TransformerEncoderLayer(
                d_model=self.hparams.dnn_neurons,
                nhead=4,
                dropout=0.1
            ),
            num_layers=4,
        )
        
        self.modules.cnn = torch.nn.Conv1d(
            in_channels=self.hparams.ENCODER_DIM + self.hparams.dnn_neurons,
            out_channels=self.hparams.output_neurons,
            kernel_size=9,
            padding=4
        )
        
        self.modules.TransformerDecoder = torch.nn.TransformerDecoder(
            decoder_layer=torch.nn.TransformerDecoderLayer(
                d_model=self.hparams.dnn_neurons,
                nhead=4,
                dropout=0.1
            ),
            num_layers=4,
        )
        
        self.modules.out_sequence = sb.nnet.linear.Linear(
            input_size=self.hparams.output_neurons,
            n_neurons=self.hparams.output_neurons
        )
        
        # wrap all the added modules in a ModuleList
        self.modules.canoPhnEmb_Hybrid_CTC_Attention = torch.nn.ModuleList([

        ]).to(self.device)

    def compute_forward(self, batch, stage):
        "Given an input batch it computes the phoneme probabilities."
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        canonicals, canonical_lens = batch.phn_encoded_canonical

        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs)

        feats = self.modules.perceived_ssl(wavs)
        x = self.modules.enc(feats)

        # output layer for ctc log-probabilities
        logits = self.modules.ctc_lin(x)
        p_ctc = self.hparams.log_softmax(logits)

        # Get canonical phoneme embeddings
        # canonical_embeds = self.modules.CanonicalPhonemeEmbedding(canonicals)
        # # Pass through LSTM
        # canonical_lstm_out, _ = self.modules.CanonicalPhonemeLSTM(canonical_embeds)
        # # Apply linear transformation
        # canonical_out = self.modules.CanonicalPhonemeLinear(canonical_lstm_out)

        # Prepare key padding mask for canonical_lstm_out

        # Use TransformerEncoderLayer as cross-attention
        # Input x: [B, T, D], src_key_padding_mask: [B, S]
        canonical_embeds = self.modules.CanonicalPhonemeEmbedding(canonicals)  # [B, T, D]
        canonical_lstm_out, _ = self.modules.CanonicalPhonemeLSTM(canonical_embeds)  # [B, T, D]
        canonical_out = self.modules.CanonicalPhonemeLinear(canonical_lstm_out)  # [B, T, D]
        # Transformer Encoding:
        
        # Concatenate CTC and attention outputs
        concat_hidden = torch.cat((feats, attn_output), dim=-1)
        concat_hidden = self.modules.cnn(concat_hidden.transpose(1, 2)).transpose(1, 2)

        out_logits = self.modules.out_sequence(concat_hidden)
        
        p_ctc = self.hparams.log_softmax(out_logits)
        
        p_attn = self.hparams.log_softmax(self.modules.attn_proj(attn_output))
        
        return p_ctc, p_attn, wav_lens, attn_map
    
    def compute_objectives(self, predictions, batch, stage):
        "Given the network predictions and targets computed the NLL loss."

        p_ctc, p_attn, wav_lens, attn_map = predictions

        ids = batch.id
        targets, target_lens = batch.phn_encoded_target
        canonicals, canonical_lens = batch.phn_encoded_canonical 
        perceiveds, perceived_lens = batch.phn_encoded_perceived 
        # len(perceiveds) == len(canonicals) != len(targets) 
        
        loss_ctc = self.hparams.ctc_cost(p_ctc, targets, wav_lens, target_lens)
        loss_attn_ctc = self.hparams.ctc_cost(p_attn, targets, wav_lens, target_lens)
        alpha = 0.5
        loss = alpha * loss_ctc + (1 - alpha) * loss_attn_ctc

        # Record losses for posterity
        if stage != sb.Stage.TRAIN:

            # Note: sb.decoders.ctc_greedy_decode will also remove padded tokens
            # that is, it return a list of list with different lengths
            sequence = sb.decoders.ctc_greedy_decode(
                p_ctc, wav_lens, blank_id=self.hparams.blank_index
            )
    
            # self.ctc_metrics.append(ids, p_ctc, targets, wav_lens, target_lens)
            # dump first attention map to file
            import matplotlib.pyplot as plt
            # output_dir = self.hparams.output_dir
            # create attn dir
            attn_dir = os.path.join(self.hparams.output_folder, "attention_maps")
            if not os.path.exists(attn_dir):
                os.makedirs(attn_dir, exist_ok=True)
            # get current epoch
            epoch = self.hparams.epoch_counter.current
            if epoch %5 == 0 or epoch==1:
                for attn_id, attn in enumerate(attn_map[0: 1]):
                    plt.figure(figsize=(5, 5))
                    # apply log
                    plt.imshow(attn.cpu().detach().numpy(), aspect='auto', origin='lower')
                    plt.title(f"Attention Map for ID {ids[attn_id]}")
                    plt.xlabel("Canonical Phoneme Index")
                    plt.ylabel("Acoustic Feature Index")
                    plt.tight_layout()
                    attn_file = os.path.join(attn_dir, f"{ids[attn_id].split('/')[-1]}_epoch{epoch}.png")
                    plt.savefig(attn_file)
                    plt.close()
                    print(f"Saved attention map to {attn_file}")
                
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

        # Log both CTC losses to wandb if available (VALID stage only)
        if stage == sb.Stage.VALID:
            try:
                import wandb
                wandb.log({
                    "loss_ctc_head": loss_ctc.item(),
                    "loss_attn_ctc_head": loss_attn_ctc.item(),
                }, step=self.hparams.epoch_counter.current)
            except Exception:
                pass

        return loss