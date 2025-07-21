"""
MDD (Mispronunciation Detection and Diagnosis) System - Main Training Script

Author: Haopeng (Kevin) Geng
Institution: University of Tokyo
Year: 2025

This code is provided for non-commercial use only.
For commercial use, please contact the author.

This script implements the main training pipeline for the MDD system using
various SSL models for speech recognition and pronunciation assessment.
"""

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
# simple_g2p  based on cmu dict
# load G2P model

sys.path.append("./trainer")

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

@staticmethod
def sentence_to_phoneme(sentence):
    words = sentence.split()
    phoneme_lists = [g2p(word) for word in words]  # list of list
    phonemes = [p.lower() for sublist in phoneme_lists for p in sublist]  # flatten + lowercase
    phonemes = [p for p in phonemes if p != " "]  # 过滤空格等
    return phonemes

logger = logging.getLogger(__name__)

def make_attn_mask(wavs, wav_lens):
    """
    wav_lens: relative lengths(i.e. 0-1) of a batch. shape: (bs, )
    return a tensor of shape (bs, seq_len), representing mask on allowed positions.
            1 for regular tokens, 0 for padded tokens
    """
    abs_lens = (wav_lens*wavs.shape[1]).long()
    attn_mask = wavs.new(wavs.shape).zero_().long()
    for i in range(len(abs_lens)):
        attn_mask[i, :abs_lens[i]] = 1
    return attn_mask

# Define training procedure
# Mono ASR model
class ASR(sb.Brain):
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
        if self.modules.perceived_ssl.feature_extractor.return_attention_mask:
            attn_mask = make_attn_mask(wavs, wav_lens)
        else:
            attn_mask = None

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
        loss += loss_ctc

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
            with open(self.hparams.wer_file, "w") as w:
                w.write("CTC loss stats:\n")
                self.ctc_metrics.write_stats(w)
                w.write("\nPER stats:\n")
                self.per_metrics.write_stats(w)
                print(
                    "CTC and PER stats written to file",
                    self.hparams.wer_file,
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

class ASR_with_misproBCE(ASR):
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
class ASR_with_misproBCE_proj(ASR):
    def __init__(self, *args, patience=30, **kwargs):
        super().__init__(*args, patience=patience, **kwargs)
        # Add mispro_head module here if not already present
        # encoded_dim should match the encoder output dimension
        # This assumes self.modules.enc exists and has output dim
        # If not available at init, you may need to set it later in model setup
        if self.modules.mispro_head is not None:
            self.modules.mispro_head.to(self.device)
                

    def compute_forward(self, batch, stage):
        "Given an input batch it computes the phoneme probabilities."
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        # phns_bos, _ = batch.phn_encoded_bos

        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs)

        # some wav2vec models (e.g. large-lv60) needs attention_mask
        if self.modules.perceived_ssl.feature_extractor.return_attention_mask:
            attn_mask = make_attn_mask(wavs, wav_lens)
        else:
            attn_mask = None

        feats = self.modules.perceived_ssl(wavs)
        encoder_out = self.modules.enc(feats)

        # output layer for ctc log-probabilities
        logits = self.modules.ctc_lin(encoder_out)
        p_ctc = self.hparams.log_softmax(logits) # (B, T, C)
        
        
        # Compute token-aligned mispronunciation logits
        # We gather encoded frames aligned to targets via CTC alignment or attention.
        # For now, we use a placeholder function `gather_ctc_aligned_reps`, to be implemented.
        token_reps = gather_ctc_aligned_reps(
            encoder_out, batch.phn_encoded_perceived[0], batch.phn_encoded_perceived[1]
        )  # [B, L, D]
        # import pdb; pdb.set_trace()  # Debugging line, remove in production
        mispro_logits = self.modules.mispro_head(token_reps).squeeze(-1)  # [B, L]

        return p_ctc, wav_lens, mispro_logits
        
        # return p_ctc, wav_lens
    def compute_objectives(self, predictions, batch, stage):
        
        # Compute token-aligned mispronunciation logits
        # We gather encoded frames aligned to targets via CTC alignment or attention.
        # For now, we use a placeholder function `gather_ctc_aligned_reps`, to be implemented.
        # Unpack predictions
        p_ctc, wav_lens, mispro_logits = predictions
        ids = batch.id
        targets, target_lens = batch.phn_encoded_target
        canonicals, canonical_lens = batch.phn_encoded_canonical
        perceiveds, perceived_lens = batch.phn_encoded_perceived
        # Additional: BCE loss on binary mispronunciation prediction
        loss_ctc = self.hparams.ctc_cost(p_ctc, targets, wav_lens, target_lens)

        # Compute BCE loss for mispronunciation detection
        mispro_label, mispro_label_lens = batch.mispro_label  # assumed shape [B, L]
        # import pdb; pdb.set_trace()  # Debugging line, remove in production
        loss_bce = torch.nn.functional.binary_cross_entropy_with_logits(mispro_logits, mispro_label.float())
        loss = loss_ctc + 0.5 * loss_bce

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

def dataio_prep(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    data_folder = hparams["data_folder_save"]
    # 1. Declarations:
    train_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["train_annotation"],
        replacements={"data_root": data_folder},
    )
    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["valid_annotation"],
        replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    test_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["test_annotation"],
        replacements={"data_root": data_folder},
    )
    test_data = test_data.filtered_sorted(sort_key="duration")
    datasets = [train_data, valid_data, test_data]
    label_encoder = sb.dataio.encoder.CTCTextEncoder()
    
    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        # sig = sb.dataio.dataio.read_audio(wav)
        # # sample rate change to 16000, e,g, using librosa
        # sig = torch.Tensor(librosa.core.load(wav, hparams["sample_rate"])[0])
        # Use wav2vec processor to do normalization
        
        # Load waveform and resample if needed
        waveform, sr = torchaudio.load(wav)  # waveform: [1, T]

        # Optional: resample to match model sample rate
        target_sr = hparams["sample_rate"]
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            waveform = resampler(waveform)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Apply feature extractor (expecting 1D numpy array)
        sig = hparams["perceived_ssl"].feature_extractor(
            waveform.squeeze(0).numpy(),  # convert to 1D numpy
            sampling_rate=target_sr
        ).input_values[0]

        sig = torch.Tensor(sig)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)
        
    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("perceived_train_target")
    @sb.utils.data_pipeline.provides(
        "phn_list_target",
        "phn_encoded_list_target",
        "phn_encoded_target",
    )
    def text_pipeline_train(phn):
        phn_list = phn.strip().split()
        yield phn_list
        phn_encoded_list = label_encoder.encode_sequence(phn_list)
        yield phn_encoded_list
        phn_encoded = torch.LongTensor(phn_encoded_list)
        yield phn_encoded

    @sb.utils.data_pipeline.takes("perceived_train_target", "canonical_aligned", "perceived_aligned")
    @sb.utils.data_pipeline.provides(
        "phn_list_target",
        "phn_encoded_list_target",
        "phn_encoded_target",
        "phn_list_canonical",
        "phn_encoded_list_canonical",
        "phn_encoded_canonical",
        "phn_list_perceived",
        "phn_encoded_list_perceived",
        "phn_encoded_perceived",
    )
    def text_pipeline_test(target, canonical, perceived):
        phn_list_target = target.strip().split()
        yield phn_list_target
        phn_encoded_list_target = label_encoder.encode_sequence(phn_list_target)
        yield phn_encoded_list_target
        phn_encoded_target = torch.LongTensor(phn_encoded_list_target)
        yield phn_encoded_target
        phn_list_canonical = canonical.strip().split()
        # remove extra spaces
        yield phn_list_canonical
        phn_encoded_list_canonical = label_encoder.encode_sequence(phn_list_canonical)
        yield phn_encoded_list_canonical
        phn_encoded_canonical = torch.LongTensor(phn_encoded_list_canonical)
        yield phn_encoded_canonical
        phn_list_perceived = perceived.strip().split()
        yield phn_list_perceived
        phn_encoded_list_perceived = label_encoder.encode_sequence(phn_list_perceived)
        yield phn_encoded_list_perceived
        phn_encoded_perceived = torch.LongTensor(phn_encoded_list_perceived)
        yield phn_encoded_perceived

    sb.dataio.dataset.add_dynamic_item([train_data], text_pipeline_train)
    sb.dataio.dataset.add_dynamic_item([valid_data, test_data], text_pipeline_test)

    # 3. Fit encoder:
    # Load or compute the label encoder
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    special_labels = {
        "blank_label": hparams["blank_index"],
    }
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[train_data],
        output_key="phn_list_target",
        special_labels=special_labels,
        sequence_input=True,
    )

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        [train_data],
        ["id", "sig", "phn_encoded_target"],
    )
    sb.dataio.dataset.set_output_keys(
        [valid_data, test_data],
        ["id", "sig", "phn_encoded_target", "phn_encoded_canonical", "phn_encoded_perceived"],
    )

    return train_data, valid_data, test_data, label_encoder

def dataio_prep_for_llm(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    data_folder = hparams["data_folder_save"]
    # 1. Declarations:
    train_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["train_annotation"],
        replacements={"data_root": data_folder},
    )
    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["valid_annotation"],
        replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    test_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["test_annotation"],
        replacements={"data_root": data_folder},
    )
    test_data = test_data.filtered_sorted(sort_key="duration")

    datasets = [train_data, valid_data, test_data]
    label_encoder = sb.dataio.encoder.CTCTextEncoder()
    
    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        # sig = sb.dataio.dataio.read_audio(wav)
        # # sample rate change to 16000, e,g, using librosa
        # sig = torch.Tensor(librosa.core.load(wav, hparams["sample_rate"])[0])
        # Use wav2vec processor to do normalization
        waveform, sr = torchaudio.load(wav)  # waveform: [1, T]

        # Optional: resample to match model sample rate
        target_sr = hparams["sample_rate"]
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            waveform = resampler(waveform)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Apply feature extractor (expecting 1D numpy array)
        sig = hparams["perceived_ssl"].feature_extractor(
            waveform.squeeze(0).numpy(),  # convert to 1D numpy
            sampling_rate=target_sr
        ).input_values[0]

        sig = torch.Tensor(sig)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)
        
    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("perceived_train_target")
    @sb.utils.data_pipeline.provides(
        "phn_list_target",
        "phn_encoded_list_target",
        "phn_encoded_target",
        "wrd"
    )
    def text_pipeline_train(phn):
        phn_list = phn.strip().split()
        yield phn_list
        phn_encoded_list = label_encoder.encode_sequence(phn_list)
        yield phn_encoded_list
        phn_encoded = torch.LongTensor(phn_encoded_list)
        yield phn_encoded

    @sb.utils.data_pipeline.takes("perceived_train_target", "canonical_aligned", "perceived_aligned")
    @sb.utils.data_pipeline.provides(
        "phn_list_target",
        "phn_encoded_list_target",
        "phn_encoded_target",
        "phn_list_canonical",
        "phn_encoded_list_canonical",
        "phn_encoded_canonical",
        "phn_list_perceived",
        "phn_encoded_list_perceived",
        "phn_encoded_perceived",
        "mispro_label",
    )
    def text_pipeline_test(target, canonical, perceived):
        phn_list_target = target.strip().split()
        yield phn_list_target
        phn_encoded_list_target = label_encoder.encode_sequence(phn_list_target)
        yield phn_encoded_list_target
        phn_encoded_target = torch.LongTensor(phn_encoded_list_target)
        yield phn_encoded_target
        phn_list_canonical = canonical.strip().split()
        # remove extra spaces
        yield phn_list_canonical
        phn_encoded_list_canonical = label_encoder.encode_sequence(phn_list_canonical)
        yield phn_encoded_list_canonical
        phn_encoded_canonical = torch.LongTensor(phn_encoded_list_canonical)
        yield phn_encoded_canonical
        phn_list_perceived = perceived.strip().split()
        yield phn_list_perceived
        phn_encoded_list_perceived = label_encoder.encode_sequence(phn_list_perceived)
        yield phn_encoded_list_perceived
        phn_encoded_perceived = torch.LongTensor(phn_encoded_list_perceived)
        yield phn_encoded_perceived
        
        mispro_label = [1 if p != c else 0 for p, c in zip(phn_list_perceived, phn_list_canonical)]
        mispro_label = torch.LongTensor(mispro_label)
        # def align_mispro_to_target(canonical_aligned, perceived_aligned, perceived_target, sil_label="sil"):
        #     # 保证都是 list，如果是 str 则 split
        #     can = canonical_aligned if isinstance(canonical_aligned, list) else canonical_aligned.strip().split()
        #     per = perceived_aligned if isinstance(perceived_aligned, list) else perceived_aligned.strip().split()
        #     tgt = perceived_target if isinstance(perceived_target, list) else perceived_target.strip().split()

        #     assert len(can) == len(per), "canonical and perceived must be aligned"

        #     # Step 1: 生成误读标签
        #     mispro_labels = [1 if p != c else 0 for p, c in zip(per, can)]

        #     # Step 2: scan perceived_aligned and align to perceived_target
        #     mapped_labels = []
        #     i = 0  # pointer in perceived_aligned
        #     j = 0  # pointer in perceived_target

        #     while i < len(per) and j < len(tgt):
        #         current_target_phoneme = tgt[j]

        #         group_labels = []
        #         # Match all repeated phonemes (e.g., "sil sil sil")
        #         while i < len(per) and per[i] == current_target_phoneme:
        #             group_labels.append(mispro_labels[i])
        #             i += 1

        #         # Special handling for silence expansion
        #         if current_target_phoneme == sil_label:
        #             while i < len(per) and per[i] == sil_label:
        #                 group_labels.append(mispro_labels[i])
        #                 i += 1

        #         # fallback if not matched
        #         if not group_labels and i < len(per):
        #             group_labels.append(mispro_labels[i])
        #             i += 1

        #         mapped_labels.append(int(any(group_labels)))
        #         j += 1

        #     assert len(mapped_labels) == len(tgt), f"Mismatch! mapped={len(mapped_labels)} vs target={len(tgt)}"
        #     return tgt, mapped_labels
        
        # _, mispro_label = align_mispro_to_target(phn_list_canonical, phn_list_perceived, phn_list_target)
        
        # convert to tensor
        yield mispro_label

    # sb.dataio.dataset.add_dynamic_item([train_data], text_pipeline_train)
    sb.dataio.dataset.add_dynamic_item([train_data], text_pipeline_test)
    sb.dataio.dataset.add_dynamic_item([valid_data, test_data], text_pipeline_test)


    # 3. Fit encoder:
    # Load or compute the label encoder
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    special_labels = {
        "blank_label": hparams["blank_index"],
    }
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[train_data],
        output_key="phn_list_target",
        special_labels=special_labels,
        sequence_input=True,
    )

    # 4. Set output: # use raw phoneme encoding
    sb.dataio.dataset.set_output_keys(
        [train_data],
        ["id",
         "sig", 
         "phn_encoded_target",
        "phn_encoded_canonical",
        "phn_encoded_perceived",
        "phn_list_target",
        "phn_list_canonical",
        "phn_list_perceived",
        "wrd",  # word list, not used in training
        "mispro_label"  # mispronunciation label
        ]
    )
    sb.dataio.dataset.set_output_keys(
        [valid_data, test_data],
        ["id",
         "sig", 
         "phn_encoded_target",
        "phn_encoded_canonical",
        "phn_encoded_perceived",
        "phn_list_target",
        "phn_list_canonical",
        "phn_list_perceived",
        "wrd",  # word list, not used in training
        "mispro_label"  # mispronunciation label
        ]
    )
    
    return train_data, valid_data, test_data, label_encoder

if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)
    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)
    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Dataset IO prep: creating Dataset objects and proper encodings for phones
    # train_data, valid_data, test_data, label_encoder = dataio_prep_for_llm(hparams)
    # train_data, valid_data, test_data, label_encoder = dataio_prep_for_timit(hparams)
    train_data, valid_data, test_data, label_encoder = dataio_prep_for_llm(hparams)
    # test_unit = test_data[0]
    # Trainer initialization
    # asr_brain = ASR(
    #     modules=hparams["modules"],
    #     hparams=hparams,
    #     run_opts=run_opts,
    #     checkpointer=hparams["checkpointer"],
    # )
    asr_brain = ASR_with_misproBCE_proj(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    asr_brain.label_encoder = label_encoder
    # Initialize wandb, 
    # get run_id with time and hparams's name
    from pathlib import Path
    # wandb init group by hparams perceived_ssl_model, canonical_ssl_model, feature_fusion
    perceived_ssl_model = hparams.get("perceived_ssl_model", "Null")
    canonical_ssl_model = hparams.get("canonical_ssl_model", "Null")    
    feature_fusion = hparams.get("feature_fusion", "Null")
    
    run_id = time.strftime("%Y%m%d-%H%M%S") 
    run_name = f"{perceived_ssl_model}_{canonical_ssl_model}_{feature_fusion}"
    # if overrides.is given append its values to run_name
    if isinstance(overrides, dict):
        overrides = [f"{k}={v}" for k, v in overrides.items()]
        run_name += "_" + "_".join(overrides)
    
    run_id = f"{run_name}_{run_id}"
    # wandb init group by hparams perceived_ssl_model, canonical_ssl_model, feature_fusion
    
    wandb.init(
        project=hparams.get("wandb_project", "mdd-v3"), 
        name=run_name,
        id=run_id,
        resume="allow"
    )
    
    # # Training/validation loop
    try:
        asr_brain.fit(
            asr_brain.hparams.epoch_counter,
            train_data,
            valid_data,
            train_loader_kwargs=hparams["train_dataloader_opts"],
            valid_loader_kwargs=hparams["valid_dataloader_opts"],
        )
    except StopIteration:
        print("Training stopped early due to no improvement.")
    # Test
    asr_brain.evaluate(
        test_data,
        test_loader_kwargs=hparams["test_dataloader_opts"],
        # min_key="PER",
        max_key="mpd_f1",  # use max_key for mpd_f1
    )

# === Add placeholder gather_ctc_aligned_reps at top of file ===
