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
from g2p_en import G2p
import re
# simple_g2p  based on cmu dict
# load G2P model
g2p = G2p()

sys.path.append("./trainer")

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
    def __init__(self, *args, patience=20, **kwargs):
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
        p_ctc = self.hparams.log_softmax(logits)
        return p_ctc, wav_lens

    def compute_objectives(self, predictions, batch, stage):
        "Given the network predictions and targets computed the NLL loss."

        p_ctc, wav_lens = predictions

        ids = batch.id
        targets, target_lens = batch.phn_encoded_target
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
                important_keys=["PER", "mpd_f1"],
                #min_key="PER",
                #max_key="mpd_f1",
            )

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
        "wrd"  # word list, not used in training
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
        "wrd"  # word list, not used in training
        ]
    )
    
    return train_data, valid_data, test_data, label_encoder

def dataio_prep_for_llm_v2(hparams):
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
        "wrd"  # word list, not used in training
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
        "wrd"  # word list, not used in training
        ]
    )
    
    # 5. select samples from test_data to do on-training evaluation
    # if given id_list, select the samples with the given id
    # if not given, select 5 samples from test_data
    if hparams.get("on_training_eval_id_list", None) is not None:
        test_data_on_training = test_data.select(hparams.get("on_training_eval_id_list"))
    else:
        test_data_on_training = test_data.select(range(5))
    
    return train_data, valid_data, test_data, test_data_on_training, label_encoder

# def dataio_prep_for_timit(hparams):
#     """This function prepares the datasets to be used in the brain class.
#     It also defines the data processing pipeline through user-defined functions."""
    
#     # split train_dev_data into train and valid
#     local_data_folder = hparams["timit_local_data_folder"]
#     label_encoder = sb.dataio.encoder.CTCTextEncoder()

#     # 0. Load TIMIT dataset
#     from datasets import load_dataset
#     train_dev_data = load_dataset("timit_asr",data_dir=local_data_folder,split="train")
#     train_dev_data = train_dev_data.train_test_split(test_size=0.1, seed=42)
#     train_data = train_dev_data["train"]
#     valid_data = train_dev_data["test"]
#     # Keep audio column so we can use the waveform
#     # Do NOT remove "audio" column since we need raw audio
    
#     test_data = load_dataset(
#         "timit_asr",
#         data_dir=local_data_folder,
#         split="test") 

#     # remove id column 
#     train_data = train_data.remove_columns(["id"])
#     valid_data = valid_data.remove_columns(["id"])
#     test_data = test_data.remove_columns(["id"])

#     # convert Dataset into speechbrain Dataset
#     train_data_sb = sb.dataio.dataset.DynamicItemDataset.from_arrow_dataset(train_data)
#     valid_data_sb = sb.dataio.dataset.DynamicItemDataset.from_arrow_dataset(valid_data)
#     test_data_sb = sb.dataio.dataset.DynamicItemDataset.from_arrow_dataset(test_data)

#     timit2cmu = {
#         # Vowels & diphthongs
#         "aa": "aa", "ae": "ae", "ah": "ah",
#         "ao": "aa", "aw": "aw", "ay": "ay",
#         "eh": "eh", "er": "er", "ey": "ey",
#         "ih": "ih", "iy": "iy", "ow": "ow",
#         "oy": "oy", "uh": "uh", "uw": "uw",
#         "axr": "er",
#         "ix": "ih",
#         "ux": "uw",
#         "em": "m",
#         "en": "n",
#         "eng": "ng",
#         "nx": "n",
#         "hv": "hh",
#         "ax": "ah",
#         "ix": "iy",
#         "ax-h": "ah",
        
    
#         # Consonants
#         "b": "b", "bcl": "b",
#         "ch": "ch",
#         "d": "d", "dcl": "d",
#         "dh": "dh", "dx": "d",
#         "f": "f",
#         "g": "g", "gcl": "g",
#         "hh": "hh", "hv": "hh",
#         "jh": "jh",
#         "k": "k", "kcl": "k",
#         "l": "l", "el": "l",
#         "m": "m", "em": "m",
#         "n": "n", "en": "n", "nx": "n",
#         "ng": "ng",
#         "p": "p", "pcl": "p",
#         "r": "r",
#         "s": "s",
#         "sh": "sh",
#         "t": "t", "tcl": "t",
#         "th": "th",
#         "v": "v",
#         "w": "w",
#         "y": "y",
#         "z": "z",
#         "zh": "zh",

#         # Silences/closures / fillers
#         "pau": "sil", "h#": "sil", "epi": "sil",
#         "cl": "sil", "q": "sil"
        
#     }
#     # convert phoneme labels to 40 phonemes
#     # (Pdb) test_data
#     # Dataset({
#     # features: ['file', 'audio', 'text', 'phonetic_detail', 'word_detail', 'dialect_region', 'sentence_type', 'speaker_id', 'id'],
#     # num_rows: 1600
#     # })
#     # (Pdb) dataset_test["phonetic_detail"][0]
#     # {'start': [0, 9640, 11240, 12783, 14078, 16157, 16880, 17103, 17587, 18760, 19720, 19962, 21514, 22680, 23800, 24104, 26280, 28591, 29179, 30337, 31880, 32500, 33170, 33829, 35150, 37370, 38568, 40546, 42357, 45119, 45624, 46855, 48680, 49240, 51033, 52378, 54500, 55461, 57395, 59179, 60600], 'stop': [9640, 11240, 12783, 14078, 16157, 16880, 17103, 17587, 18760, 19720, 19962, 21514, 22680, 23800, 24104, 26280, 28591, 29179, 30337, 31880, 32500, 33170, 33829, 35150, 37370, 38568, 40546, 42357, 45119, 45624, 46855, 48680, 49240, 51033, 52378, 54500, 55461, 57395, 59179, 60600, 63440], 'utterance': ['h#', 'sh', 'iy', 'hv', 'ae', 'dcl', 'd', 'y', 'er', 'dcl', 'd', 'aa', 'r', 'kcl', 'k', 's', 'uw', 'dx', 'ih', 'ng', 'gcl', 'g', 'r', 'iy', 's', 'iy', 'w', 'aa', 'sh', 'epi', 'w', 'aa', 'dx', 'er', 'q', 'ao', 'l', 'y', 'iy', 'axr', 'h#']}
#     # (Pdb)  dataset_test["word_detail"][0]
#     # {'start': [9640, 12783, 17103, 18760, 24104, 29179, 31880, 38568, 45624, 52378, 55461], 'stop': [12783, 17103, 18760, 24104, 29179, 31880, 38568, 45119, 51033, 55461, 60600], 'utterance': ['she', 'had', 'your', 'dark', 'suit', 'in', 'greasy', 'wash', 'water', 'all', 'year']}
#     # (Pdb)  dataset_test[0]
#     # {'file': '/common/db/TIMIT/timit/test/dr1/faks0/sa1.wav', 'audio': {'path': '/common/db/TIMIT/timit/test/dr1/faks0/sa1.wav', 'array': array([9.15527344e-05, 1.52587891e-04, 6.10351562e-05, ...,   2.44140625e-04, 3.05175781e-04, 2.13623047e-04]), 'sampling_rate': 16000}, 'text': 'She had your dark suit in greasy wash water all year.', 'phonetic_detail': {'start': [0, 9640, 11240, 12783, 14078, 16157, 16880, 17103, 17587, 18760, 19720, 19962, 21514, 22680, 23800, 24104, 26280, 28591, 29179, 30337, 31880, 32500, 33170, 33829, 35150, 37370, 38568, 40546, 42357, 45119, 45624, 46855, 48680, 49240, 51033, 52378, 54500, 55461, 57395, 59179, 60600], 'stop': [9640, 11240, 12783, 14078, 16157, 16880, 17103, 17587, 18760, 19720, 19962, 21514, 22680, 23800, 24104, 26280, 28591, 29179, 30337, 31880, 32500, 33170, 33829, 35150, 37370, 38568, 40546, 42357, 45119, 45624, 46855, 48680, 49240, 51033, 52378, 54500, 55461, 57395, 59179, 60600, 63440], 'utterance': ['h#', 'sh', 'iy', 'hv', 'ae', 'dcl', 'd', 'y', 'er', 'dcl', 'd', 'aa', 'r', 'kcl', 'k', 's', 'uw', 'dx', 'ih', 'ng', 'gcl', 'g', 'r', 'iy', 's', 'iy', 'w', 'aa', 'sh', 'epi', 'w', 'aa', 'dx', 'er', 'q', 'ao', 'l', 'y', 'iy', 'axr', 'h#']}, 'word_detail': {'start': [9640, 12783, 17103, 18760, 24104, 29179, 31880, 38568, 45624, 52378, 55461], 'stop': [12783, 17103, 18760, 24104, 29179, 31880, 38568, 45119, 51033, 55461, 60600], 'utterance': ['she', 'had', 'your', 'dark', 'suit', 'in', 'greasy', 'wash', 'water', 'all', 'year']}, 'dialect_region': 'dr1', 'sentence_type': 'sa', 'speaker_id': 'aks0', 'id': 'sa1'}
    
#     # 1. Define text pipeline:
#     @sb.utils.data_pipeline.takes("text", "phonetic_detail", "word_detail", "audio")
#     @sb.utils.data_pipeline.provides(
#         "phn_list_target",
#         "phn_encoded_target",
#         "phn_encoded_target_lens",
#         "phn_list_canonical",
#         "phn_encoded_canonical",
#         "phn_encoded_canonical_lens",
#         "phn_list_perceived",
#         "phn_encoded_perceived",
#         "phn_encoded_perceived_lens",
#         "wrd",
#         "sig"
#     )
#     def text_pipeline(text: str, phonetic_detail: dict, word_detail: dict, audio: dict):
#         # Convert perceived phonemes to 40 phonemes
#         # Use word sequence to get the canonical phonemes
#         # phonetic_detail is a dict key,
        
#         ## Perceived phonemes
#         phonemes = phonetic_detail["utterance"]
#                 # convert to 40 phonemes using timit2cmu mapping
#         # clean the phonemes
#         phn_list_perceived = [simplify_phoneme(p) for p in phonemes]
#         phn_list_perceived = [timit2cmu.get(p, p) for p in phn_list_perceived]

#         # 

#         ## Canonical phonemes and Wrd
#         # apply word detail to get the canonical phonemes and wrd
#         word_phonemes = word_detail["utterance"] # list of words
#         # get the canonical phonemes for each word
#         canonical_phonemes = [sentence_to_phoneme(word) for word in word_phonemes]
#         # flatten the list
#         canonical_phonemes = [p for sublist in canonical_phonemes for p in sublist]
#         # simplify the phonemes
#         canonical_phonemes = [simplify_phoneme(p) for p in canonical_phonemes]
#         canonical_phonemes = [timit2cmu.get(p, p) for p in canonical_phonemes]
#         phn_list_canonical = canonical_phonemes

#         ## Wrd
#         wrd = word_detail["utterance"]

#         ## Target  == Perceived
#         phn_list_target = phn_list_perceived
        
#         # Encode the phonemes
#         phn_encoded_target = label_encoder.encode_sequence(phn_list_target)
#         phn_encoded_canonical = label_encoder.encode_sequence(canonical_phonemes)
#         phn_encoded_perceived = label_encoder.encode_sequence(phn_list_perceived)

#         # Yield the results
#         yield phn_list_target
#         yield phn_encoded_target
#         yield phn_list_canonical
#         yield phn_encoded_canonical
#         yield phn_list_perceived
#         yield phn_encoded_perceived
#         yield wrd
#         # Audio waveform to sig
#         waveform = torch.tensor(audio["array"]).float()
#         sr = audio["sampling_rate"]
#         if sr != hparams["sample_rate"]:
#             resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=hparams["sample_rate"])
#             waveform = resampler(waveform)
#         if waveform.ndim > 1:
#             waveform = waveform.mean(dim=0)  # Convert to mono
#         yield waveform

#     sb.dataio.dataset.add_dynamic_item([train_data_sb, valid_data_sb, test_data_sb], text_pipeline)

#     # 3. Fit encoder:
#     # Load or compute the label encoder
#     # get hparams. label_encoder_file, if not set create one
#     if "label_encoder_file" in hparams:
#         lab_enc_file = hparams["label_encoder_file"]
#         special_labels = {
#             "blank_label": hparams["blank_index"],
#         }
        
#         label_encoder.load_or_create(
#             path=lab_enc_file,
#             from_didatasets=[train_data_sb, valid_data_sb, test_data_sb],
#             output_key="phn_list_target",
#             special_labels=special_labels,
#             sequence_input=True,
#         )
#     else:
#         lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")

#         special_labels = {
#             "blank_label": hparams["blank_index"],
#         }
        
#         label_encoder.load_or_create(
#             path=lab_enc_file,
#             from_didatasets=[train_data_sb, valid_data_sb, test_data_sb],
#             output_key="phn_list_target",
#             special_labels=special_labels,
#             sequence_input=True,
#         )
#     # 4. Set output: # use raw phoneme encoding
#     sb.dataio.dataset.set_output_keys(
#         [train_data_sb],
#         [
#             "id",
#             "phn_encoded_target",
#             "phn_encoded_target_lens",
#             "phn_encoded_canonical",
#             "phn_encoded_canonical_lens",
#             "phn_encoded_perceived",
#             "phn_encoded_perceived_lens",
#             "phn_list_target",
#             "phn_list_canonical",
#             "phn_list_perceived",
#             "wrd",
#             "sig"
#         ]
#     )
#     sb.dataio.dataset.set_output_keys(
#         [valid_data_sb, test_data_sb],
#         [
#             "id",
#             "phn_encoded_target",
#             "phn_encoded_target_lens",
#             "phn_encoded_canonical",
#             "phn_encoded_canonical_lens",
#             "phn_encoded_perceived",
#             "phn_encoded_perceived_lens",
#             "phn_list_target",
#             "phn_list_canonical",
#             "phn_list_perceived",
#             "wrd",
#             "sig"
#         ]
#     )

#     return train_data_sb, valid_data_sb, test_data_sb, label_encoder

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
    train_data, valid_data, test_data, label_encoder = dataio_prep_for_llm(hparams)
    # train_data, valid_data, test_data, label_encoder = dataio_prep_for_timit(hparams)
    # test_unit = test_data[0]
    # Trainer initialization
    asr_brain = ASR(
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
    run_id = time.strftime("%Y%m%d-%H%M%S") + "_"
    run_name = f"{perceived_ssl_model}_{canonical_ssl_model}_{feature_fusion}"
    run_id = f"{run_name}_{run_id}"
    # wandb init group by hparams perceived_ssl_model, canonical_ssl_model, feature_fusion
    
    wandb.init(
        project=hparams.get("wandb_project", "mdd-v3"), 
        name=run_name,
        id=run_id,
        resume="allow"
    )
    
    # Training/validation loop
    # try:
    #     asr_brain.fit(
    #         asr_brain.hparams.epoch_counter,
    #         train_data,
    #         valid_data,
    #         train_loader_kwargs=hparams["train_dataloader_opts"],
    #         valid_loader_kwargs=hparams["valid_dataloader_opts"],
    #     )
    # except StopIteration:
    #     print("Training stopped early due to no improvement.")
    # Test
    asr_brain.evaluate(
        test_data,
        test_loader_kwargs=hparams["test_dataloader_opts"],
        min_key="PER",
    )
