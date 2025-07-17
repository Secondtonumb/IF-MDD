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
# from g2p_en import G2p

# g2p = G2p

logger = logging.getLogger(__name__)

@staticmethod
def sentence_to_phoneme(sentence):
    words = sentence.split()
    phonemes = [g2p(word) for word in words]
    phonemes = [p.lower() for p in phonemes]
    # remove " "
    phonemes = [p for p in phonemes if p != " "]
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
class ASR(sb.Brain):
    def on_evaluate_start(self, max_key=None, min_key=None):
        """Gets called at the beginning of evaluation."""
        pass
    
    def compute_forward(self, batch, stage):
        "Given an input batch it computes the phoneme probabilities."
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        # phns_bos, _ = batch.phn_encoded_bos

        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs, wav_lens)

        # some wav2vec models (e.g. large-lv60) needs attention_mask
        if self.modules.wav2vec2.feature_extractor.return_attention_mask:
            attn_mask = make_attn_mask(wavs, wav_lens)
        else:
            attn_mask = None
        feats = self.modules.wav2vec2(wavs, attention_mask=attn_mask)
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
        if self.hparams.wav2vec2_specaug:
            self.modules.wav2vec2.model.config.apply_spec_augment = True

        if stage != sb.Stage.TRAIN:
            self.modules.wav2vec2.model.config.apply_spec_augment = False
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
                    "lr_wav2vec": self.wav2vec_optimizer.param_groups[0]["lr"],
                },
                train_stats={"loss": self.train_loss},
                valid_stats={
                    "loss": stage_loss,
                    "ctc_loss": self.ctc_metrics.summarize("average"),
                    "PER": per,
                    "mpd_f1": mpd_f1
                },
            )
            # wandb logging for validation
            wandb.log({
                "epoch": epoch,
                "train_loss": self.train_loss,
                "valid_loss": stage_loss,
                "ctc_loss": self.ctc_metrics.summarize("average"),
                "PER": per,
                "mpd_f1": mpd_f1,
                "lr_adam": self.adam_optimizer.param_groups[0]["lr"],
                "lr_wav2vec": self.wav2vec_optimizer.param_groups[0]["lr"],
            }, step=epoch)
            
            # Log best models to wandb
            if wandb.run is not None:
                if hasattr(self, 'best_per') and per < self.best_per:
                    self.best_per = per
                    wandb.run.summary["best_per"] = per
                    wandb.run.summary["best_per_epoch"] = epoch
                elif not hasattr(self, 'best_per'):
                    self.best_per = per
                    wandb.run.summary["best_per"] = per
                    wandb.run.summary["best_per_epoch"] = epoch
                    
                if hasattr(self, 'best_mpd_f1') and mpd_f1 > self.best_mpd_f1:
                    self.best_mpd_f1 = mpd_f1
                    wandb.run.summary["best_mpd_f1"] = mpd_f1
                    wandb.run.summary["best_mpd_f1_epoch"] = epoch
                elif not hasattr(self, 'best_mpd_f1'):
                    self.best_mpd_f1 = mpd_f1
                    wandb.run.summary["best_mpd_f1"] = mpd_f1
                    wandb.run.summary["best_mpd_f1_epoch"] = epoch
            # Save best model based on PER (lower is better)
            self.checkpointer.save_and_keep_only(
                meta={"PER": per, "mpd_f1": mpd_f1}, min_keys=["PER"]
            )
            
            # # Save best model based on MPD-F1 (higher is better)
            # # We'll use a separate checkpoint name to avoid conflicts
            # self.checkpointer.save_checkpoint(
            #     meta={"PER": per, "mpd_f1": mpd_f1, "epoch": epoch},
            #     name="best_mpd_f1_{}.ckpt".format(epoch),
            # )

        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats={"loss": stage_loss, "PER": per, "mpd_f1": mpd_f1},
            )
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
        # Managing automatic mixed precision
        if self.auto_mix_prec:

            self.wav2vec_optimizer.zero_grad()
            self.adam_optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.wav2vec_optimizer)
            self.scaler.unscale_(self.adam_optimizer)

            if self.check_gradients(loss):
                self.scaler.step(self.wav2vec_optimizer)
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
                    self.wav2vec_optimizer.step()
                    self.adam_optimizer.step()

                self.wav2vec_optimizer.zero_grad()
                self.adam_optimizer.zero_grad()

        return loss.detach().cpu()

    def init_optimizers(self):
        "Initializes the wav2vec2 optimizer and model optimizer"
        self.wav2vec_optimizer = self.hparams.wav2vec_opt_class(
            self.modules.wav2vec2.model.parameters()
        )
        self.adam_optimizer = self.hparams.adam_opt_class(
            self.hparams.model.parameters()
        )

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable(
                "wav2vec_opt", self.wav2vec_optimizer
            )
            self.checkpointer.add_recoverable("adam_opt", self.adam_optimizer)
    def on_fit_start(self):
        """Gets called at the beginning of ``fit()``, on multiple processes
        if ``distributed_count > 0`` and backend is ddp.

        Default implementation compiles the jit modules, initializes
        optimizers, and loads the latest checkpoint to resume training.
        """
        # Run this *after* starting all processes since jit modules cannot be
        # pickled.
        self._compile_jit()

        # Wrap modules with parallel backend after jit
        self._wrap_distributed()

        # Initialize optimizers after parameters are configured
        self.init_optimizers()

        # Load latest checkpoint to resume training if interrupted
        ## NOTE: make sure to use the "best" model to continual training
        ## so we set the `min_key` argument
        if self.checkpointer is not None:
            self.checkpointer.recover_if_possible(
                device=torch.device(self.device),
                min_key="PER"
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
        sig = hparams["wav2vec2"].feature_extractor(
            librosa.core.load(wav, hparams["sample_rate"])[0],
            sampling_rate=hparams["sample_rate"],
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
    
    if "label_encoder_file" in hparams:
        lab_enc_file = hparams["label_encoder_file"]
    else:
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
        sig = hparams["wav2vec2"].feature_extractor(
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
    # get hparams. label_encoder_file, if not set create one
    if "label_encoder_file" in hparams:
        lab_enc_file = hparams["label_encoder_file"]
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
    else:
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


def dataio_prep_for_timit(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    
    # split train_dev_data into train and valid
    local_data_folder = hparams["timit_local_data_folder"]
    from datasets import load_dataset
    train_dev_data = load_dataset(
        "timit_asr",
        data_dir=local_data_folder,
        split="train")
    train_dev_data = train_dev_data.train_test_split(test_size=0.1, seed=42)
    train_data = train_dev_data["train"]
    valid_data = train_dev_data["test"]
    
    
    test_data = load_dataset(
        "timit_asr",
        data_dir=local_data_folder,
        split="test") 
    import pdb; pdb.set_trace()

    # remove id column
    train_data = train_data.remove_columns("id")
    valid_data = valid_data.remove_columns("id")
    test_data = test_data.remove_columns("id")
    
    # convert Dataset into speechbrain Dataset
    

    train_data_sb = sb.dataio.dataset.DynamicItemDataset.from_arrow_dataset(train_data)
    valid_data_sb = sb.dataio.dataset.DynamicItemDataset.from_arrow_dataset(valid_data)
    test_data_sb = sb.dataio.dataset.DynamicItemDataset.from_arrow_dataset(test_data)
    

    timit2cmu = {
        # Vowels & diphthongs
        "aa": "aa", "ae": "ae", "ah": "ah",
        "ao": "aa", "aw": "aw", "ay": "ay",
        "eh": "eh", "er": "er", "ey": "ey",
        "ih": "ih", "iy": "iy", "ow": "ow",
        "oy": "oy", "uh": "uh", "uw": "uw",

        # Consonants
        "b": "b", "bcl": "b",
        "ch": "ch",
        "d": "d", "dcl": "d",
        "dh": "dh", "dx": "dx",
        "f": "f",
        "g": "g", "gcl": "g",
        "hh": "hh", "hv": "hh",
        "jh": "jh",
        "k": "k", "kcl": "k",
        "l": "l", "el": "l",
        "m": "m", "em": "m",
        "n": "n", "en": "n", "nx": "n",
        "ng": "ng",
        "p": "p", "pcl": "p",
        "r": "r",
        "s": "s",
        "sh": "sh",
        "t": "t", "tcl": "t",
        "th": "th",
        "v": "v",
        "w": "w",
        "y": "y",
        "z": "z",
        "zh": "zh",

        # Silences/closures / fillers
        "pau": "sil", "h#": "sil", "epi": "sil",
        "cl": "sil", "q": "sil"
        
    }
    
    # convert phoneme labels to 40 phonemes
    # (Pdb) test_data
    # Dataset({
    # features: ['file', 'audio', 'text', 'phonetic_detail', 'word_detail', 'dialect_region', 'sentence_type', 'speaker_id', 'id'],
    # num_rows: 1600
    # })
    # (Pdb) dataset_test["phonetic_detail"][0]
    # {'start': [0, 9640, 11240, 12783, 14078, 16157, 16880, 17103, 17587, 18760, 19720, 19962, 21514, 22680, 23800, 24104, 26280, 28591, 29179, 30337, 31880, 32500, 33170, 33829, 35150, 37370, 38568, 40546, 42357, 45119, 45624, 46855, 48680, 49240, 51033, 52378, 54500, 55461, 57395, 59179, 60600], 'stop': [9640, 11240, 12783, 14078, 16157, 16880, 17103, 17587, 18760, 19720, 19962, 21514, 22680, 23800, 24104, 26280, 28591, 29179, 30337, 31880, 32500, 33170, 33829, 35150, 37370, 38568, 40546, 42357, 45119, 45624, 46855, 48680, 49240, 51033, 52378, 54500, 55461, 57395, 59179, 60600, 63440], 'utterance': ['h#', 'sh', 'iy', 'hv', 'ae', 'dcl', 'd', 'y', 'er', 'dcl', 'd', 'aa', 'r', 'kcl', 'k', 's', 'uw', 'dx', 'ih', 'ng', 'gcl', 'g', 'r', 'iy', 's', 'iy', 'w', 'aa', 'sh', 'epi', 'w', 'aa', 'dx', 'er', 'q', 'ao', 'l', 'y', 'iy', 'axr', 'h#']}
    # (Pdb)  dataset_test["word_detail"][0]
    # {'start': [9640, 12783, 17103, 18760, 24104, 29179, 31880, 38568, 45624, 52378, 55461], 'stop': [12783, 17103, 18760, 24104, 29179, 31880, 38568, 45119, 51033, 55461, 60600], 'utterance': ['she', 'had', 'your', 'dark', 'suit', 'in', 'greasy', 'wash', 'water', 'all', 'year']}
    # (Pdb)  dataset_test[0]
    # {'file': '/common/db/TIMIT/timit/test/dr1/faks0/sa1.wav', 'audio': {'path': '/common/db/TIMIT/timit/test/dr1/faks0/sa1.wav', 'array': array([9.15527344e-05, 1.52587891e-04, 6.10351562e-05, ...,   2.44140625e-04, 3.05175781e-04, 2.13623047e-04]), 'sampling_rate': 16000}, 'text': 'She had your dark suit in greasy wash water all year.', 'phonetic_detail': {'start': [0, 9640, 11240, 12783, 14078, 16157, 16880, 17103, 17587, 18760, 19720, 19962, 21514, 22680, 23800, 24104, 26280, 28591, 29179, 30337, 31880, 32500, 33170, 33829, 35150, 37370, 38568, 40546, 42357, 45119, 45624, 46855, 48680, 49240, 51033, 52378, 54500, 55461, 57395, 59179, 60600], 'stop': [9640, 11240, 12783, 14078, 16157, 16880, 17103, 17587, 18760, 19720, 19962, 21514, 22680, 23800, 24104, 26280, 28591, 29179, 30337, 31880, 32500, 33170, 33829, 35150, 37370, 38568, 40546, 42357, 45119, 45624, 46855, 48680, 49240, 51033, 52378, 54500, 55461, 57395, 59179, 60600, 63440], 'utterance': ['h#', 'sh', 'iy', 'hv', 'ae', 'dcl', 'd', 'y', 'er', 'dcl', 'd', 'aa', 'r', 'kcl', 'k', 's', 'uw', 'dx', 'ih', 'ng', 'gcl', 'g', 'r', 'iy', 's', 'iy', 'w', 'aa', 'sh', 'epi', 'w', 'aa', 'dx', 'er', 'q', 'ao', 'l', 'y', 'iy', 'axr', 'h#']}, 'word_detail': {'start': [9640, 12783, 17103, 18760, 24104, 29179, 31880, 38568, 45624, 52378, 55461], 'stop': [12783, 17103, 18760, 24104, 29179, 31880, 38568, 45119, 51033, 55461, 60600], 'utterance': ['she', 'had', 'your', 'dark', 'suit', 'in', 'greasy', 'wash', 'water', 'all', 'year']}, 'dialect_region': 'dr1', 'sentence_type': 'sa', 'speaker_id': 'aks0', 'id': 'sa1'}
    
    @sb.utils.data_pipeline.takes("file", "audio", "text", "phonetic_detail", "word_detail", "dialect_region", "sentence_type", "speaker_id")
    @sb.utils.data_pipeline.provides("phn_list_target", "phn_encoded_target", "phn_list_canonical", "phn_encoded_canonical")
    def simplified_perceived_phonemes(example):
        # Convert perceived phonemes to 40 phonemes
        # Use word sequence to get the canonical phonemes
        if "phonetic_detail" in example and "utterance" in example["phonetic_detail"]:
            phonemes = example["phonetic_detail"]["utterance"]
            # convert to 40 phonemes using timit2cmu mapping
            example["phn_list_target"] = [timit2cmu.get(p, p) for p in phonemes]
            example["phn_list_canonical"] = sentence_to_phoneme(example["text"])

            example["phn_encoded_target"] = label_encoder.encode_sequence(example["phn_list_target"])
            example["phn_encoded_canonical"] = label_encoder.encode_sequence(example["phn_list_canonical"])
        return example
    
    import pdb; pdb.set_trace()
    sb.dataio.dataset.add_dynamic_item([train_data_sb], simplified_perceived_phonemes)
    import pdb; pdb.set_trace()
    sb.dataio.dataset.add_dynamic_item([valid_data_sb, test_data_sb], simplified_perceived_phonemes)
    
    # # 1. Declarations:
    # train_data = sb.dataio.dataset.DynamicItemDataset.from_json(
    #     json_path=hparams["train_annotation"],
    #     replacements={"data_root": data_folder},
    # )
    # if hparams["sorting"] == "ascending":
    #     # we sort training data to speed up training and get better results.
    #     train_data = train_data.filtered_sorted(sort_key="duration")
    #     # when sorting do not shuffle in dataloader ! otherwise is pointless
    #     hparams["train_dataloader_opts"]["shuffle"] = False

    # elif hparams["sorting"] == "descending":
    #     train_data = train_data.filtered_sorted(
    #         sort_key="duration", reverse=True
    #     )
    #     # when sorting do not shuffle in dataloader ! otherwise is pointless
    #     hparams["train_dataloader_opts"]["shuffle"] = False

    # elif hparams["sorting"] == "random":
    #     pass

    # else:
    #     raise NotImplementedError(
    #         "sorting must be random, ascending or descending"
    #     )

    # valid_data = sb.dataio.dataset.DynamicItemDataset.from_json(
    #     json_path=hparams["valid_annotation"],
    #     replacements={"data_root": data_folder},
    # )
    # valid_data = valid_data.filtered_sorted(sort_key="duration")

    # test_data = sb.dataio.dataset.DynamicItemDataset.from_json(
    #     json_path=hparams["test_annotation"],
    #     replacements={"data_root": data_folder},
    # )
    # test_data = test_data.filtered_sorted(sort_key="duration")

    datasets = [train_data, valid_data, test_data]
    label_encoder = sb.dataio.encoder.CTCTextEncoder()
    
    import pdb; pdb.set_trace()
    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("file")
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
        sig = hparams["wav2vec2"].feature_extractor(
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
    # get hparams. label_encoder_file, if not set create one
    if "label_encoder_file" in hparams:
        lab_enc_file = hparams["label_encoder_file"]
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
    else:
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

    
if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # update hparams
    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Dataset IO prep: creating Dataset objects and proper encodings for phones
    train_data, valid_data, test_data, label_encoder = dataio_prep_for_llm(hparams)
    # train_data, valid_data, test_data, label_encoder = dataio_prep_for_timit(hparams)
    
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
    stem = Path(hparams_file).stem
    run_id = time.strftime("%Y%m%d-%H%M%S") + "_" + stem
    run_name = hparams.get("run_name", f"{run_id}")
    
    wandb.init(
        project=hparams.get("wandb_project", "mpl-mdd"), 
        name=run_name,
        id=run_id,
        resume="allow"
    )
    # Training/validation loop
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    # Test
    asr_brain.evaluate(
        test_data,
        test_loader_kwargs=hparams["test_dataloader_opts"],
        min_key="PER",
    )
