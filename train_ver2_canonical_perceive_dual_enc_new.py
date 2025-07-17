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
from mpd_eval_v3 import *

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
class ASR_dual_loss(sb.Brain):
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
        feats_can = self.modules.hubert(wavs, attention_mask=attn_mask)
        feats_per = self.modules.wav2vec2(wavs, attention_mask=attn_mask)
        x_can = self.modules.enc1(feats_can)
        x_per = self.modules.enc2(feats_per)

        # output layer for ctc log-probabilities
        logits_can = self.modules.ctc_lin1(x_can)
        logits_per = self.modules.ctc_lin2(x_per)
        
        p_ctc_can = self.hparams.log_softmax(logits_can)
        p_ctc_per = self.hparams.log_softmax(logits_per)
        
        # 这里也可以改一下
        p_ctc = self.hparams.alpha * p_ctc_can + (1 - self.hparams.alpha) * p_ctc_per

        return p_ctc, p_ctc_can, p_ctc_per, wav_lens

    def compute_objectives(self, predictions, batch, stage):
        "Given the network predictions and targets computed the NLL loss."

        p_ctc, p_ctc_can, p_ctc_per, wav_lens = predictions

        ids = batch.id
        targets, target_lens = batch.phn_encoded_target
        canonicals, canonical_lens = batch.phn_encoded_canonical
        
        if stage != sb.Stage.TRAIN:
            perceiveds, perceived_lens = batch.phn_encoded_perceived
        #     canonicals, canonical_lens = batch.phn_encoded_canonical

        # loss_ctc = self.hparams.ctc_cost(p_ctc, targets, wav_lens, target_lens)
        loss_ctc1 = self.hparams.ctc_cost(p_ctc_can, canonicals, wav_lens, canonical_lens)
        loss_ctc2 = self.hparams.ctc_cost(p_ctc_per, targets, wav_lens, target_lens)
        loss = self.hparams.alpha * loss_ctc1 + (1 - self.hparams.alpha) * loss_ctc2
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
        self.ctc_metrics_can = self.hparams.ctc_stats_can()
        self.ctc_metrics_per = self.hparams.ctc_stats_per()
        
        if self.hparams.wav2vec2_specaug:
            self.modules.wav2vec2.model.config.apply_spec_augment = True

        if stage != sb.Stage.TRAIN:
            self.modules.wav2vec2.model.config.apply_spec_augment = False
            self.per_metrics_can = self.hparams.per_stats_can()
            self.per_metrics_per = self.hparams.per_stats_per()
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
                    "lr_wav2vec": self.wav2vec_optimizer.param_groups[0]["lr"],
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
            # wandb logging for validation
            wandb.log({
                "epoch": epoch,
                "train_loss": self.train_loss,
                "valid_loss": stage_loss,
                # "ctc_loss": self.ctc_metrics.summarize("average"),
                "ctc_loss_can": self.ctc_metrics_can.summarize("average"),
                "ctc_loss_per": self.ctc_metrics_per.summarize("average"),
                "PER_can": per_can,
                "PER_per": per_per,
                "mpd_f1": mpd_f1,
                "lr_adam": self.adam_optimizer.param_groups[0]["lr"],
                "lr_wav2vec": self.wav2vec_optimizer.param_groups[0]["lr"],
            }, step=epoch)
            
            # Log best models to wandb
            if wandb.run is not None:
                if hasattr(self, 'best_per') and per_per < self.best_per:
                    self.best_per = per_per
                    wandb.run.summary["best_per"] = per_per
                    wandb.run.summary["best_per_epoch"] = epoch
                elif not hasattr(self, 'best_per'):
                    self.best_per = per_per
                    wandb.run.summary["best_per"] = per_per
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
                meta={"PER": per_per, "mpd_f1": mpd_f1}, min_keys=["PER"]
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
                test_stats={"loss": stage_loss, "PER": per_per, "mpd_f1": mpd_f1},
            )
            with open(self.hparams.wer_file, "w") as w:
                w.write("CTC loss stats:\n")
                
                self.ctc_metrics_can.write_stats(w)
                self.ctc_metrics_per.write_stats(w)
                w.write("\nPER stats:\n")
                # self.per_metrics.write_stats(w)
                # print(
                #     "CTC and PER stats written to file",
                #     self.hparams.wer_file,
                # )
                self.per_metrics_can.write_stats(w)
                self.per_metrics_per.write_stats(w)
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
            self.hubert_optimizer.zero_grad()
            self.adam_optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.wav2vec_optimizer)
            self.scaler.unscale_(self.hubert_optimizer)
            self.scaler.unscale_(self.adam_optimizer)

            if self.check_gradients(loss):
                self.scaler.step(self.wav2vec_optimizer)
                self.scaler.step(self.hubert_optimizer)
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
                    self.hubert_optimizer.step()
                    self.adam_optimizer.step()

                self.wav2vec_optimizer.zero_grad()
                self.hubert_optimizer.zero_grad()
                self.adam_optimizer.zero_grad()

        return loss.detach().cpu()

    def init_optimizers(self):
        "Initializes the wav2vec2 optimizer and model optimizer"
        self.wav2vec_optimizer = self.hparams.wav2vec_opt_class(
            self.modules.wav2vec2.model.parameters()
        )
        self.hubert_optimizer = self.hparams.hubert_opt_class(
            self.modules.hubert.model.parameters()
        )
        self.adam_optimizer = self.hparams.adam_opt_class(
            self.hparams.model.parameters()
        )

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable(
                "wav2vec_opt", self.wav2vec_optimizer
            )
            self.checkpointer.add_recoverable(
                "hubert_opt", self.hubert_optimizer
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

    import pdb; pdb.set_trace()
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
    # 5. calculate confusion matrix between canonical and perceived phonemes, based on label encoder
    
    def create_confusion_matrix(
        from_datasets,
        output_key="phn_list_canonical",
        target_key="phn_list_target",
        normalize=True,
    ):
        """Create a confusion matrix between two sets of phoneme labels."""
        # Get the unique labels from both output and target keys
        output_labels = label_encoder.lab2ind
        output_labels = sorted(output_labels.items(), key=lambda x: x[0])
        # Convert back to dictionary
        output_labels = {k: v for k, v in output_labels}
        
        
        # Create confusion matrix
        cm = torch.zeros(len(output_labels), len(output_labels), dtype=torch.int64)
        for dataset in from_datasets:
            from tqdm import tqdm
            for item in tqdm(dataset):
                output_phns = item[output_key]
                target_phns = item[target_key]
                
                output_phns, target_phns = rm_parallel_sil(output_phns, target_phns)
                assert len(output_phns) == len(target_phns), \
                    f"Output and target phoneme lists must have the same length, got {len(output_phns)} and {len(target_phns)}"
                alignment = extract_alignment(output_phns, target_phns)
                # e.g [('S', 0, 0), ('=', 1, 1), ('S', 2, 2), ('S', 3, 3), ('=', 4, 4), ('=', 5, 5), ('=', 6, 6), ('=', 7, 7), ('S', 8, 8), ('=', 9, 9), ('=', 10, 10), ('=', 11, 11)]
                # if S, count as a mistake, canonical -> perceived phoneme
                # if I, count as mistake as sil-> perceived phoneme
                # if D, count as mistake as canonical -> sil
                # if =, count as a correct prediction
                for mark, o, t in alignment:
                    # o is the index of output phoneme, t is the index of target phoneme
                    # if o is a sil, skip it
                    if mark == "=":  
                        # correct prediction
                        cm[output_labels[output_phns[o]], output_labels[target_phns[t]]] += 1
                    elif mark == "S":
                        # perceived phoneme is a mistake, canonical -> perceived phoneme
                        # increment the confusion matrix
                        cm[output_labels[output_phns[o]], output_labels[target_phns[t]]] += 1
                    elif mark == "I":
                        # perceived phoneme is a mistake, sil -> perceived phoneme
                        # increment the confusion matrix
                        cm[output_labels["sil"], output_labels[target_phns[t]]] += 1
                    elif mark == "D":
                        # canonical phoneme is a mistake, canonical -> sil
                        # increment the confusion matrix
                        cm[output_labels[output_phns[o]], output_labels["sil"]] += 1
                    else:
                        raise ValueError(f"Unknown alignment mark: {mark}")
        if normalize:
            # normalize by output_phns,
            # i.e. divide each row by the sum of the row
            cm = cm.float()
            row_sums = cm.sum(dim=1, keepdim=True)
            cm = cm / row_sums.clamp(min=1e-10)  # avoid division by zero
            # convert to percentage
            cm = cm * 100.0  # convert to percentage
        else:
            # do not normalize, keep the counts
            cm = cm.int()        
        return cm
    label_encoder.create_confusion_matrix = create_confusion_matrix
    
    if hparams["confusion_matrix"]:
        # Create confusion matrix for canonical and perceived phonemes
        # for train dev and test sets
        logger.info("Creating confusion matrix for canonical and perceived phonemes...")
        for dataset in [train_data, valid_data, test_data]:
            cm_can_per_cnt = label_encoder.create_confusion_matrix(
                from_datasets=[dataset],
                output_key="phn_list_perceived",
                target_key="phn_list_canonical",
                normalize=False,
            )
            cm_can_per = label_encoder.create_confusion_matrix(
                from_datasets=[dataset],
                output_key="phn_list_perceived",
                target_key="phn_list_canonical",
                normalize=True,
            )

            # Save the plot by train, valid and test sets
            if dataset == train_data:
                json_file = os.path.join(hparams["save_folder"], "cm_can_per_train.json")
                plot_cnt_file = os.path.join(hparams["save_folder"], "cm_can_per_train.png")
                plot_file = os.path.join(hparams["save_folder"], "cm_can_per_train_per.png")
            elif dataset == valid_data:
                json_file = os.path.join(hparams["save_folder"], "cm_can_per_valid.json")
                plot_cnt_file = os.path.join(hparams["save_folder"], "cm_can_per_valid.png")
                plot_file = os.path.join(hparams["save_folder"], "cm_can_per_valid_per.png")
            else:
                json_file = os.path.join(hparams["save_folder"], "cm_can_per_test.json")
                plot_cnt_file = os.path.join(hparams["save_folder"], "cm_can_per_test.png")
                plot_file = os.path.join(hparams["save_folder"], "cm_can_per_test_per.png")

            # Save the confusion matrix JSON immediately
            with open(json_file, "w") as f:
                json.dump(cm_can_per.tolist(), f)
            logger.info(f"Confusion matrix saved to {json_file}")

            # plot
            import matplotlib.pyplot as plt
            import seaborn as sns

            # Count map
            plt.figure(figsize=(10, 10), dpi=300)
            labels = label_encoder.lab2ind
            sorted_labels = sorted(labels.items(), key=lambda x: x[0])
            sorted_labels = [x[0] for x in sorted_labels]
            # Compute top 10 frequent errors for count map
            error_counts_cnt = []
            for i, can_phn in enumerate(sorted_labels):
                for j, per_phn in enumerate(sorted_labels):
                    if i != j:
                        cnt = cm_can_per_cnt[i, j].item()
                        if cnt > 0:
                            error_counts_cnt.append((cnt, i, j))
            error_counts_cnt.sort(key=lambda x: x[0], reverse=True)
            top_cnt = error_counts_cnt[:10]  # list of (cnt, i, j)
            sns.heatmap(
                cm_can_per_cnt,
                annot=False,
                fmt="d",
                cmap="Blues",
                xticklabels=sorted_labels,
                yticklabels=sorted_labels,
                cbar_kws={"label": "Count"},
                linecolor='red',
                square=True,
            )
            # Highlight top 10 errors in red rectangles
            ax = plt.gca()
            for _, row_idx, col_idx in top_cnt:
                rect = plt.Rectangle((col_idx, row_idx), 1, 1, fill=False, edgecolor='red', linewidth=2)
                ax.add_patch(rect)
            plt.xticks(rotation=90)
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.xlabel("Canonical Phonemes")
            plt.ylabel("Perceived Phonemes")
            plt.title("Confusion Matrix: Count (Perceived vs Canonical Phonemes)")
            plt.savefig(plot_cnt_file)
            plt.close()

            # Percent matrix
            plt.figure(figsize=(10, 10), dpi=300)
            # Compute top 10 frequent errors for percent map
            error_counts_per = []
            for i, can_phn in enumerate(sorted_labels):
                for j, per_phn in enumerate(sorted_labels):
                    if i != j:
                        val = cm_can_per[i, j].item()
                        if val > 0:
                            error_counts_per.append((val, i, j))
            error_counts_per.sort(key=lambda x: x[0], reverse=True)
            top_per = error_counts_per[:10]
            sns.heatmap(
                cm_can_per,
                annot=False,
                fmt=".2f",
                cmap="Blues",
                xticklabels=sorted_labels,
                yticklabels=sorted_labels,
                cbar_kws={"label": "Proportion"},
                linecolor='red',
                square=True,
            )
            # Highlight top 10 errors in red rectangles
            ax = plt.gca()
            for _, row_idx, col_idx in top_per:
                rect = plt.Rectangle((col_idx, row_idx), 1, 1, fill=False, edgecolor='red', linewidth=2)
                ax.add_patch(rect)
            plt.xticks(rotation=90)
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.xlabel("Canonical Phonemes")
            plt.ylabel("Perceived Phonemes")
            plt.title("Confusion Matrix: Proportion (Perceived vs Canonical Phonemes)")
            plt.savefig(plot_file)
            plt.close()

            # Compute and log top N most frequent errors and save mispronunciation dict
            # Determine dataset name for output
            if dataset == train_data:
                dataset_name = "train"
            elif dataset == valid_data:
                dataset_name = "valid"
            elif dataset == test_data:
                dataset_name = "test"
            else:
                dataset_name = "unknown"

            # Compute top N most frequent errors (i.e., non-diagonal counts)
            error_counts = []
            for i, can_phn in enumerate(sorted_labels):
                for j, per_phn in enumerate(sorted_labels):
                    if i != j:
                        cnt = cm_can_per_cnt[i, j].item()
                        if cnt > 0:
                            error_counts.append((cnt, can_phn, per_phn))
            # Sort descending and take top 10
            error_counts.sort(key=lambda x: x[0], reverse=True)
            top_errors = error_counts[:10]
            logger.info(f"Top 10 frequent errors ({dataset_name}):")
            for cnt, can_phn, per_phn in top_errors:
                logger.info(f"{can_phn} -> {per_phn}: {cnt}")

            # Build potential mispronunciations dictionary
            potential_mispronunciations = {}
            for _, can_phn, per_phn in top_errors:
                potential_mispronunciations.setdefault(can_phn, []).append(per_phn)
            # Save to JSON
            mispronunciation_file = os.path.join(
                hparams["save_folder"],
                f"potential_mispronunciations_{dataset_name}.json",
            )
            with open(mispronunciation_file, "w") as f:
                json.dump(potential_mispronunciations, f, indent=4)
            logger.info(f"Potential mispronunciations saved to {mispronunciation_file}")
        import pdb; pdb.set_trace()
    else:
        logger.info("Confusion matrix not created, set confusion_matrix to True in hparams")
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
    train_data, valid_data, test_data, label_encoder = dataio_prep_for_llm(hparams)
    
    # Trainer initialization
    asr_brain = ASR_dual_loss(
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
