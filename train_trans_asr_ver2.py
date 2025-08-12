#!/usr/bin/env python3
"""Recipe for training a Transformer ASR system with CommonVoice
The system employs an encoder, a decoder, and an attention mechanism
between them. Decoding is performed with (CTC/Att joint) beamsearch.

To run this recipe, do the following:
> python train.py hparams/conformer_large.yaml


Authors
 * Titouan Parcollet 2021, 2024
 * Jianyuan Zhong 2020
 * Pooneh Mousavi 2023
"""
import os
import sys

import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.tokenizers.SentencePiece import SentencePiece
from speechbrain.utils.data_utils import undo_padding
from speechbrain.utils.distributed import if_main_process, run_on_main
from speechbrain.utils.logger import get_logger

# add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ver5_train import TimestampDataIOPrepforHybridCTCAttn

logger = get_logger(__name__)


# Define training procedure
class ASR(sb.core.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        # tokens_bos, _ = batch.tokens_bos
        tokens_bos, tokens_bos_lens = batch.phn_encoded_perceived_bos

        # Add waveform augmentation if specified.
        if (
            stage == sb.Stage.TRAIN
            and hasattr(self.hparams, "wav_augment")
            and self.optimizer_step > self.hparams.augment_warmup
        ):
            wavs, wav_lens = self.hparams.wav_augment(wavs, wav_lens)
            tokens_bos = self.hparams.wav_augment.replicate_labels(tokens_bos)

        # compute features
        # feats = self.hparams.compute_features(wavs)
        
        current_epoch = self.hparams.epoch_counter.current
        # feats = self.hparams.normalize(feats, wav_lens, epoch=current_epoch)
        feats = self.modules.perceived_ssl(wavs)
        # pdb.set_trace()
        # Add feature augmentation if specified.
        if (
            stage == sb.Stage.TRAIN
            and hasattr(self.hparams, "fea_augment")
            and self.optimizer_step > self.hparams.augment_warmup
        ):
            feats, fea_lens = self.hparams.fea_augment(feats, wav_lens)
            tokens_bos = self.hparams.fea_augment.replicate_labels(tokens_bos)

        # forward modules
        # src = self.modules.CNN(feats)
        src = self.modules.AcousticEnc(feats)
        # import pdb; pdb.set_trace()
        
        enc_out, pred = self.modules.Transformer(
            src, tokens_bos, wav_lens, pad_idx=self.hparams.pad_index
        )

        # output layer for ctc log-probabilities
        logits = self.modules.ctc_lin(enc_out)
        p_ctc = self.hparams.log_softmax(logits)

        # output layer for seq2seq log-probabilities
        pred = self.modules.seq_lin(pred)
        p_seq = self.hparams.log_softmax(pred)

        # Compute outputs
        hyps = None
        current_epoch = self.hparams.epoch_counter.current
        is_valid_search = (
            stage == sb.Stage.VALID
            and current_epoch % self.hparams.valid_search_interval == 0
        )
        is_test_search = stage == sb.Stage.TEST

        if is_valid_search:
            hyps, _, _, _ = self.hparams.valid_search(
                enc_out.detach(), wav_lens
            )

        elif is_test_search:
            hyps, _, _, _ = self.hparams.test_search(enc_out.detach(), wav_lens)

        return p_ctc, p_seq, wav_lens, hyps

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""

        (p_ctc, p_seq, wav_lens, predicted_tokens) = predictions

        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.phn_encoded_perceived_eos
        # tokens, tokens_lens = batch.tokens
        tokens, tokens_lens = batch.phn_encoded_perceived

        # Augment Labels
        if stage == sb.Stage.TRAIN:
            # Labels must be extended if parallel augmentation or concatenated
            # augmentation was performed on the input (increasing the time dimension)
            if (
                hasattr(self.hparams, "wav_augment")
                and self.optimizer_step > self.hparams.augment_warmup
            ):
                (
                    tokens,
                    tokens_lens,
                    tokens_eos,
                    tokens_eos_lens,
                ) = self.hparams.wav_augment.replicate_multiple_labels(
                    tokens, tokens_lens, tokens_eos, tokens_eos_lens
                )
            if (
                hasattr(self.hparams, "fea_augment")
                and self.optimizer_step > self.hparams.augment_warmup
            ):
                (
                    tokens,
                    tokens_lens,
                    tokens_eos,
                    tokens_eos_lens,
                ) = self.hparams.fea_augment.replicate_multiple_labels(
                    tokens, tokens_lens, tokens_eos, tokens_eos_lens
                )

        loss_seq = self.hparams.seq_cost(
            p_seq, tokens_eos, length=tokens_eos_lens
        )
        loss_ctc = self.hparams.ctc_cost(p_ctc, tokens, wav_lens, tokens_lens)
        # decode ctc branch
        # predicted_ctc = sb.decoders.ctc_greedy_decode(p_ctc, wav_lens, self.hparams.ctc_decoder)
        # print(f"CTC decoded: {predicted_ctc}")
        loss = (
            self.hparams.ctc_weight * loss_ctc
            + (1 - self.hparams.ctc_weight) * loss_seq
        )
        print(f"CTC Loss: {loss_ctc.item():.4f}, Seq2Seq Loss: {loss_seq.item():.4f}, Total Loss: {loss.item():.4f}")
        if stage != sb.Stage.TRAIN:
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval
            if current_epoch % valid_search_interval == 0 or (
                stage == sb.Stage.TEST
            ):
                # Decode token terms to words
                # predicted_words = self.tokenizer(
                #     predicted_tokens, task="decode_from_list"
                # )
                predicted_words = self.label_encoder.decode_ndim(predicted_tokens)
                # Convert indices to words
                target_words = undo_padding(tokens, tokens_lens)
                # target_words = self.tokenizer(
                #     target_words, task="decode_from_list"
                # )
                target_words = self.label_encoder.decode_ndim(target_words)

                self.wer_metric.append(ids, predicted_words, target_words)
                self.cer_metric.append(ids, predicted_words, target_words)
            # compute the accuracy of the one-step-forward prediction
            self.acc_metric.append(p_seq, tokens_eos, tokens_eos_lens)
        return loss

    def on_fit_batch_end(self, batch, outputs, loss, should_step):
        """At the end of the optimizer step, apply noam annealing."""
        if should_step:
            self.hparams.noam_annealing(self.optimizer)

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.acc_metric = self.hparams.acc_computer()
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["ACC"] = self.acc_metric.summarize()
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval
            if (
                current_epoch % valid_search_interval == 0
                or stage == sb.Stage.TEST
            ):
                stage_stats["WER"] = self.wer_metric.summarize("error_rate")
                stage_stats["CER"] = self.cer_metric.summarize("error_rate")

        # log stats and save checkpoint at end-of-epoch
        if stage == sb.Stage.VALID:
            # report different epoch stages according current stage
            current_epoch = self.hparams.epoch_counter.current
            lr = self.hparams.noam_annealing.current_lr
            steps = self.hparams.noam_annealing.n_steps

            epoch_stats = {
                "epoch": epoch,
                "lr": lr,
                "steps": steps,
            }
            self.hparams.train_logger.log_stats(
                stats_meta=epoch_stats,
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"ACC": stage_stats["ACC"], "epoch": epoch},
                max_keys=["ACC"],
            )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            if if_main_process():
                with open(
                    self.hparams.test_wer_file, "w", encoding="utf-8"
                ) as w:
                    self.wer_metric.write_stats(w)

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
            self.optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)

            # normalize the loss by gradient_accumulation and scale for mixed precision
            self.scaler.scale(loss / self.hparams.gradient_accumulation).backward()
            self.scaler.unscale_(self.optimizer)

            if self.check_gradients():
                self.scaler.step(self.optimizer)

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
                self.optimizer.zero_grad()

        return loss.detach().cpu()
    def init_optimizers(self):
        self.optimizer = self.hparams.Adam(
            self.hparams.model.parameters(), 
        )
        if self.checkpointer is not None:
            # if self.hparams.perceived_ssl is not None and not self.hparams.perceived_ssl.freeze:
            self.checkpointer.add_recoverable("adam_opt", self.optimizer)
    
# Define custom data procedure
def dataio_prepare(hparams, tokenizer):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    """

    # 1. Define datasets
    data_folder = hparams["data_folder_save"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["train_annotation"],
        replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            reverse=True,
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
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
    # We also sort the validation data so it is faster to validate
    valid_data = valid_data.filtered_sorted(
        sort_key="duration",
        key_max_value={"duration": hparams["avoid_if_longer_than_val_test"]},
    )

    test_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["test_annotation"],
        replacements={"data_root": data_folder},
    )

    # We also sort the validation data so it is faster to validate
    test_data = test_data.filtered_sorted(
        sort_key="duration",
        key_max_value={"duration": hparams["avoid_if_longer_than_val_test"]},
    )

    datasets = [train_data, valid_data, test_data]
    
    label_encoder = sb.dataio.encoder.TextEncoder()
    # import pdb; pdb.set_trace()
    label_encoder.expect_len = hparams["output_neurons"] 
    # lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    # touch
    
    # special_labels = {
    #     "blank_label": getattr(hparams, "blank_index", 0),
    # }
    
    # label_encoder.load_or_create(
    #     path=lab_enc_file,
    #     from_didatasets=[datasets[0]],
    #     output_key="phn_list_target",
    #     special_labels=special_labels,
    #     sequence_input=True,
    # )

    # max_label = max(label_encoder.lab2ind.values())

    # label_encoder.insert_bos_eos(bos_label="<bos>", eos_label="<eos>",
    #                                     bos_index=max_label + 1, 
    #                                     eos_index=max_label + 2,
    #                                     )

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        info = torchaudio.info(wav)
        sig = sb.dataio.dataio.read_audio(wav)
        resampled = torchaudio.transforms.Resample(
            info.sample_rate,
            hparams["sample_rate"],
        )(sig)
        return resampled

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("perceived_train_target", "canonical_aligned", "perceived_aligned")
    @sb.utils.data_pipeline.provides(
                 
            "phn_list_target",
            "phn_encoded_list_target",
            "phn_encoded_target",
            "phn_list_target_bos",
            "phn_encoded_list_target_bos",
            "phn_encoded_target_bos",
            "phn_list_target_eos",
            "phn_encoded_list_target_eos",
            "phn_encoded_target_eos",

            "phn_list_canonical",
            "phn_encoded_list_canonical",
            "phn_encoded_canonical",
            "phn_list_canonical_bos",
            "phn_encoded_list_canonical_bos",
            "phn_encoded_canonical_bos",
            "phn_list_canonical_eos",
            "phn_encoded_list_canonical_eos",
            "phn_encoded_canonical_eos",

            "phn_list_perceived",
            "phn_encoded_list_perceived",
            "phn_encoded_perceived",
            "phn_list_perceived_bos",
            "phn_encoded_list_perceived_bos",
            "phn_encoded_perceived_bos",
            "phn_list_perceived_eos",
            "phn_encoded_list_perceived_eos",
            "phn_encoded_perceived_eos",
    )
    def text_pipeline(target, canonical, perceived):
        # phn_list_target = target.strip()
        # yield phn_list_target
        # phn_encoded_list_target = tokenizer.sp.encode_as_ids(target.strip())
        # yield phn_encoded_list_target
        # phn_encoded_target = torch.LongTensor(phn_encoded_list_target)
        # yield phn_encoded_target
        # phn_list_target_bos = "<bos> " + phn_list_target
        # yield phn_list_target_bos
        # phn_encoded_list_target_bos = tokenizer.sp.encode_as_ids(phn_list_target_bos)
        # yield phn_encoded_list_target_bos
        # phn_encoded_target_bos = torch.LongTensor(phn_encoded_list_target_bos)
        # yield phn_encoded_target_bos
        # phn_list_target_eos = phn_list_target + " <eos>"
        # yield phn_list_target_eos
        # phn_encoded_list_target_eos = tokenizer.sp.encode_as_ids(phn_list_target_eos)
        # yield phn_encoded_list_target_eos
        # phn_encoded_target_eos = torch.LongTensor(phn_encoded_list_target_eos)
        # yield phn_encoded_target_eos
        # # Canonical
        # phn_list_canonical = canonical.strip()
        # yield phn_list_canonical
        # phn_encoded_list_canonical = tokenizer.sp.encode_as_ids(canonical.strip())
        # yield phn_encoded_list_canonical
        # phn_encoded_canonical = torch.LongTensor(phn_encoded_list_canonical)
        # yield phn_encoded_canonical
        # phn_list_canonical_bos = "<bos> " + phn_list_canonical
        # yield phn_list_canonical_bos
        # phn_encoded_list_canonical_bos = tokenizer.sp.encode_as_ids(phn_list_canonical_bos)
        # yield phn_encoded_list_canonical_bos
        # phn_encoded_canonical_bos = torch.LongTensor(phn_encoded_list_canonical_bos)
        # yield phn_encoded_canonical_bos
        # phn_list_canonical_eos = phn_list_canonical + " <eos>"
        # yield phn_list_canonical_eos
        # phn_encoded_list_canonical_eos = tokenizer.sp.encode_as_ids(phn_list_canonical_eos)
        # yield phn_encoded_list_canonical_eos
        # phn_encoded_canonical_eos = torch.LongTensor(phn_encoded_list_canonical_eos)
        # yield phn_encoded_canonical_eos
        # # Perceived
        # phn_list_perceived = perceived.strip()
        # yield phn_list_perceived
        # phn_encoded_list_perceived = tokenizer.sp.encode_as_ids(perceived.strip())
        # yield phn_encoded_list_perceived
        # phn_encoded_perceived = torch.LongTensor(phn_encoded_list_perceived)
        # yield phn_encoded_perceived
        # phn_list_perceived_bos = "<bos> " + phn_list_perceived
        # yield phn_list_perceived_bos
        # phn_encoded_list_perceived_bos = tokenizer.sp.encode_as_ids(phn_list_perceived_bos)
        # yield phn_encoded_list_perceived_bos
        # phn_encoded_perceived_bos = torch.LongTensor(phn_encoded_list_perceived_bos)
        # yield phn_encoded_perceived_bos
        # phn_list_perceived_eos = phn_list_perceived + " <eos>"
        # yield phn_list_perceived_eos
        # phn_encoded_list_perceived_eos = tokenizer.sp.encode_as_ids(phn_list_perceived_eos)
        # yield phn_encoded_list_perceived_eos
        # phn_encoded_perceived_eos = torch.LongTensor(phn_encoded_list_perceived_eos)
        # yield phn_encoded_perceived_eos
        
        phn_list_target = target.strip().split()
        yield phn_list_target
        phn_encoded_list_target = label_encoder.encode_sequence(phn_list_target)
        yield phn_encoded_list_target
        phn_encoded_target = torch.LongTensor(phn_encoded_list_target)
        yield phn_encoded_target
        phn_list_target_bos = ["<bos>"] + phn_list_target
        yield phn_list_target_bos
        phn_encoded_list_target_bos = label_encoder.encode_sequence(phn_list_target_bos)
        yield phn_encoded_list_target_bos
        phn_encoded_target_bos = torch.LongTensor(phn_encoded_list_target_bos)
        yield phn_encoded_target_bos
        phn_list_target_eos = phn_list_target + ["<eos>"]
        yield phn_list_target_eos
        phn_encoded_list_target_eos = label_encoder.encode_sequence(phn_list_target_eos)
        yield phn_encoded_list_target_eos
        phn_encoded_target_eos = torch.LongTensor(phn_encoded_list_target_eos)
        yield phn_encoded_target_eos
        # Canonical
        phn_list_canonical = canonical.strip().split()
        yield phn_list_canonical
        phn_encoded_list_canonical = label_encoder.encode_sequence(phn_list_canonical)
        yield phn_encoded_list_canonical
        phn_encoded_canonical = torch.LongTensor(phn_encoded_list_canonical)
        yield phn_encoded_canonical
        phn_list_canonical_bos = ["<bos>"] + phn_list_canonical
        yield phn_list_canonical_bos
        phn_encoded_list_canonical_bos = label_encoder.encode_sequence(phn_list_canonical_bos)
        yield phn_encoded_list_canonical_bos
        phn_encoded_canonical_bos = torch.LongTensor(phn_encoded_list_canonical_bos)
        yield phn_encoded_canonical_bos
        phn_list_canonical_eos = phn_list_canonical + ["<eos>"]
        yield phn_list_canonical_eos
        phn_encoded_list_canonical_eos = label_encoder.encode_sequence(phn_list_canonical_eos)
        yield phn_encoded_list_canonical_eos
        phn_encoded_canonical_eos = torch.LongTensor(phn_encoded_list_canonical_eos)
        yield phn_encoded_canonical_eos
        # Perceived
        phn_list_perceived = perceived.strip().split()
        yield phn_list_perceived
        phn_encoded_list_perceived = label_encoder.encode_sequence(phn_list_perceived)
        yield phn_encoded_list_perceived
        phn_encoded_perceived = torch.LongTensor(phn_encoded_list_perceived)
        yield phn_encoded_perceived
        phn_list_perceived_bos = ["<bos>"] + phn_list_perceived
        yield phn_list_perceived_bos
        phn_encoded_list_perceived_bos = label_encoder.encode_sequence(phn_list_perceived_bos)
        yield phn_encoded_list_perceived_bos
        phn_encoded_perceived_bos = torch.LongTensor(phn_encoded_list_perceived_bos)
        yield phn_encoded_perceived_bos
        phn_list_perceived_eos = phn_list_perceived + ["<eos>"]
        yield phn_list_perceived_eos
        phn_encoded_list_perceived_eos = label_encoder.encode_sequence(phn_list_perceived_eos)
        yield phn_encoded_list_perceived_eos
        phn_encoded_perceived_eos = torch.LongTensor(phn_encoded_list_perceived_eos)
        yield phn_encoded_perceived_eos
        
        # tokens_list = tokenizer.sp.encode_as_ids(perceived)
        # yield tokens_list
        # tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        # yield tokens_bos
        # tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        # yield tokens_eos
        # tokens = torch.LongTensor(tokens_list)
        # yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 3. Label encoder:
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    # touch
    
    special_labels = {
        "blank_label": getattr(hparams, "blank_index", 43),
        "pad_label": getattr(hparams, "pad_index", 0),
    }
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[train_data],
        output_key="phn_list_target",
        special_labels=special_labels,
        sequence_input=True,
    )

    label_encoder.insert_bos_eos(bos_label="<bos>", eos_label="<eos>",
                                        bos_index=hparams["bos_index"], 
                                        eos_index=hparams["eos_index"],
                                        )

    label_encoder.expect_len = hparams["output_neurons"]
    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "sig", 
            "phn_list_target",
            "phn_encoded_list_target",
            "phn_encoded_target",
            "phn_list_target_bos",
            "phn_encoded_list_target_bos",
            "phn_encoded_target_bos",
            "phn_list_target_eos",
            "phn_encoded_list_target_eos",
            "phn_encoded_target_eos",
            
            "phn_list_canonical",
            "phn_encoded_list_canonical",
            "phn_encoded_canonical",
            "phn_list_canonical_bos",
            "phn_encoded_list_canonical_bos",
            "phn_encoded_canonical_bos",
            "phn_list_canonical_eos",
            "phn_encoded_list_canonical_eos",
            "phn_encoded_canonical_eos",

            "phn_list_perceived",
            "phn_encoded_list_perceived",
            "phn_encoded_perceived",
            "phn_list_perceived_bos",
            "phn_encoded_list_perceived_bos",
            "phn_encoded_perceived_bos",
            "phn_list_perceived_eos",
            "phn_encoded_list_perceived_eos",
            "phn_encoded_perceived_eos",],
    )

    # 5. If Dynamic Batching is used, we instantiate the needed samplers.
    train_batch_sampler = None
    valid_batch_sampler = None
    if hparams["dynamic_batching"]:
        from speechbrain.dataio.sampler import DynamicBatchSampler  # noqa

        dynamic_hparams_train = hparams["dynamic_batch_sampler_train"]
        dynamic_hparams_valid = hparams["dynamic_batch_sampler_valid"]

        train_batch_sampler = DynamicBatchSampler(
            train_data,
            length_func=lambda x: x["duration"],
            **dynamic_hparams_train,
        )
        valid_batch_sampler = DynamicBatchSampler(
            valid_data,
            length_func=lambda x: x["duration"],
            **dynamic_hparams_valid,
        )

    return (
        train_data,
        valid_data,
        test_data,
        train_batch_sampler,
        valid_batch_sampler,
        label_encoder
    )


if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # Dataset preparation (parsing CommonVoice)
    # from common_voice_prepare import prepare_common_voice  # noqa

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Due to DDP, we do the preparation ONLY on the main python process
    # run_on_main(
    #     prepare_common_voice,
    #     kwargs={
    #         "data_folder": hparams["data_folder"],
    #         "save_folder": hparams["output_folder"],
    #         "train_tsv_file": hparams["train_tsv_file"],
    #         "dev_tsv_file": hparams["dev_tsv_file"],
    #         "test_tsv_file": hparams["test_tsv_file"],
    #         "accented_letters": hparams["accented_letters"],
    #         "language": hparams["language"],
    #         "skip_prep": hparams["skip_prep"],
    #         "convert_to_wav": hparams["convert_to_wav"],
    #     },
    # )

    # Defining tokenizer and loading it
    tokenizer = SentencePiece(
        model_dir=hparams["save_folder"],
        vocab_size=hparams["output_neurons"],
        annotation_train=hparams["train_annotation"],
        annotation_read='perceived_train_target',
        model_type=hparams["token_type"],
        character_coverage=hparams["character_coverage"],
        bos_id=hparams["bos_index"],
        eos_id=hparams["eos_index"],
        annotation_format="json",
        char_format_input=True,
    )

    # here we create the datasets objects as well as tokenization and encoding
    (
        train_data,
        valid_data,
        test_data,
        train_bsampler,
        valid_bsampler,
        label_encoder,
    ) = dataio_prepare(hparams, tokenizer)

    # DataPrep = TimestampDataIOPrepforHybridCTCAttn(hparams)
    
    # train_data, valid_data, test_data, label_encoder = DataPrep.prepare()
    train_bsampler = None
    valid_bsampler = None
    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["Adam"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # adding objects to trainer:
    asr_brain.tokenizer = tokenizer
    asr_brain.label_encoder = label_encoder
    asr_brain.label_encoder.expect_len = hparams["output_neurons"]

    # Manage dynamic batching
    train_dataloader_opts = hparams["train_dataloader_opts"]
    valid_dataloader_opts = hparams["valid_dataloader_opts"]
    if train_bsampler is not None:
        collate_fn = None
        if "collate_fn" in train_dataloader_opts:
            collate_fn = train_dataloader_opts["collate_fn"]

        train_dataloader_opts = {
            "batch_sampler": train_bsampler,
            "num_workers": hparams["num_workers"],
        }

        if collate_fn is not None:
            train_dataloader_opts["collate_fn"] = collate_fn

    if valid_bsampler is not None:
        collate_fn = None
        if "collate_fn" in valid_dataloader_opts:
            collate_fn = valid_dataloader_opts["collate_fn"]

        valid_dataloader_opts = {"batch_sampler": valid_bsampler}

        if collate_fn is not None:
            valid_dataloader_opts["collate_fn"] = collate_fn

    # Training
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=train_dataloader_opts,
        valid_loader_kwargs=valid_dataloader_opts,
    )

    # Testing
    os.makedirs(hparams["output_wer_folder"], exist_ok=True)

    asr_brain.hparams.test_wer_file = os.path.join(
        hparams["output_wer_folder"], "wer_valid.txt"
    )
    asr_brain.evaluate(
        valid_data,
        max_key="ACC",
        test_loader_kwargs=hparams["test_dataloader_opts"],
    )

    asr_brain.hparams.test_wer_file = os.path.join(
        hparams["output_wer_folder"], "wer_test.txt"
    )
    asr_brain.evaluate(
        test_data,
        max_key="ACC",
        test_loader_kwargs=hparams["test_dataloader_opts"],
    )