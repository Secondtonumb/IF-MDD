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
from models.phn_mono_ssl_model import PhnMonoSSLModel,PhnMonoSSLModel_misproBCE
from models.phn_mono_ssl_model import PhnMonoSSLModel_withcanoPhnEmb_HMA_CTC, HMA_attn_ctc_to_canonical
from models.phn_mono_ssl_model import PhnMonoSSLModel_withcanoPhnEmb_MHA_Guided_Attention_CTC
from models.phn_mono_ssl_model import HMA_attn_ctc_to_mispro

from models.phn_dual_ssl_model import PhnDualSSLModel, PhnDualSSLModel_with_SimpleResidual
from models.phn_dual_ssl_model import PhnDualSSLModel_Hybrid_CTC_Attention
from models.phn_mono_ssl_model_ver2 import (HMA_attn_ctc_to_mispro_ver2,
    HMA_attn_ctc_to_mispro_ver2_1,
    HMA_attn_ctc_to_mispro_ver2_1_perceived,
    HMA_attn_ctc_to_mispro_ver2_2)

from models.phn_mono_ssl_model_ver2 import Hybrid_CTC_Attention

wandb.login(key="1e2455bc962bb682012326b2964a299ed63c3690")

sys.path.append("./trainer")

logger = logging.getLogger(__name__)

# Define training procedure
# Mono ASR model
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
    # label_encoder = sb.dataio.encoder.TextEncoder()
    # label_encoder.add_bos_eos()

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
        # convert to tensor
        yield mispro_label # len(mispro_label) == len(phn_list_perceived) (silence dupilicated)

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

def dataio_prep_for_llm_with_timestamps(hparams):
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
    
    
    # label_encoder.insert_bos_eos("<bos>", "<eos>")  # Add BOS and EOS tokens
    # import pdb; pdb.set_trace()
    # 2. Define audio and text pipeline:
    @sb.utils.data_pipeline.takes("wav", "target_starts", "target_ends", "canonical_starts", "canonical_ends", "perceived_train_target", "canonical_aligned", "perceived_aligned")
    @sb.utils.data_pipeline.provides(
        "sig", 
        "target_start_frames", 
        "target_end_frames", 
        "canonical_start_frames", 
        "canonical_end_frames",
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
        "mispro_label_framewise", # frame-wise mispronunciation label
        "phn_encoded_target_bin",
    )
    def combined_pipeline(wav, target_starts, target_ends, canonical_starts, canonical_ends, target, canonical, perceived):
        # Audio processing
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
        # Move sig to the same device as the SSL model to avoid device/type mismatch

        # Convert string timestamps to frame indices with sample rate and feature_extractor's hop size
        hop_length = sr / 1000 * 20
        target_start_frames = librosa.time_to_frames(target_starts, sr=sr, hop_length=hop_length)
        target_end_frames = librosa.time_to_frames(target_ends, sr=sr, hop_length=hop_length)
        canonical_start_frames = librosa.time_to_frames(canonical_starts, sr=sr, hop_length=hop_length)
        canonical_end_frames = librosa.time_to_frames(canonical_ends, sr=sr, hop_length=hop_length)

        yield sig
        yield target_start_frames
        yield target_end_frames
        yield canonical_start_frames
        yield canonical_end_frames

        # Text processing
        phn_list_target = target.strip().split()
        yield phn_list_target
        phn_encoded_list_target = label_encoder.encode_sequence(phn_list_target)
        yield phn_encoded_list_target
        phn_encoded_target = torch.LongTensor(phn_encoded_list_target)
        yield phn_encoded_target

        phn_list_canonical = canonical.strip().split()
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
        yield mispro_label
        
        # for frame-wise mispronunciation label, 20ms
        # create a tensor of zeros with the same length as sig extrace by hop size == 320
        # feat_unlabeled_length = int(len(sig) / (sr / 1000 * 20))  # 20ms hop size
        
        # def get_feature_length(num_samples, conv_kernel, conv_stride):
        #     L = num_samples
        #     for k, s in zip(conv_kernel, conv_stride):
        #         L = (L - k) // s + 1
        #     return int(L)
        # num_samples = sig.shape[0]
        # feat_unlabeled_length = get_feature_length(
        #     num_samples,
        #     [10, 3, 3, 3, 3, 3, 3],
        #     [5, 2, 2, 2, 2, 2, 2],
        # )
        mispro_label_framewise = torch.zeros(len(sig), dtype=torch.long)
        canonical_starts_bins = [int(x * sr) for x in canonical_starts]
        canonical_ends_bins = [int(x * sr) for x in canonical_ends]
        for flag, start, end in zip(mispro_label, canonical_starts_bins, canonical_ends_bins):
            # label the frames between start and end as mispronounced if the label is 1
            if flag == 1:
                mispro_label_framewise[start:end] = 1
        mispro_label_framewise = torch.LongTensor(mispro_label_framewise)
        yield mispro_label_framewise
        # for binarized target
        phn_encoded_target_bin = torch.zeros(len(sig), dtype=torch.long)
        for phn, start, end in zip(phn_encoded_target,canonical_starts_bins, canonical_ends_bins):
            # label the frames between start and end as 1
            phn_encoded_target_bin[start:end] = phn
        # convert to long tensor
        phn_encoded_target_bin = torch.LongTensor(phn_encoded_target_bin)
        yield phn_encoded_target_bin

    # sb.dataio.dataset.add_dynamic_item([train_data], text_pipeline_train)
    sb.dataio.dataset.add_dynamic_item(datasets, combined_pipeline)
    sb.dataio.dataset.add_dynamic_item([valid_data, test_data], combined_pipeline)
    
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
        "target_start_frames",
        "target_end_frames",
        "canonical_start_frames",
        "canonical_end_frames",
        "mispro_label",  # mispronunciation label
        "mispro_label_framewise",  # frame-wise mispronunciation label
        "phn_encoded_target_bin",  # binarized target for CTC loss
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
        "target_start_frames",
        "target_end_frames",
        "canonical_start_frames",
        "canonical_end_frames",
        "mispro_label",  # mispronunciation label
        "mispro_label_framewise",  # frame-wise mispronunciation label, 20ms htopsize
        "phn_encoded_target_bin",  # binarized target for CTC loss
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

    # train_data, valid_data, test_data, label_encoder = dataio_prep_for_llm(hparams)
    train_data, valid_data, test_data, label_encoder = dataio_prep_for_llm_with_timestamps(hparams)
    import pdb; pdb.set_trace()
    # shrink the datasets to a smaller size for debugging
    # if hparams["debug"]:
    #     train_data = train_data.filtered_sorted()[:int(len(train_data) * 0.1)]
    #     valid_data = valid_data.filtered_sorted()[:int(len(valid_data) * 0.1)]
    #     test_data = test_data.filtered_sorted()[:int(len(test_data) * 0.1)]
    #     logger.info(f"Debug mode: using {hparams['debug_shard_size']} samples for training, validation and testing.")
    # import pdb; pdb.set_trace()
    asr_brain_vars = {
        "mono_ssl_model": PhnMonoSSLModel,
        "mono_ssl_model_misproBCE": PhnMonoSSLModel_misproBCE,
        "dual_ssl_model": PhnDualSSLModel,
        # "dual_ssl_model_with_simple_residual": PhnDualSSLModel_with_SimpleResidual,
    }
    
    if hparams["feature_fusion"] == "mono":
        asr_brain_class = PhnMonoSSLModel
    elif hparams["feature_fusion"] == "mono_misproBCE":
        asr_brain_class = PhnMonoSSLModel_misproBCE
    elif hparams["feature_fusion"] == "mono_att_MHA":
        asr_brain_class = PhnMonoSSLModel_withcanoPhnEmb_HMA_CTC
    elif hparams["feature_fusion"] == "mono_att_HMA_ctc_to_canonical":
        asr_brain_class = HMA_attn_ctc_to_canonical
    elif hparams["feature_fusion"] == "mono_att_MHA_guided_attn":  
        asr_brain_class = PhnMonoSSLModel_withcanoPhnEmb_MHA_Guided_Attention_CTC
    elif hparams["feature_fusion"] == "HMA_attn_ctc_to_mispro":
        asr_brain_class = HMA_attn_ctc_to_mispro
    elif hparams["feature_fusion"] == "HMA_attn_ctc_to_mispro_ver2":
        asr_brain_class = HMA_attn_ctc_to_mispro_ver2
    elif hparams["feature_fusion"] == "HMA_attn_ctc_to_mispro_ver2_1":
        # Change Transformer Decoder to MHA decoder
        asr_brain_class = HMA_attn_ctc_to_mispro_ver2_1
    elif hparams["feature_fusion"] == "HMA_attn_ctc_to_mispro_ver2_1_perceived":
        # Change Transformer Decoder to MHA decoder, and use perceived phoneme embeddings
        asr_brain_class = HMA_attn_ctc_to_mispro_ver2_1_perceived
    elif hparams["feature_fusion"] == "HMA_attn_ctc_to_mispro_ver2_2":
        # Change Transformer Decoder to MHA decoder, and use mispronunciation BCE loss
        asr_brain_class = HMA_attn_ctc_to_mispro_ver2_2
    elif hparams["feature_fusion"] == "PGMDD":
        from models.phn_mono_ssl_model import PGMDD
        asr_brain_class = PGMDD
    elif hparams["feature_fusion"] == "hybrid_ctc_attention":
        asr_brain_class = Hybrid_CTC_Attention
    elif hparams["feature_fusion"] == "dual_ssl_enc":
        asr_brain_class = PhnDualSSLModel
    elif hparams["feature_fusion"] == "dual_ssl_enc_with_simple_residual":
        asr_brain_class = PhnDualSSLModel_with_SimpleResidual
    elif hparams["feature_fusion"] == "dual_ssl_enc_hybrid_ctc_attention":
        asr_brain_class = PhnDualSSLModel_Hybrid_CTC_Attention
    logger.info(f"Using ASR brain class: {asr_brain_class.__name__}")
    
    asr_brain = asr_brain_class(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    asr_brain.label_encoder = label_encoder

    from pathlib import Path
    # wandb init group by hparams perceived_ssl_model, canonical_ssl_model, feature_fusion
    
    perceived_ssl_model = hparams.get("perceived_ssl_model", "Null")
    canonical_ssl_model = hparams.get("canonical_ssl_model", "Null")    
    feature_fusion = hparams.get("feature_fusion", "Null")
    # use the asr_brain's type as model name
    model_type = type(asr_brain).__name__  # e.g., ASR_with_misproBCE_proj
    # use stem of model_type 
    model_stem = Path(model_type).stem 

    run_id = time.strftime("%Y%m%d-%H%M%S") 
    run_name = f"{perceived_ssl_model}_{canonical_ssl_model}_{feature_fusion}_{model_stem}"
    # if overrides.is given append its values to run_name
    if isinstance(overrides, dict):
        overrides = [f"{k}={v}" for k, v in overrides.items()]
        run_name += "_" + "_".join(overrides)
    
    run_id = f"{run_name}_{run_id}"
    # wandb init group by hparams perceived_ssl_model, canonical_ssl_model, feature_fusion
    wandb.init(
        project=hparams.get("wandb_project", "mdd-v5"), 
        name=run_name,
        id=run_id,
        resume="allow"
    )
    
    # Training/validation loop
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
