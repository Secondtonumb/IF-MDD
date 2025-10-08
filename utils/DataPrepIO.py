import os
import sys
import torch
import logging
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
# from mpd_eval_v3 import MpdStats
from mpd_eval_v4 import MpdStats
import librosa


import torchaudio


class BaseDataIOPrep:
    """Base class for data IO preparation."""
    
    def __init__(self, hparams):
        self.hparams = hparams
        self.data_folder = hparams["data_folder_save"]
        self.label_encoder = sb.dataio.encoder.CTCTextEncoder()
        
    def _prepare_datasets(self):
        """Prepare train, valid, and test datasets with sorting."""
        # 1. Declarations:
        train_data = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=self.hparams["train_annotation"],
            replacements={"data_root": self.data_folder},
        )
        
        # Apply sorting
        if self.hparams["sorting"] == "ascending":
            train_data = train_data.filtered_sorted(sort_key="duration")
            self.hparams["train_dataloader_opts"]["shuffle"] = False
        elif self.hparams["sorting"] == "descending":
            train_data = train_data.filtered_sorted(sort_key="duration", reverse=True)
            self.hparams["train_dataloader_opts"]["shuffle"] = False
        elif self.hparams["sorting"] == "random":
            pass
        else:
            raise NotImplementedError("sorting must be random, ascending or descending")

        valid_data = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=self.hparams["valid_annotation"],
            replacements={"data_root": self.data_folder},
        )
        valid_data = valid_data.filtered_sorted(sort_key="duration")

        test_data = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=self.hparams["test_annotation"],
            replacements={"data_root": self.data_folder},
        )
        test_data = test_data.filtered_sorted(sort_key="duration")
        
        return train_data, valid_data, test_data
    
    def _create_audio_pipeline(self):
        """Create audio processing pipeline."""
        @sb.utils.data_pipeline.takes("wav")
        @sb.utils.data_pipeline.provides("sig")
        def audio_pipeline(wav):
            # Load waveform and resample if needed
            waveform, sr = torchaudio.load(wav)  # waveform: [1, T]

            # Optional: resample to match model sample rate
            target_sr = self.hparams["sample_rate"]
            if sr != target_sr:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
                waveform = resampler(waveform)

            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Apply feature extractor (expecting 1D numpy array)
            sig = self.hparams["perceived_ssl"].feature_extractor(
                waveform.squeeze(0).numpy(),  # convert to 1D numpy
                sampling_rate=target_sr
            ).input_values[0]

            sig = torch.Tensor(sig)
            return sig
        
        return audio_pipeline
    
    def _setup_label_encoder(self, datasets):
        """Setup label encoder."""
        lab_enc_file = os.path.join(self.hparams["save_folder"], "label_encoder.txt")
        self.label_encoder.insert_bos_eos(
            bos_label="<bos>",
            eos_label="<eos>",
            bos_index=42,
            eos_index=43,
        )
        special_labels = {
            "blank_label": self.hparams["blank_index"],
        }
        self.label_encoder.load_or_create(
            path=lab_enc_file,
            from_didatasets=[datasets[0]],  # train_data
            output_key="phn_list_target",
            special_labels=special_labels,
            sequence_input=True,
        )
    
    def prepare(self):
        """Main method to prepare datasets. Should be overridden by subclasses."""
        raise NotImplementedError("Subclasses must implement prepare method")

class BasicDataIOPrep(BaseDataIOPrep):
    """Basic data IO preparation for simple ASR."""
    
    def _create_text_pipelines(self):
        """Create text processing pipelines."""
        @sb.utils.data_pipeline.takes("perceived_train_target")
        @sb.utils.data_pipeline.provides(
            "phn_list_target",
            "phn_encoded_list_target",
            "phn_encoded_target",
        )
        def text_pipeline_train(phn):
            phn_list = phn.strip().split()
            yield phn_list
            phn_encoded_list = self.label_encoder.encode_sequence(phn_list)
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
            phn_encoded_list_target = self.label_encoder.encode_sequence(phn_list_target)
            yield phn_encoded_list_target
            phn_encoded_target = torch.LongTensor(phn_encoded_list_target)
            yield phn_encoded_target
            phn_list_canonical = canonical.strip().split()
            yield phn_list_canonical
            phn_encoded_list_canonical = self.label_encoder.encode_sequence(phn_list_canonical)
            yield phn_encoded_list_canonical
            phn_encoded_canonical = torch.LongTensor(phn_encoded_list_canonical)
            yield phn_encoded_canonical
            phn_list_perceived = perceived.strip().split()
            yield phn_list_perceived
            phn_encoded_list_perceived = self.label_encoder.encode_sequence(phn_list_perceived)
            yield phn_encoded_list_perceived
            phn_encoded_perceived = torch.LongTensor(phn_encoded_list_perceived)
            yield phn_encoded_perceived

        return text_pipeline_train, text_pipeline_test
    
    def prepare(self):
        """Prepare datasets for basic ASR."""
        train_data, valid_data, test_data = self._prepare_datasets()
        datasets = [train_data, valid_data, test_data]
        
        # Add audio pipeline
        audio_pipeline = self._create_audio_pipeline()
        sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)
        
        # Add text pipelines
        text_pipeline_train, text_pipeline_test = self._create_text_pipelines()
        sb.dataio.dataset.add_dynamic_item([train_data], text_pipeline_train)
        sb.dataio.dataset.add_dynamic_item([valid_data, test_data], text_pipeline_test)

        # Setup label encoder
        self._setup_label_encoder(datasets)

        # Set output keys
        sb.dataio.dataset.set_output_keys(
            [train_data],
            ["id", "sig", "phn_encoded_target"],
        )
        sb.dataio.dataset.set_output_keys(
            [valid_data, test_data],
            ["id", "sig", "phn_encoded_target", "phn_encoded_canonical", "phn_encoded_perceived"],
        )

        return train_data, valid_data, test_data, self.label_encoder

class LLMDataIOPrep(BaseDataIOPrep):
    """Data IO preparation for LLM with mispronunciation labels."""
    
    def _create_text_pipelines(self):
        """Create text processing pipelines with mispronunciation labels."""
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
            
            "mispro_label",
        )
        def text_pipeline_test(target, canonical, perceived):
            phn_list_target = target.strip().split()
            yield phn_list_target
            phn_encoded_list_target = self.label_encoder.encode_sequence(phn_list_target)
            yield phn_encoded_list_target
            phn_encoded_target = torch.LongTensor(phn_encoded_list_target)
            yield phn_encoded_target
            
            phn_list_target_bos = ["<bos>"] + phn_list_target
            yield phn_list_target_bos
            phn_encoded_list_target_bos = self.label_encoder.encode_sequence(phn_list_target_bos)
            yield phn_encoded_list_target_bos
            phn_encoded_target_bos = torch.LongTensor(phn_encoded_list_target_bos)
            yield phn_encoded_target_bos
            
            phn_list_target_eos = phn_list_target + ["<eos>"]
            yield phn_list_target_eos
            phn_encoded_list_target_eos = self.label_encoder.encode_sequence(phn_list_target_eos)
            yield phn_encoded_list_target_eos
            phn_encoded_target_eos = torch.LongTensor(phn_encoded_list_target_eos)
            yield phn_encoded_target_eos
            

            phn_list_canonical = canonical.strip().split()
            yield phn_list_canonical
            phn_encoded_list_canonical = self.label_encoder.encode_sequence(phn_list_canonical)
            yield phn_encoded_list_canonical
            phn_encoded_canonical = torch.LongTensor(phn_encoded_list_canonical)
            yield phn_encoded_canonical
            
            phn_list_canonical_bos = ["<bos>"] + phn_list_canonical
            yield phn_list_canonical_bos
            phn_encoded_list_canonical_bos = self.label_encoder.encode_sequence(phn_list_canonical_bos)
            yield phn_encoded_list_canonical_bos
            phn_encoded_canonical_bos = torch.LongTensor(phn_encoded_list_canonical_bos)
            yield phn_encoded_canonical_bos

            phn_list_canonical_eos = phn_list_canonical + ["<eos>"]
            yield phn_list_canonical_eos
            phn_encoded_list_canonical_eos = self.label_encoder.encode_sequence(phn_list_canonical_eos)
            yield phn_encoded_list_canonical_eos
            phn_encoded_canonical_eos = torch.LongTensor(phn_encoded_list_canonical_eos)
            yield phn_encoded_canonical_eos
            
            phn_list_perceived = perceived.strip().split()
            yield phn_list_perceived
            phn_encoded_list_perceived = self.label_encoder.encode_sequence(phn_list_perceived)
            yield phn_encoded_list_perceived
            phn_encoded_perceived = torch.LongTensor(phn_encoded_list_perceived)
            yield phn_encoded_perceived
            
            phn_list_perceived_bos = ["<bos>"] + phn_list_perceived
            yield phn_list_perceived_bos
            phn_encoded_list_perceived_bos = self.label_encoder.encode_sequence(phn_list_perceived_bos)
            yield phn_encoded_list_perceived_bos
            phn_encoded_perceived_bos = torch.LongTensor(phn_encoded_list_perceived_bos)
            yield phn_encoded_perceived_bos

            phn_list_perceived_eos = phn_list_perceived + ["<eos>"]
            yield phn_list_perceived_eos
            phn_encoded_list_perceived_eos = self.label_encoder.encode_sequence(phn_list_perceived_eos)
            yield phn_encoded_list_perceived_eos
            phn_encoded_perceived_eos = torch.LongTensor(phn_encoded_list_perceived_eos)
            yield phn_encoded_perceived_eos

            mispro_label = [1 if p != c else 0 for p, c in zip(phn_list_perceived, phn_list_canonical)]
            mispro_label = torch.LongTensor(mispro_label)
        
            yield mispro_label

        return text_pipeline_test
    
    def prepare(self):
        """Prepare datasets for LLM with mispronunciation detection."""
        train_data, valid_data, test_data = self._prepare_datasets()
        datasets = [train_data, valid_data, test_data]
        
        # Add audio pipeline
        audio_pipeline = self._create_audio_pipeline()
        sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)
        
        # Add text pipeline
        text_pipeline_test = self._create_text_pipelines()
        sb.dataio.dataset.add_dynamic_item([train_data], text_pipeline_test)
        sb.dataio.dataset.add_dynamic_item([valid_data, test_data], text_pipeline_test)

        # Setup label encoder
        self._setup_label_encoder(datasets)

        # Set output keys
        output_keys = [
            "id", "sig", "phn_encoded_target", "phn_encoded_canonical", "phn_encoded_perceived",
            "phn_list_target", "phn_list_canonical", "phn_list_perceived", 
            "phn_list_target_bos", "phn_list_target_eos",
            "phn_list_canonical_bos", "phn_list_canonical_eos",
            "phn_list_perceived_bos", "phn_list_perceived_eos",
            "phn_encoded_target_bos", "phn_encoded_target_eos",
            "phn_encoded_canonical_bos", "phn_encoded_canonical_eos",
            "phn_encoded_perceived_bos", "phn_encoded_perceived_eos",
            "wrd", "mispro_label"
        ]
        sb.dataio.dataset.set_output_keys([train_data], output_keys)
        sb.dataio.dataset.set_output_keys([valid_data, test_data], output_keys)

        return train_data, valid_data, test_data, self.label_encoder

class LLMDataIOPrep_ver2(LLMDataIOPrep):
    # Allow mispro label in various types, 0=correct, 1=substitution, 2=deletion, 3=insertion
    # if phn_list_canonical != phn_list_perceived:
    # #     if len(phn_list_canonical) != sil and len(phn_list_perceived) != sil: substitution
    # # else if len(phn_list_canonical) ==sil and len(phn_list_perceived) != sil: insertion
    # # else if len(phn_list_canonical) != sil and len(phn_list_perceived) == sil: deletion
    def _create_text_pipelines(self):
        """Create text processing pipelines with mispronunciation labels."""
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
            
            "mispro_label",
        )
        def text_pipeline_test(target, canonical, perceived):
            phn_list_target = target.strip().split()
            yield phn_list_target
            phn_encoded_list_target = self.label_encoder.encode_sequence(phn_list_target)
            yield phn_encoded_list_target
            phn_encoded_target = torch.LongTensor(phn_encoded_list_target)
            yield phn_encoded_target
            
            phn_list_target_bos = ["<bos>"] + phn_list_target
            yield phn_list_target_bos
            phn_encoded_list_target_bos = self.label_encoder.encode_sequence(phn_list_target_bos)
            yield phn_encoded_list_target_bos
            phn_encoded_target_bos = torch.LongTensor(phn_encoded_list_target_bos)
            yield phn_encoded_target_bos
            
            phn_list_target_eos = phn_list_target + ["<eos>"]
            yield phn_list_target_eos
            phn_encoded_list_target_eos = self.label_encoder.encode_sequence(phn_list_target_eos)
            yield phn_encoded_list_target_eos
            phn_encoded_target_eos = torch.LongTensor(phn_encoded_list_target_eos)
            yield phn_encoded_target_eos
            

            phn_list_canonical = canonical.strip().split()
            yield phn_list_canonical
            phn_encoded_list_canonical = self.label_encoder.encode_sequence(phn_list_canonical)
            yield phn_encoded_list_canonical
            phn_encoded_canonical = torch.LongTensor(phn_encoded_list_canonical)
            yield phn_encoded_canonical
            
            phn_list_canonical_bos = ["<bos>"] + phn_list_canonical
            yield phn_list_canonical_bos
            phn_encoded_list_canonical_bos = self.label_encoder.encode_sequence(phn_list_canonical_bos)
            yield phn_encoded_list_canonical_bos
            phn_encoded_canonical_bos = torch.LongTensor(phn_encoded_list_canonical_bos)
            yield phn_encoded_canonical_bos

            phn_list_canonical_eos = phn_list_canonical + ["<eos>"]
            yield phn_list_canonical_eos
            phn_encoded_list_canonical_eos = self.label_encoder.encode_sequence(phn_list_canonical_eos)
            yield phn_encoded_list_canonical_eos
            phn_encoded_canonical_eos = torch.LongTensor(phn_encoded_list_canonical_eos)
            yield phn_encoded_canonical_eos
            
            phn_list_perceived = perceived.strip().split()
            yield phn_list_perceived
            phn_encoded_list_perceived = self.label_encoder.encode_sequence(phn_list_perceived)
            yield phn_encoded_list_perceived
            phn_encoded_perceived = torch.LongTensor(phn_encoded_list_perceived)
            yield phn_encoded_perceived
            
            phn_list_perceived_bos = ["<bos>"] + phn_list_perceived
            yield phn_list_perceived_bos
            phn_encoded_list_perceived_bos = self.label_encoder.encode_sequence(phn_list_perceived_bos)
            yield phn_encoded_list_perceived_bos
            phn_encoded_perceived_bos = torch.LongTensor(phn_encoded_list_perceived_bos)
            yield phn_encoded_perceived_bos

            phn_list_perceived_eos = phn_list_perceived + ["<eos>"]
            yield phn_list_perceived_eos
            phn_encoded_list_perceived_eos = self.label_encoder.encode_sequence(phn_list_perceived_eos)
            yield phn_encoded_list_perceived_eos
            phn_encoded_perceived_eos = torch.LongTensor(phn_encoded_list_perceived_eos)
            yield phn_encoded_perceived_eos

            # mispro_label = [1 if p != c else 0 for p, c in zip(phn_list_perceived, phn_list_canonical)]
            mispro_label = []
            for p, c in zip(phn_list_perceived, phn_list_canonical):
                if p != c:
                    if p == "sil" and c != "sil":
                        mispro_label.append(3)  # insertion
                    elif p != "sil" and c == "sil":
                        mispro_label.append(2)  # deletion
                    elif p != "sil" and c != "sil":
                        mispro_label.append(1)  # substitution
                    else:
                        raise ValueError("Unexpected case in mispronunciation labeling")
                else:
                    mispro_label.append(0)  # correct
                        
            mispro_label = torch.LongTensor(mispro_label)
        
            yield mispro_label

        return text_pipeline_test
    
class TimestampDataIOPrep(BaseDataIOPrep):
    """Data IO preparation with timestamp information."""
    
    def _create_combined_pipeline(self):
        """Create combined audio and text pipeline with timestamp processing."""
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
            "mispro_label_framewise",
            "phn_encoded_target_bin",
        )
        def combined_pipeline(wav, target_starts, target_ends, canonical_starts, canonical_ends, target, canonical, perceived):
            # Audio processing
            waveform, sr = torchaudio.load(wav)
            
            target_sr = self.hparams["sample_rate"]
            if sr != target_sr:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
                waveform = resampler(waveform)

            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            sig = self.hparams["perceived_ssl"].feature_extractor(
                waveform.squeeze(0).numpy(),
                sampling_rate=target_sr
            ).input_values[0]
            sig = torch.Tensor(sig)

            # Convert timestamps to frame indices
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
            phn_encoded_list_target = self.label_encoder.encode_sequence(phn_list_target)
            yield phn_encoded_list_target
            phn_encoded_target = torch.LongTensor(phn_encoded_list_target)
            yield phn_encoded_target

            phn_list_canonical = canonical.strip().split()
            yield phn_list_canonical
            phn_encoded_list_canonical = self.label_encoder.encode_sequence(phn_list_canonical)
            yield phn_encoded_list_canonical
            phn_encoded_canonical = torch.LongTensor(phn_encoded_list_canonical)
            yield phn_encoded_canonical

            phn_list_perceived = perceived.strip().split()
            yield phn_list_perceived
            phn_encoded_list_perceived = self.label_encoder.encode_sequence(phn_list_perceived)
            yield phn_encoded_list_perceived
            phn_encoded_perceived = torch.LongTensor(phn_encoded_list_perceived)
            yield phn_encoded_perceived
    
            mispro_label = [1 if p != c else 0 for p, c in zip(phn_list_perceived, phn_list_canonical)]
            mispro_label = torch.LongTensor(mispro_label)
            yield mispro_label
            
            # Frame-wise mispronunciation labels
            mispro_label_framewise = torch.zeros(len(sig), dtype=torch.long)
            canonical_starts_bins = [int(x * sr) for x in canonical_starts]
            canonical_ends_bins = [int(x * sr) for x in canonical_ends]
            for flag, start, end in zip(mispro_label, canonical_starts_bins, canonical_ends_bins):
                if flag == 1:
                    mispro_label_framewise[start:end] = 1
            yield torch.LongTensor(mispro_label_framewise)
            
            # Binarized target for CTC
            phn_encoded_target_bin = torch.zeros(len(sig), dtype=torch.long)
            for phn, start, end in zip(phn_encoded_target, canonical_starts_bins, canonical_ends_bins):
                phn_encoded_target_bin[start:end] = phn
            yield torch.LongTensor(phn_encoded_target_bin)

        return combined_pipeline
    
    def prepare(self):
        """Prepare datasets with timestamp information."""
        train_data, valid_data, test_data = self._prepare_datasets()
        datasets = [train_data, valid_data, test_data]
        
        # Add combined pipeline
        combined_pipeline = self._create_combined_pipeline()
        sb.dataio.dataset.add_dynamic_item(datasets, combined_pipeline)

        # Setup label encoder
        self._setup_label_encoder(datasets)

        # Set output keys
        output_keys = [
            "id", "sig", "phn_encoded_target", "phn_encoded_canonical", "phn_encoded_perceived",
            "phn_list_target", "phn_list_canonical", "phn_list_perceived", "wrd",
            "target_start_frames", "target_end_frames", "canonical_start_frames", "canonical_end_frames",
            "mispro_label", "mispro_label_framewise", "phn_encoded_target_bin",
        ]
        sb.dataio.dataset.set_output_keys([train_data], output_keys)
        sb.dataio.dataset.set_output_keys([valid_data, test_data], output_keys)

        return train_data, valid_data, test_data, self.label_encoder

class TimestampDataIOPrepforHybridCTCAttn(TimestampDataIOPrep):
    """Data IO preparation with timestamp information and Extend with <bos> <eos>."""
    def __init__(self, hparams):
        super().__init__(hparams)
        self.hparams = hparams
        self.label_encoder = sb.dataio.encoder.CTCTextEncoder()
        
    def _create_combined_pipeline(self):
        super_combined_pipeline = super()._create_combined_pipeline()
        """Create combined audio and text pipeline with G2P processing."""
        
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
            
            "mispro_label",
            "mispro_label_bos",
            "mispro_label_eos",
            "mispro_label_framewise",
            "phn_encoded_target_bin",
        )
        def combined_pipeline(wav, target_starts, target_ends, canonical_starts, canonical_ends, target, canonical, perceived):
            # Audio processing
            waveform, sr = torchaudio.load(wav)
            
            target_sr = self.hparams["sample_rate"]
            if sr != target_sr:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
                waveform = resampler(waveform)

            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            sig = self.hparams["perceived_ssl"].feature_extractor(
                waveform.squeeze(0).numpy(),
                sampling_rate=target_sr
            ).input_values[0]
            sig = torch.Tensor(sig)

            # Convert timestamps to frame indices
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
            phn_encoded_list_target = self.label_encoder.encode_sequence(phn_list_target)
            yield phn_encoded_list_target
            phn_encoded_target = torch.LongTensor(phn_encoded_list_target)
            yield phn_encoded_target
            
            # for expd target
            phn_list_target_bos = ["<bos>"] + phn_list_target
            yield phn_list_target_bos
            phn_encoded_list_target_bos = self.label_encoder.encode_sequence(phn_list_target_bos)
            yield phn_encoded_list_target_bos
            phn_encoded_target_bos = torch.LongTensor(phn_encoded_list_target_bos)
            yield phn_encoded_target_bos
            
            phn_list_target_eos = phn_list_target + ["<eos>"]
            yield phn_list_target_eos
            phn_encoded_list_target_eos = self.label_encoder.encode_sequence(phn_list_target_eos)
            yield phn_encoded_list_target_eos
            phn_encoded_target_eos = torch.LongTensor(phn_encoded_list_target_eos)
            yield phn_encoded_target_eos

            phn_list_canonical = canonical.strip().split()
            yield phn_list_canonical
            phn_encoded_list_canonical = self.label_encoder.encode_sequence(phn_list_canonical)
            yield phn_encoded_list_canonical
            phn_encoded_canonical = torch.LongTensor(phn_encoded_list_canonical)
            yield phn_encoded_canonical
            
            # for expd canonical
            phn_list_canonical_bos = ["<bos>"] + phn_list_canonical
            yield phn_list_canonical_bos
            phn_encoded_list_canonical_bos = self.label_encoder.encode_sequence(phn_list_canonical_bos)
            yield phn_encoded_list_canonical_bos
            phn_encoded_canonical_bos = torch.LongTensor(phn_encoded_list_canonical_bos)
            yield phn_encoded_canonical_bos
            phn_list_canonical_eos = phn_list_canonical + ["<eos>"]
            yield phn_list_canonical_eos
            phn_encoded_list_canonical_eos = self.label_encoder.encode_sequence(phn_list_canonical_eos)
            yield phn_encoded_list_canonical_eos
            phn_encoded_canonical_eos = torch.LongTensor(phn_encoded_list_canonical_eos)
            yield phn_encoded_canonical_eos

            phn_list_perceived = perceived.strip().split()
            yield phn_list_perceived
            phn_encoded_list_perceived = self.label_encoder.encode_sequence(phn_list_perceived)
            yield phn_encoded_list_perceived
            phn_encoded_perceived = torch.LongTensor(phn_encoded_list_perceived)
            yield phn_encoded_perceived

            # for expd perceived
            phn_list_perceived_bos = ["<bos>"] + phn_list_perceived
            yield phn_list_perceived_bos
            phn_encoded_list_perceived_bos = self.label_encoder.encode_sequence(phn_list_perceived_bos)
            yield phn_encoded_list_perceived_bos
            phn_encoded_perceived_bos = torch.LongTensor(phn_encoded_list_perceived_bos)
            yield phn_encoded_perceived_bos
            phn_list_perceived_eos = phn_list_perceived + ["<eos>"]
            yield phn_list_perceived_eos
            phn_encoded_list_perceived_eos = self.label_encoder.encode_sequence(phn_list_perceived_eos)
            yield phn_encoded_list_perceived_eos
            phn_encoded_perceived_eos = torch.LongTensor(phn_encoded_list_perceived_eos)
            yield phn_encoded_perceived_eos

            
            mispro_label = [1 if p != c else 0 for p, c in zip(phn_list_perceived, phn_list_canonical)]
            # append a dummy phn for eos
            mispro_label_bos = [0] + mispro_label
            mispro_label_eos = mispro_label + [0]  # append a dummy for eos
            mispro_label = torch.LongTensor(mispro_label)
            mispro_label_bos = torch.LongTensor(mispro_label_bos)
            mispro_label_eos = torch.LongTensor(mispro_label_eos)


            yield mispro_label
            yield mispro_label_bos
            yield mispro_label_eos
            
            # Frame-wise mispronunciation labels
            mispro_label_framewise = torch.zeros(len(sig), dtype=torch.long)
            canonical_starts_bins = [int(x * sr) for x in canonical_starts]
            canonical_ends_bins = [int(x * sr) for x in canonical_ends]
            for flag, start, end in zip(mispro_label, canonical_starts_bins, canonical_ends_bins):
                if flag == 1:
                    mispro_label_framewise[start:end] = 1
            yield torch.LongTensor(mispro_label_framewise)
            
            # Binarized target for CTC
            phn_encoded_target_bin = torch.zeros(len(sig), dtype=torch.long)
            for phn, start, end in zip(phn_encoded_target, canonical_starts_bins, canonical_ends_bins):
                phn_encoded_target_bin[start:end] = phn
            yield torch.LongTensor(phn_encoded_target_bin)

        return combined_pipeline

    def _setup_label_encoder(self, datasets):
        """Setup label encoder."""
        lab_enc_file = os.path.join(self.hparams["save_folder"], "label_encoder.txt")
        print("Loading or creating label encoder from file:", lab_enc_file)
        special_labels = {
            "blank_label": getattr(self.hparams, "blank_index", 0),
        }
            # "bos_label": getattr(self.hparams, "bos_label", 1),
            # "eos_label": getattr(self.hparams, "eos_label", 2),
        self.label_encoder.load_or_create(
            path=lab_enc_file,
            from_didatasets=[datasets[0]],  # train_data
            output_key="phn_list_target",
            special_labels=special_labels,
            sequence_input=True,
        )
    
    
    def prepare(self):
        """Prepare datasets with timestamp information."""
        train_data, valid_data, test_data = self._prepare_datasets()
        datasets = [train_data, valid_data, test_data]
        
        # Add combined pipeline
        combined_pipeline = self._create_combined_pipeline()
        sb.dataio.dataset.add_dynamic_item(datasets, combined_pipeline)

        # Setup label encoder
        self._setup_label_encoder(datasets)
        # self.label_encoder.add_bos_eos("<bos>", "<eos>")
        max_label = max(self.label_encoder.lab2ind.values())
        try:
            self.label_encoder.insert_bos_eos(bos_label="<bos>", eos_label="<eos>",
                                          bos_index=max_label + 1, eos_index=max_label + 2,
                                          )
        except:
            print("eos added already")
        

        # Set output keys
        output_keys = [
            "id", "sig", 
            "phn_encoded_target", "phn_encoded_canonical", "phn_encoded_perceived",
            "phn_list_target", "phn_list_canonical", "phn_list_perceived", 
            "wrd",
            "target_start_frames", "target_end_frames", 
            "canonical_start_frames", "canonical_end_frames",
            "mispro_label", "mispro_label_bos", "mispro_label_eos", "mispro_label_framewise", "phn_encoded_target_bin",
            "phn_list_target_bos", "phn_encoded_list_target_bos", "phn_encoded_target_bos",
            "phn_list_target_eos", "phn_encoded_list_target_eos", "phn_encoded_target_eos",
            "phn_list_canonical_bos", "phn_encoded_list_canonical_bos", "phn_encoded_canonical_bos",
            "phn_list_canonical_eos", "phn_encoded_list_canonical_eos", "phn_encoded_canonical_eos",
            "phn_list_perceived_bos", "phn_encoded_list_perceived_bos", "phn_encoded_perceived_bos",
            "phn_list_perceived_eos", "phn_encoded_list_perceived_eos", "phn_encoded_perceived_eos",
        ]

        sb.dataio.dataset.set_output_keys([train_data], output_keys)
        sb.dataio.dataset.set_output_keys([valid_data, test_data], output_keys)

        return train_data, valid_data, test_data, self.label_encoder

class LLMDataIOPrep_ver3(LLMDataIOPrep):
    # Allow mispro label in various types, 0=correct, 1=substitution, 2=deletion, 3=insertion
    # if phn_list_canonical != phn_list_perceived:
    # #     if len(phn_list_canonical) != sil and len(phn_list_perceived) != sil: substitution
    # # else if len(phn_list_canonical) ==sil and len(phn_list_perceived) != sil: insertion
    # # else if len(phn_list_canonical) != sil and len(phn_list_perceived) == sil: deletion
    def _create_text_pipelines(self):
        """Create text processing pipelines with mispronunciation labels."""
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
            "phn_list_target_bos_eos",
            "phn_encoded_list_target_bos_eos",
            "phn_encoded_target_bos_eos",
            
            "phn_list_canonical",
            "phn_encoded_list_canonical",
            "phn_encoded_canonical",
            "phn_list_canonical_bos",
            "phn_encoded_list_canonical_bos",
            "phn_encoded_canonical_bos",
            "phn_list_canonical_eos",
            "phn_encoded_list_canonical_eos",
            "phn_encoded_canonical_eos",
            "phn_list_canonical_bos_eos",
            "phn_encoded_list_canonical_bos_eos",
            "phn_encoded_canonical_bos_eos",

            "phn_list_perceived",
            "phn_encoded_list_perceived",
            "phn_encoded_perceived",
            "phn_list_perceived_bos",
            "phn_encoded_list_perceived_bos",
            "phn_encoded_perceived_bos",
            "phn_list_perceived_eos",
            "phn_encoded_list_perceived_eos",
            "phn_encoded_perceived_eos",
            "phn_list_perceived_bos_eos",
            "phn_encoded_list_perceived_bos_eos",
            "phn_encoded_perceived_bos_eos",
            
            "mispro_label",
        )
        def text_pipeline_test(target, canonical, perceived):
            phn_list_target = target.strip().split()
            yield phn_list_target
            phn_encoded_list_target = self.label_encoder.encode_sequence(phn_list_target)
            yield phn_encoded_list_target
            phn_encoded_target = torch.LongTensor(phn_encoded_list_target)
            yield phn_encoded_target
            
            phn_list_target_bos = ["<bos>"] + phn_list_target
            yield phn_list_target_bos
            phn_encoded_list_target_bos = self.label_encoder.encode_sequence(phn_list_target_bos)
            yield phn_encoded_list_target_bos
            phn_encoded_target_bos = torch.LongTensor(phn_encoded_list_target_bos)
            yield phn_encoded_target_bos
            
            phn_list_target_eos = phn_list_target + ["<eos>"]
            yield phn_list_target_eos
            phn_encoded_list_target_eos = self.label_encoder.encode_sequence(phn_list_target_eos)
            yield phn_encoded_list_target_eos
            phn_encoded_target_eos = torch.LongTensor(phn_encoded_list_target_eos)
            yield phn_encoded_target_eos
            
            phn_list_target_bos_eos = ["<bos>"] + phn_list_target + ["<eos>"]
            yield phn_list_target_bos_eos
            phn_encoded_list_target_bos_eos = self.label_encoder.encode_sequence(phn_list_target_bos_eos)
            yield phn_encoded_list_target_bos_eos
            phn_encoded_target_bos_eos = torch.LongTensor(phn_encoded_list_target_bos_eos)
            yield phn_encoded_target_bos_eos
            
            phn_list_canonical = canonical.strip().split()
            yield phn_list_canonical
            phn_encoded_list_canonical = self.label_encoder.encode_sequence(phn_list_canonical)
            yield phn_encoded_list_canonical
            phn_encoded_canonical = torch.LongTensor(phn_encoded_list_canonical)
            yield phn_encoded_canonical
            
            phn_list_canonical_bos = ["<bos>"] + phn_list_canonical
            yield phn_list_canonical_bos
            phn_encoded_list_canonical_bos = self.label_encoder.encode_sequence(phn_list_canonical_bos)
            yield phn_encoded_list_canonical_bos
            phn_encoded_canonical_bos = torch.LongTensor(phn_encoded_list_canonical_bos)
            yield phn_encoded_canonical_bos

            phn_list_canonical_eos = phn_list_canonical + ["<eos>"]
            yield phn_list_canonical_eos
            phn_encoded_list_canonical_eos = self.label_encoder.encode_sequence(phn_list_canonical_eos)
            yield phn_encoded_list_canonical_eos
            phn_encoded_canonical_eos = torch.LongTensor(phn_encoded_list_canonical_eos)
            yield phn_encoded_canonical_eos
            
            phn_list_canonical_bos_eos = ["<bos>"] + phn_list_canonical + ["<eos>"]
            yield phn_list_canonical_bos_eos
            phn_encoded_list_canonical_bos_eos = self.label_encoder.encode_sequence(phn_list_canonical_bos_eos)
            yield phn_encoded_list_canonical_bos_eos
            phn_encoded_canonical_bos_eos = torch.LongTensor(phn_encoded_list_canonical_bos_eos)
            yield phn_encoded_canonical_bos_eos
            
            phn_list_perceived = perceived.strip().split()
            yield phn_list_perceived
            phn_encoded_list_perceived = self.label_encoder.encode_sequence(phn_list_perceived)
            yield phn_encoded_list_perceived
            phn_encoded_perceived = torch.LongTensor(phn_encoded_list_perceived)
            yield phn_encoded_perceived
            
            phn_list_perceived_bos = ["<bos>"] + phn_list_perceived
            yield phn_list_perceived_bos
            phn_encoded_list_perceived_bos = self.label_encoder.encode_sequence(phn_list_perceived_bos)
            yield phn_encoded_list_perceived_bos
            phn_encoded_perceived_bos = torch.LongTensor(phn_encoded_list_perceived_bos)
            yield phn_encoded_perceived_bos

            phn_list_perceived_eos = phn_list_perceived + ["<eos>"]
            yield phn_list_perceived_eos
            phn_encoded_list_perceived_eos = self.label_encoder.encode_sequence(phn_list_perceived_eos)
            yield phn_encoded_list_perceived_eos
            phn_encoded_perceived_eos = torch.LongTensor(phn_encoded_list_perceived_eos)
            yield phn_encoded_perceived_eos
            
            phn_list_perceived_bos_eos = ["<bos>"] + phn_list_perceived + ["<eos>"]
            yield phn_list_perceived_bos_eos
            phn_encoded_list_perceived_bos_eos = self.label_encoder.encode_sequence(phn_list_perceived_bos_eos)
            yield phn_encoded_list_perceived_bos_eos
            phn_encoded_perceived_bos_eos = torch.LongTensor(phn_encoded_list_perceived_bos_eos)
            yield phn_encoded_perceived_bos_eos
            

            # mispro_label = [1 if p != c else 0 for p, c in zip(phn_list_perceived, phn_list_canonical)]
            mispro_label = []
            for p, c in zip(phn_list_perceived, phn_list_canonical):
                if p != c:
                    if p == "sil" and c != "sil":
                        mispro_label.append(3)  # insertion
                    elif p != "sil" and c == "sil":
                        mispro_label.append(2)  # deletion
                    elif p != "sil" and c != "sil":
                        mispro_label.append(1)  # substitution
                    else:
                        raise ValueError("Unexpected case in mispronunciation labeling")
                else:
                    mispro_label.append(0)  # correct
                        
            mispro_label = torch.LongTensor(mispro_label)
        
            yield mispro_label

        return text_pipeline_test
    
    def prepare(self):
        """Prepare datasets for LLM with mispronunciation detection."""
        train_data, valid_data, test_data = self._prepare_datasets()
        datasets = [train_data, valid_data, test_data]
        
        # Add audio pipeline
        audio_pipeline = self._create_audio_pipeline()
        sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)
        
        # Add text pipeline
        text_pipeline_test = self._create_text_pipelines()
        sb.dataio.dataset.add_dynamic_item([train_data], text_pipeline_test)
        sb.dataio.dataset.add_dynamic_item([valid_data, test_data], text_pipeline_test)

        # Setup label encoder
        self._setup_label_encoder(datasets)

        # Set output keys
        output_keys = [
            "id", "sig", "phn_encoded_target", "phn_encoded_canonical", "phn_encoded_perceived",
            "phn_list_target", "phn_list_canonical", "phn_list_perceived", 
            "phn_list_target_bos", "phn_list_target_eos", "phn_list_target_bos_eos",
            "phn_list_canonical_bos", "phn_list_canonical_eos", "phn_list_canonical_bos_eos",
            "phn_list_perceived_bos", "phn_list_perceived_eos", "phn_list_perceived_bos_eos",
            "phn_encoded_target_bos", "phn_encoded_target_eos", "phn_encoded_target_bos_eos",
            "phn_encoded_canonical_bos", "phn_encoded_canonical_eos", "phn_encoded_canonical_bos_eos",
            "phn_encoded_perceived_bos", "phn_encoded_perceived_eos", "phn_encoded_perceived_bos_eos",
            "wrd", "mispro_label"
        ]
        sb.dataio.dataset.set_output_keys([train_data], output_keys)
        sb.dataio.dataset.set_output_keys([valid_data, test_data], output_keys)

        return train_data, valid_data, test_data, self.label_encoder