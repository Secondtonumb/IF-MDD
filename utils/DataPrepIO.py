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
        # Ensure save_folder exists
        save_folder = self.hparams["save_folder"]
        os.makedirs(save_folder, exist_ok=True)
        if self.hparams['lab_enc_file'] is None:
            # raise ValueError("Label encoder must support insert_bos_eos method")
            lab_enc_file = os.path.join(save_folder, "label_encoder.txt")
            # for L2_arctic only
            self.label_encoder.insert_bos_eos(
                bos_label="<bos>",
                eos_label="<eos>",
                bos_index=42,
                eos_index=43,
            )
        else:
            lab_enc_file = self.hparams['lab_enc_file']
            # copy lab_enc_file to current exp folder
            import shutil
            shutil.copy(lab_enc_file, os.path.join(save_folder, "label_encoder.txt"))
            lab_enc_file = os.path.join(save_folder, "label_encoder.txt")
        
        special_labels = {
            "blank_label": self.hparams["blank_index"],
        }
        
        # touch lab_enc_file if not exist
        
        # Load or create label encoder
        # import pdb; pdb.set_trace()
        self.label_encoder.load_or_create(
            path=lab_enc_file,
            from_didatasets=[datasets[0]], 
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

class LLMDataIOPrep_WordLevel(LLMDataIOPrep):
    """Data IO preparation for LLM with word-level scores."""
    
    def _create_text_pipelines(self):
        """Create text processing pipelines with mispronunciation labels and word-level information."""
        @sb.utils.data_pipeline.takes("perceived_train_target", "canonical_aligned", "perceived_aligned", "word_scores")
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
            "word_boundaries",
            "word_accuracy_scores",
            "word_stress_scores",
            "word_total_scores",
        )
        def text_pipeline_with_words(target, canonical, perceived, word_scores_list):
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
            
            # Process word-level scores
            if word_scores_list:
                word_boundaries = [(w['phone_start'], w['phone_end']) for w in word_scores_list]
                word_acc_scores = [float(w['accuracy']) for w in word_scores_list]
                word_stress_scores = [float(w['stress']) for w in word_scores_list]
                word_total_scores = [float(w['total']) for w in word_scores_list]
            else:
                # Fallback: treat entire utterance as one word
                word_boundaries = [(0, len(phn_list_canonical))]
                word_acc_scores = [10.0]
                word_stress_scores = [10.0]
                word_total_scores = [10.0]
            
            yield word_boundaries
            yield torch.FloatTensor(word_acc_scores)
            yield torch.FloatTensor(word_stress_scores)
            yield torch.FloatTensor(word_total_scores)

        return text_pipeline_with_words
    
    def prepare(self):
        """Prepare datasets for LLM with mispronunciation detection and word-level scores."""
        train_data, valid_data, test_data = self._prepare_datasets()
        datasets = [train_data, valid_data, test_data]
        
        # Add audio pipeline
        audio_pipeline = self._create_audio_pipeline()
        sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)
        
        # Add text pipeline with word-level info
        text_pipeline_with_words = self._create_text_pipelines()
        sb.dataio.dataset.add_dynamic_item([train_data], text_pipeline_with_words)
        sb.dataio.dataset.add_dynamic_item([valid_data, test_data], text_pipeline_with_words)

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
            "wrd", "mispro_label", 
            "word_boundaries", "word_accuracy_scores", "word_stress_scores", "word_total_scores"
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

class PhonemeFrameTimestampDataIOPrep(BaseDataIOPrep):
    """Data IO preparation that converts canonical phoneme timestamps (seconds) to frame indices.

    Expected JSON fields per utterance:
      - wav
      - canonical_aligned (space-separated phones)
      - perceived_aligned (space-separated phones)
      - perceived_train_target (space-separated phones)
      - canonical_starts: list[float] (seconds)
      - canonical_ends:   list[float] (seconds)

    Output dynamic items extend the basic phoneme lists with:
      - canonical_phone_start_frames: LongTensor[L]
      - canonical_phone_end_frames:   LongTensor[L]
      - canonical_phone_frame_ranges: List[(start_frame, end_frame)] length L
    Frame indices are computed with a hop length derived from sample_rate and a configurable
    milliseconds stride (default 20ms). Frames are clamped to valid range.
    """

    def __init__(self, hparams):
        super().__init__(hparams)
        self.frame_ms = hparams.get("frame_ms", 20.0)
        
    def _setup_label_encoder(self, datasets):
        """Setup label encoder with BOS/EOS tokens."""
        # Ensure save_folder exists
        save_folder = self.hparams["save_folder"]
        os.makedirs(save_folder, exist_ok=True)
        
        lab_enc_file = os.path.join(save_folder, "label_encoder.txt")
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

    def _create_pipeline(self):
        @sb.utils.data_pipeline.takes(
            "wav", "canonical_starts", "canonical_ends", "perceived_train_target", "canonical_aligned", "perceived_aligned"
        )
        @sb.utils.data_pipeline.provides(
            "sig",
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
            "canonical_phone_start_frames",
            "canonical_phone_end_frames",
            "canonical_phone_frame_ranges",
            "mispro_label",
        )
        def pipeline(wav, canonical_starts, canonical_ends, target, canonical, perceived):
            # Load & resample audio
            waveform, sr = torchaudio.load(wav)
            target_sr = self.hparams["sample_rate"]
            if sr != target_sr:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
                waveform = resampler(waveform)
                sr = target_sr
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            sig = self.hparams["perceived_ssl"].feature_extractor(
                waveform.squeeze(0).numpy(), sampling_rate=sr
            ).input_values[0]
            sig = torch.tensor(sig)
            yield sig

            # Text target (already preprocessed perceived_train_target)
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

            # Convert seconds -> frame indices
            starts_sec = canonical_starts if isinstance(canonical_starts, list) else []
            ends_sec = canonical_ends if isinstance(canonical_ends, list) else []
            L = len(phn_list_canonical)
            if len(starts_sec) != L or len(ends_sec) != L:
                # Fallback: approximate even segmentation
                duration_sec = len(sig) / sr
                seg_dur = duration_sec / max(L, 1)
                starts_sec = [i * seg_dur for i in range(L)]
                ends_sec = [(i + 1) * seg_dur for i in range(L)]

            hop_length_samples = int(sr * (self.frame_ms / 1000.0))
            if hop_length_samples <= 0:
                hop_length_samples = int(sr * 0.02)  # safety fallback 20ms

            total_samples = len(sig)
            max_frame = max(total_samples // hop_length_samples - 1, 0)

            def to_frame(t: float) -> int:
                frame = int(round((t * sr) / hop_length_samples))
                if frame < 0:
                    frame = 0
                if frame > max_frame:
                    frame = max_frame
                return frame

            start_frames = [to_frame(s) for s in starts_sec]
            end_frames = [max(to_frame(e), to_frame(s)) for s, e in zip(starts_sec, ends_sec)]

            yield torch.LongTensor(start_frames)
            yield torch.LongTensor(end_frames)
            yield [(int(s), int(e)) for s, e in zip(start_frames, end_frames)]

            mispro_label = [1 if p != c else 0 for p, c in zip(phn_list_perceived, phn_list_canonical)]
            yield torch.LongTensor(mispro_label)

        return pipeline

    def prepare(self):
        train_data, valid_data, test_data = self._prepare_datasets()
        datasets = [train_data, valid_data, test_data]

        pipeline = self._create_pipeline()
        sb.dataio.dataset.add_dynamic_item(datasets, pipeline)

        # Label encoder based on target list
        self._setup_label_encoder(datasets)

        output_keys = [
            "id", "sig",
            "phn_encoded_target", "phn_encoded_canonical", "phn_encoded_perceived",
            "phn_list_target", "phn_list_canonical", "phn_list_perceived",
            "phn_list_target_bos", "phn_list_target_eos",
            "phn_list_canonical_bos", "phn_list_canonical_eos",
            "phn_list_perceived_bos", "phn_list_perceived_eos",
            "phn_encoded_target_bos", "phn_encoded_target_eos",
            "phn_encoded_canonical_bos", "phn_encoded_canonical_eos",
            "phn_encoded_perceived_bos", "phn_encoded_perceived_eos",
            "canonical_phone_start_frames", "canonical_phone_end_frames", "canonical_phone_frame_ranges",
            "wrd", "mispro_label"
        ]
        sb.dataio.dataset.set_output_keys([train_data], output_keys)
        sb.dataio.dataset.set_output_keys([valid_data, test_data], output_keys)
        return train_data, valid_data, test_data, self.label_encoder


class MFATimestampDataIOPrep(BaseDataIOPrep):
    """Data IO preparation with MFA word-level and phone-level timestamps.
    
    This class handles datasets with Montreal Forced Aligner (MFA) outputs,
    providing both word-level and phone-level timestamp information.
    
    Expected JSON fields per utterance:
      - wav: audio file path
      - canonical_aligned: space-separated phones
      - perceived_aligned: space-separated phones  
      - perceived_train_target: space-separated phones
      - mfa_phone_aligned: space-separated phones from MFA
      - mfa_phone_starts: list[float] phone start times in seconds
      - mfa_phone_ends: list[float] phone end times in seconds
      - mfa_word_aligned: space-separated words from MFA
      - mfa_word_starts: list[float] word start times in seconds
      - mfa_word_ends: list[float] word end times in seconds
      - mfa_word_phone_start_idx: list[int] phone index where each word starts
      - mfa_word_phone_end_idx: list[int] phone index where each word ends
      - word_scores: list[dict] word-level pronunciation scores (optional)
    
    Output dynamic items include:
      - Phone-level: phn_list_*, phn_encoded_*, mfa_phone_start_frames, mfa_phone_end_frames
      - Word-level: mfa_word_list, mfa_word_start_frames, mfa_word_end_frames, 
                    mfa_word_phone_ranges
    """

    def __init__(self, hparams):
        super().__init__(hparams)
        self.frame_ms = hparams.get("frame_ms", 20.0)
        
    def _setup_label_encoder(self, datasets):
        """Setup label encoder with BOS/EOS tokens."""
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
            from_didatasets=[datasets[0]],
            output_key="phn_list_target",
            special_labels=special_labels,
            sequence_input=True,
        )

    def _create_pipeline(self):
        """Create pipeline with MFA word and phone level timestamps."""
        @sb.utils.data_pipeline.takes(
            "wav", 
            "perceived_train_target", 
            "canonical_aligned", 
            "perceived_aligned",
            "mfa_phone_aligned",
            "mfa_phone_starts", 
            "mfa_phone_ends",
            "mfa_word_aligned",
            "mfa_word_starts",
            "mfa_word_ends", 
            "mfa_word_phone_start_idx",
            "mfa_word_phone_end_idx",
        )
        @sb.utils.data_pipeline.provides(
            "sig",
            # Target phoneme sequence (for training)
            "phn_list_target",
            "phn_encoded_list_target",
            "phn_encoded_target",
            "phn_list_target_bos",
            "phn_encoded_list_target_bos",
            "phn_encoded_target_bos",
            "phn_list_target_eos",
            "phn_encoded_list_target_eos",
            "phn_encoded_target_eos",
            # Canonical phoneme sequence
            "phn_list_canonical",
            "phn_encoded_list_canonical",
            "phn_encoded_canonical",
            "phn_list_canonical_bos",
            "phn_encoded_list_canonical_bos",
            "phn_encoded_canonical_bos",
            "phn_list_canonical_eos",
            "phn_encoded_list_canonical_eos",
            "phn_encoded_canonical_eos",
            # Perceived phoneme sequence
            "phn_list_perceived",
            "phn_encoded_list_perceived",
            "phn_encoded_perceived",
            "phn_list_perceived_bos",
            "phn_encoded_list_perceived_bos",
            "phn_encoded_perceived_bos",
            "phn_list_perceived_eos",
            "phn_encoded_list_perceived_eos",
            "phn_encoded_perceived_eos",
            # MFA phone-level timestamps (in frames)
            "mfa_phone_list",
            "mfa_phone_start_frames",
            "mfa_phone_end_frames",
            "mfa_phone_frame_ranges",
            # MFA word-level timestamps (in frames)
            "mfa_word_list",
            "mfa_word_start_frames",
            "mfa_word_end_frames",
            "mfa_word_phone_ranges",
            # Mispronunciation labels
            "mispro_label",
        )
        def pipeline(
            wav, 
            perceived_train_target, 
            canonical_aligned, 
            perceived_aligned,
            mfa_phone_aligned,
            mfa_phone_starts,
            mfa_phone_ends,
            mfa_word_aligned,
            mfa_word_starts,
            mfa_word_ends,
            mfa_word_phone_start_idx,
            mfa_word_phone_end_idx,
        ):
            # Load audio
            sig = sb.dataio.dataio.read_audio(wav)
            
            # Get audio properties for frame calculation
            sample_rate = 16000  # Assuming 16kHz
            hop_length_samples = int((self.frame_ms / 1000.0) * sample_rate)
            total_samples = sig.shape[0]
            total_frames = (total_samples + hop_length_samples - 1) // hop_length_samples
            
            def time_to_frame(time_sec):
                """Convert time in seconds to frame index."""
                frame_idx = int(time_sec * sample_rate / hop_length_samples)
                return max(0, min(frame_idx, total_frames - 1))
            
            # Process target phonemes
            phn_list_target = perceived_train_target.strip().split()
            phn_encoded_list_target = self.label_encoder.encode_sequence(phn_list_target)
            phn_encoded_target = torch.LongTensor(phn_encoded_list_target)
            
            phn_list_target_bos = ["<bos>"] + phn_list_target
            phn_encoded_list_target_bos = self.label_encoder.encode_sequence(phn_list_target_bos)
            phn_encoded_target_bos = torch.LongTensor(phn_encoded_list_target_bos)
            
            phn_list_target_eos = phn_list_target + ["<eos>"]
            phn_encoded_list_target_eos = self.label_encoder.encode_sequence(phn_list_target_eos)
            phn_encoded_target_eos = torch.LongTensor(phn_encoded_list_target_eos)
            
            # Process canonical phonemes
            phn_list_canonical = canonical_aligned.strip().split()
            phn_encoded_list_canonical = self.label_encoder.encode_sequence(phn_list_canonical)
            phn_encoded_canonical = torch.LongTensor(phn_encoded_list_canonical)
            
            phn_list_canonical_bos = ["<bos>"] + phn_list_canonical
            phn_encoded_list_canonical_bos = self.label_encoder.encode_sequence(phn_list_canonical_bos)
            phn_encoded_canonical_bos = torch.LongTensor(phn_encoded_list_canonical_bos)
            
            phn_list_canonical_eos = phn_list_canonical + ["<eos>"]
            phn_encoded_list_canonical_eos = self.label_encoder.encode_sequence(phn_list_canonical_eos)
            phn_encoded_canonical_eos = torch.LongTensor(phn_encoded_list_canonical_eos)
            
            # Process perceived phonemes
            phn_list_perceived = perceived_aligned.strip().split()
            phn_encoded_list_perceived = self.label_encoder.encode_sequence(phn_list_perceived)
            phn_encoded_perceived = torch.LongTensor(phn_encoded_list_perceived)
            
            phn_list_perceived_bos = ["<bos>"] + phn_list_perceived
            phn_encoded_list_perceived_bos = self.label_encoder.encode_sequence(phn_list_perceived_bos)
            phn_encoded_perceived_bos = torch.LongTensor(phn_encoded_list_perceived_bos)
            
            phn_list_perceived_eos = phn_list_perceived + ["<eos>"]
            phn_encoded_list_perceived_eos = self.label_encoder.encode_sequence(phn_list_perceived_eos)
            phn_encoded_perceived_eos = torch.LongTensor(phn_encoded_list_perceived_eos)
            
            # Process MFA phone-level timestamps
            mfa_phone_list = mfa_phone_aligned.strip().split()
            mfa_phone_start_frames = torch.LongTensor([time_to_frame(t) for t in mfa_phone_starts])
            mfa_phone_end_frames = torch.LongTensor([time_to_frame(t) for t in mfa_phone_ends])
            mfa_phone_frame_ranges = [(s.item(), e.item()) for s, e in zip(mfa_phone_start_frames, mfa_phone_end_frames)]
            
            # Process MFA word-level timestamps
            mfa_word_list = mfa_word_aligned.strip().split()
            mfa_word_start_frames = torch.LongTensor([time_to_frame(t) for t in mfa_word_starts])
            mfa_word_end_frames = torch.LongTensor([time_to_frame(t) for t in mfa_word_ends])
            
            # Create word-phone range mapping
            mfa_word_phone_ranges = [
                (start_idx, end_idx) 
                for start_idx, end_idx in zip(mfa_word_phone_start_idx, mfa_word_phone_end_idx)
            ]
            
            # Compute mispronunciation labels
            mispro_label = torch.LongTensor([
                0 if c == p else 1 
                for c, p in zip(phn_list_canonical, phn_list_perceived)
            ])
            
            yield sig
            yield phn_list_target
            yield phn_encoded_list_target
            yield phn_encoded_target
            yield phn_list_target_bos
            yield phn_encoded_list_target_bos
            yield phn_encoded_target_bos
            yield phn_list_target_eos
            yield phn_encoded_list_target_eos
            yield phn_encoded_target_eos
            yield phn_list_canonical
            yield phn_encoded_list_canonical
            yield phn_encoded_canonical
            yield phn_list_canonical_bos
            yield phn_encoded_list_canonical_bos
            yield phn_encoded_canonical_bos
            yield phn_list_canonical_eos
            yield phn_encoded_list_canonical_eos
            yield phn_encoded_canonical_eos
            yield phn_list_perceived
            yield phn_encoded_list_perceived
            yield phn_encoded_perceived
            yield phn_list_perceived_bos
            yield phn_encoded_list_perceived_bos
            yield phn_encoded_perceived_bos
            yield phn_list_perceived_eos
            yield phn_encoded_list_perceived_eos
            yield phn_encoded_perceived_eos
            yield mfa_phone_list
            yield mfa_phone_start_frames
            yield mfa_phone_end_frames
            yield mfa_phone_frame_ranges
            yield mfa_word_list
            yield mfa_word_start_frames
            yield mfa_word_end_frames
            yield mfa_word_phone_ranges
            yield mispro_label

        return pipeline

    def prepare(self):
        """Prepare datasets with MFA word and phone level timestamps."""
        train_data, valid_data, test_data = self._prepare_datasets()
        datasets = [train_data, valid_data, test_data]

        # Add pipeline
        pipeline = self._create_pipeline()
        sb.dataio.dataset.add_dynamic_item(datasets, pipeline)

        # Setup label encoder
        self._setup_label_encoder(datasets)

        # Set output keys
        output_keys = [
            "id", "sig",
            "phn_encoded_target", "phn_encoded_canonical", "phn_encoded_perceived",
            "phn_list_target", "phn_list_canonical", "phn_list_perceived",
            "phn_list_target_bos", "phn_list_target_eos",
            "phn_list_canonical_bos", "phn_list_canonical_eos",
            "phn_list_perceived_bos", "phn_list_perceived_eos",
            "phn_encoded_target_bos", "phn_encoded_target_eos",
            "phn_encoded_canonical_bos", "phn_encoded_canonical_eos",
            "phn_encoded_perceived_bos", "phn_encoded_perceived_eos",
            "mfa_phone_list", "mfa_phone_start_frames", "mfa_phone_end_frames", "mfa_phone_frame_ranges",
            "mfa_word_list", "mfa_word_start_frames", "mfa_word_end_frames", "mfa_word_phone_ranges",
            "wrd", "mispro_label"
        ]
        sb.dataio.dataset.set_output_keys([train_data], output_keys)
        sb.dataio.dataset.set_output_keys([valid_data, test_data], output_keys)
        
        return train_data, valid_data, test_data, self.label_encoder


class ComprehensiveDataIOPrep(BaseDataIOPrep):
    """Comprehensive data IO preparation that combines all features from multiple DataPrep classes.
    
    This unified class integrates:
    - MFATimestampDataIOPrep: MFA word & phone level timestamps
    - LLMDataIOPrep_WordLevel: Word-level pronunciation scores
    - PhonemeFrameTimestampDataIOPrep: Canonical phone timestamps
    
    Expected JSON fields per utterance:
      - wav: audio file path
      - canonical_aligned, perceived_aligned, perceived_train_target: space-separated phones
      - wrd: text transcription
      
      # MFA outputs (from MFATimestampDataIOPrep)
      - mfa_phone_aligned, mfa_phone_starts, mfa_phone_ends
      - mfa_word_aligned, mfa_word_starts, mfa_word_ends
      - mfa_word_phone_start_idx, mfa_word_phone_end_idx
      
      # Word scores (from LLMDataIOPrep_WordLevel)
      - word_scores: list[dict] with accuracy/stress/total scores
      
      # Canonical timestamps (optional, from PhonemeFrameTimestampDataIOPrep)
      - canonical_starts, canonical_ends: list[float] in seconds
    
    Output: All fields from the constituent classes combined
    """

    def __init__(self, hparams):
        super().__init__(hparams)
        self.frame_ms = hparams.get("frame_ms", 20.0)
        
    def _setup_label_encoder(self, datasets):
        """Setup label encoder with BOS/EOS tokens."""
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
            from_didatasets=[datasets[0]],
            output_key="phn_list_target",
            special_labels=special_labels,
            sequence_input=True,
        )

    def _create_pipeline(self):
        """Create comprehensive pipeline with all timestamp and score information."""
        @sb.utils.data_pipeline.takes(
            "wav", 
            "perceived_train_target", 
            "canonical_aligned", 
            "perceived_aligned",
            # MFA fields
            "mfa_phone_aligned",
            "mfa_phone_starts", 
            "mfa_phone_ends",
            "mfa_word_aligned",
            "mfa_word_starts",
            "mfa_word_ends", 
            "mfa_word_phone_start_idx",
            "mfa_word_phone_end_idx",
            # Word scores
            "word_scores",
            # Optional canonical timestamps
            "canonical_starts",
            "canonical_ends",
        )
        @sb.utils.data_pipeline.provides(
            "sig",
            # Target phoneme sequence (for training)
            "phn_list_target",
            "phn_encoded_list_target",
            "phn_encoded_target",
            "phn_list_target_bos",
            "phn_encoded_list_target_bos",
            "phn_encoded_target_bos",
            "phn_list_target_eos",
            "phn_encoded_list_target_eos",
            "phn_encoded_target_eos",
            # Canonical phoneme sequence
            "phn_list_canonical",
            "phn_encoded_list_canonical",
            "phn_encoded_canonical",
            "phn_list_canonical_bos",
            "phn_encoded_list_canonical_bos",
            "phn_encoded_canonical_bos",
            "phn_list_canonical_eos",
            "phn_encoded_list_canonical_eos",
            "phn_encoded_canonical_eos",
            # Perceived phoneme sequence
            "phn_list_perceived",
            "phn_encoded_list_perceived",
            "phn_encoded_perceived",
            "phn_list_perceived_bos",
            "phn_encoded_list_perceived_bos",
            "phn_encoded_perceived_bos",
            "phn_list_perceived_eos",
            "phn_encoded_list_perceived_eos",
            "phn_encoded_perceived_eos",
            # MFA phone-level timestamps (in frames)
            "mfa_phone_list",
            "mfa_phone_start_frames",
            "mfa_phone_end_frames",
            "mfa_phone_frame_ranges",
            # MFA word-level timestamps (in frames)
            "mfa_word_list",
            "mfa_word_start_frames",
            "mfa_word_end_frames",
            "mfa_word_phone_ranges",
            # Word-level pronunciation scores
            "word_boundaries",
            "word_accuracy_scores",
            "word_stress_scores",
            "word_total_scores",
            # Canonical phone timestamps (in frames)
            "canonical_phone_start_frames",
            "canonical_phone_end_frames",
            "canonical_phone_frame_ranges",
            # Mispronunciation labels
            "mispro_label",
        )
        def pipeline(
            wav, 
            perceived_train_target, 
            canonical_aligned, 
            perceived_aligned,
            mfa_phone_aligned,
            mfa_phone_starts,
            mfa_phone_ends,
            mfa_word_aligned,
            mfa_word_starts,
            mfa_word_ends,
            mfa_word_phone_start_idx,
            mfa_word_phone_end_idx,
            word_scores,
            canonical_starts,
            canonical_ends,
        ):
            # Load audio
            sig = sb.dataio.dataio.read_audio(wav)
            
            # Get audio properties for frame calculation
            sample_rate = 16000  # Assuming 16kHz
            hop_length_samples = int((self.frame_ms / 1000.0) * sample_rate)
            total_samples = sig.shape[0]
            total_frames = (total_samples + hop_length_samples - 1) // hop_length_samples
            
            def time_to_frame(time_sec):
                """Convert time in seconds to frame index."""
                frame_idx = int(time_sec * sample_rate / hop_length_samples)
                return max(0, min(frame_idx, total_frames - 1))
            
            # Process target phonemes
            phn_list_target = perceived_train_target.strip().split()
            phn_encoded_list_target = self.label_encoder.encode_sequence(phn_list_target)
            phn_encoded_target = torch.LongTensor(phn_encoded_list_target)
            
            phn_list_target_bos = ["<bos>"] + phn_list_target
            phn_encoded_list_target_bos = self.label_encoder.encode_sequence(phn_list_target_bos)
            phn_encoded_target_bos = torch.LongTensor(phn_encoded_list_target_bos)
            
            phn_list_target_eos = phn_list_target + ["<eos>"]
            phn_encoded_list_target_eos = self.label_encoder.encode_sequence(phn_list_target_eos)
            phn_encoded_target_eos = torch.LongTensor(phn_encoded_list_target_eos)
            
            # Process canonical phonemes
            phn_list_canonical = canonical_aligned.strip().split()
            phn_encoded_list_canonical = self.label_encoder.encode_sequence(phn_list_canonical)
            phn_encoded_canonical = torch.LongTensor(phn_encoded_list_canonical)
            
            phn_list_canonical_bos = ["<bos>"] + phn_list_canonical
            phn_encoded_list_canonical_bos = self.label_encoder.encode_sequence(phn_list_canonical_bos)
            phn_encoded_canonical_bos = torch.LongTensor(phn_encoded_list_canonical_bos)
            
            phn_list_canonical_eos = phn_list_canonical + ["<eos>"]
            phn_encoded_list_canonical_eos = self.label_encoder.encode_sequence(phn_list_canonical_eos)
            phn_encoded_canonical_eos = torch.LongTensor(phn_encoded_list_canonical_eos)
            
            # Process perceived phonemes
            phn_list_perceived = perceived_aligned.strip().split()
            phn_encoded_list_perceived = self.label_encoder.encode_sequence(phn_list_perceived)
            phn_encoded_perceived = torch.LongTensor(phn_encoded_list_perceived)
            
            phn_list_perceived_bos = ["<bos>"] + phn_list_perceived
            phn_encoded_list_perceived_bos = self.label_encoder.encode_sequence(phn_list_perceived_bos)
            phn_encoded_perceived_bos = torch.LongTensor(phn_encoded_list_perceived_bos)
            
            phn_list_perceived_eos = phn_list_perceived + ["<eos>"]
            phn_encoded_list_perceived_eos = self.label_encoder.encode_sequence(phn_list_perceived_eos)
            phn_encoded_perceived_eos = torch.LongTensor(phn_encoded_list_perceived_eos)
            
            # Process MFA phone-level timestamps
            mfa_phone_list = mfa_phone_aligned.strip().split()
            mfa_phone_start_frames = torch.LongTensor([time_to_frame(t) for t in mfa_phone_starts])
            mfa_phone_end_frames = torch.LongTensor([time_to_frame(t) for t in mfa_phone_ends])
            mfa_phone_frame_ranges = [(s.item(), e.item()) for s, e in zip(mfa_phone_start_frames, mfa_phone_end_frames)]
            
            # Process MFA word-level timestamps
            mfa_word_list = mfa_word_aligned.strip().split()
            mfa_word_start_frames = torch.LongTensor([time_to_frame(t) for t in mfa_word_starts])
            mfa_word_end_frames = torch.LongTensor([time_to_frame(t) for t in mfa_word_ends])
            mfa_word_phone_ranges = [
                (start_idx, end_idx) 
                for start_idx, end_idx in zip(mfa_word_phone_start_idx, mfa_word_phone_end_idx)
            ]
            
            # Process word-level pronunciation scores
            word_boundaries = []
            word_accuracy_scores = []
            word_stress_scores = []
            word_total_scores = []
            
            for ws in word_scores:
                phone_start = ws["phone_start"]
                phone_end = ws["phone_end"]
                word_boundaries.append((phone_start, phone_end))
                word_accuracy_scores.append(ws["accuracy"])
                word_stress_scores.append(ws["stress"])
                word_total_scores.append(ws["total"])
            
            # Process canonical phone timestamps (if available)
            if canonical_starts and canonical_ends:
                canonical_phone_start_frames = torch.LongTensor([time_to_frame(t) for t in canonical_starts])
                canonical_phone_end_frames = torch.LongTensor([time_to_frame(t) for t in canonical_ends])
                canonical_phone_frame_ranges = [
                    (s.item(), e.item()) 
                    for s, e in zip(canonical_phone_start_frames, canonical_phone_end_frames)
                ]
            else:
                # Fallback to empty if not provided
                canonical_phone_start_frames = torch.LongTensor([])
                canonical_phone_end_frames = torch.LongTensor([])
                canonical_phone_frame_ranges = []
            
            # Compute mispronunciation labels
            mispro_label = torch.LongTensor([
                0 if c == p else 1 
                for c, p in zip(phn_list_canonical, phn_list_perceived)
            ])
            
            yield sig
            yield phn_list_target
            yield phn_encoded_list_target
            yield phn_encoded_target
            yield phn_list_target_bos
            yield phn_encoded_list_target_bos
            yield phn_encoded_target_bos
            yield phn_list_target_eos
            yield phn_encoded_list_target_eos
            yield phn_encoded_target_eos
            yield phn_list_canonical
            yield phn_encoded_list_canonical
            yield phn_encoded_canonical
            yield phn_list_canonical_bos
            yield phn_encoded_list_canonical_bos
            yield phn_encoded_canonical_bos
            yield phn_list_canonical_eos
            yield phn_encoded_list_canonical_eos
            yield phn_encoded_canonical_eos
            yield phn_list_perceived
            yield phn_encoded_list_perceived
            yield phn_encoded_perceived
            yield phn_list_perceived_bos
            yield phn_encoded_list_perceived_bos
            yield phn_encoded_perceived_bos
            yield phn_list_perceived_eos
            yield phn_encoded_list_perceived_eos
            yield phn_encoded_perceived_eos
            yield mfa_phone_list
            yield mfa_phone_start_frames
            yield mfa_phone_end_frames
            yield mfa_phone_frame_ranges
            yield mfa_word_list
            yield mfa_word_start_frames
            yield mfa_word_end_frames
            yield mfa_word_phone_ranges
            yield word_boundaries
            yield word_accuracy_scores
            yield word_stress_scores
            yield word_total_scores
            yield canonical_phone_start_frames
            yield canonical_phone_end_frames
            yield canonical_phone_frame_ranges
            yield mispro_label

        return pipeline

    def prepare(self):
        """Prepare comprehensive datasets with all timestamp and score information."""
        train_data, valid_data, test_data = self._prepare_datasets()
        datasets = [train_data, valid_data, test_data]

        # Add pipeline
        pipeline = self._create_pipeline()
        sb.dataio.dataset.add_dynamic_item(datasets, pipeline)

        # Setup label encoder
        self._setup_label_encoder(datasets)

        # Set output keys - combination of all constituent classes
        output_keys = [
            "id", "sig",
            # Encoded sequences
            "phn_encoded_target", "phn_encoded_canonical", "phn_encoded_perceived",
            # Phoneme lists
            "phn_list_target", "phn_list_canonical", "phn_list_perceived",
            # BOS variants
            "phn_list_target_bos", "phn_list_canonical_bos", "phn_list_perceived_bos",
            # EOS variants
            "phn_list_target_eos", "phn_list_canonical_eos", "phn_list_perceived_eos",
            # Encoded BOS variants
            "phn_encoded_target_bos", "phn_encoded_canonical_bos", "phn_encoded_perceived_bos",
            # Encoded EOS variants
            "phn_encoded_target_eos", "phn_encoded_canonical_eos", "phn_encoded_perceived_eos",
            # MFA phone-level
            "mfa_phone_list", "mfa_phone_start_frames", "mfa_phone_end_frames", "mfa_phone_frame_ranges",
            # MFA word-level
            "mfa_word_list", "mfa_word_start_frames", "mfa_word_end_frames", "mfa_word_phone_ranges",
            # Word scores
            "word_boundaries", "word_accuracy_scores", "word_stress_scores", "word_total_scores",
            # Canonical timestamps
            "canonical_phone_start_frames", "canonical_phone_end_frames", "canonical_phone_frame_ranges",
            # Labels
            "wrd", "mispro_label"
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