"""
Data IO preparation with blank insertion between all phonemes.

This module provides BlankInsertionBetweenAllDataIOPrep class which extends LLMDataIOPrep
to insert artificial <blank> tokens between ALL phonemes (not just consecutive duplicates).
"""

import torch
import speechbrain as sb
from .DataPrepIO import LLMDataIOPrep


class BlankInsertionBetweenAllDataIOPrep(LLMDataIOPrep):
    """
    Data IO preparation that inserts artificial <blank> tags between ALL phonemes.
    
    This class extends LLMDataIOPrep to insert a blank token between every pair of phonemes,
    creating a more explicit separation for the model to learn.
    
    Example:
        Input:  "a a b b a c"
        Output: "a <blank> a <blank> b <blank> b <blank> a <blank> c"
    
    Important:
        - Canonical and perceived phoneme sequences have blanks inserted
        - Training target sequences do NOT have blanks inserted (kept as original)
    
    This is useful for:
    - Providing explicit phoneme boundaries to the model
    - Improving CTC alignment by inserting optional blanks between all phonemes
    - Giving the model more flexibility in alignment
    """
    
    def _insert_blanks_between_all(self, phoneme_list, blank_label="<blank>"):
        """
        Insert blank tags between ALL phonemes (not just consecutive duplicates).
        
        Args:
            phoneme_list (list): List of phoneme strings
            blank_label (str): The blank token to insert (default: "<blank>")
            
        Returns:
            list: New phoneme list with blanks inserted between all phonemes
            
        Example:
            >>> prep = BlankInsertionBetweenAllDataIOPrep({})
            >>> prep._insert_blanks_between_all(['a', 'b', 'c'])
            ['a', '<blank>', 'b', '<blank>', 'c']
            >>> prep._insert_blanks_between_all(['a', 'a', 'b'])
            ['a', '<blank>', 'a', '<blank>', 'b']
        """
        if len(phoneme_list) <= 1:
            return phoneme_list
        
        result = []
        for i, phoneme in enumerate(phoneme_list):
            result.append(phoneme)
            # Add blank after every phoneme except the last one
            if i < len(phoneme_list) - 1:
                result.append(blank_label)
        
        return result
    
    def _create_text_pipelines(self):
        """Create text processing pipelines with blank insertion between all phonemes."""
        blank_label = "<blank>"
        
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
            # Process target WITHOUT blank insertion (保持原样)
            phn_list_target = target.strip().split()
            # 注意：target 不插入 blank
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
            
            # Process canonical WITH blank insertion between all phonemes
            phn_list_canonical_orig = canonical.strip().split()
            phn_list_canonical = self._insert_blanks_between_all(phn_list_canonical_orig, blank_label)
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
            
            # Process perceived WITH blank insertion between all phonemes
            phn_list_perceived_orig = perceived.strip().split()
            phn_list_perceived = self._insert_blanks_between_all(phn_list_perceived_orig, blank_label)
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

            # Mispronunciation label (based on original sequences without blanks)
            mispro_label = [1 if p != c else 0 for p, c in zip(phn_list_perceived_orig, phn_list_canonical_orig)]
            mispro_label = torch.LongTensor(mispro_label)
        
            yield mispro_label

        return text_pipeline_test
    
    def prepare(self):
        """Prepare datasets for LLM with mispronunciation detection and blank insertion between all phonemes."""
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
