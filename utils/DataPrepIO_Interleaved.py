"""
Data IO preparation for LLM with interleaved canonical-perceived-error sequences.
Extends LLMDataIOPrep_ver2 to generate interleaved target sequences.
"""

import torch
import speechbrain as sb
from .DataPrepIO import LLMDataIOPrep_ver2
import logging

logger = logging.getLogger(__name__)


class LLMDataIOPrep_Interleaved(LLMDataIOPrep_ver2):
    """
    Data IO preparation for LLM with interleaved canonical-perceived-error sequences.
    
    Output format:
        phn_list_target_interleaved: [can_1, perc_1, err_1, can_2, perc_2, err_2, ...]
    where err is one of: "=", "S", "D", "I"
        "=" = correct (0)
        "S" = substitution (1)
        "D" = deletion (2)
        "I" = insertion (3)
    """
    
    def _create_text_pipelines(self):
        """Create text processing pipelines with interleaved canonical-perceived-error sequences."""
        
        @sb.utils.data_pipeline.takes(
            "perceived_train_target", 
            "canonical_aligned", 
            "perceived_aligned"
        )
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
            
            "mispro_label",  # [L_phn] with values 0/1/2/3
            "phn_list_target_interleaved",  # [can_1, perc_1, err_1, ...]
            "phn_encoded_list_target_interleaved",
            "phn_encoded_target_interleaved",
        )
        def text_pipeline_interleaved(target, canonical, perceived):
            # ===== Basic sequences =====
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

            # ===== Compute mispro_label =====
            # 0=correct, 1=substitution, 2=deletion, 3=insertion
            mispro_label = []
            for c, p in zip(phn_list_canonical, phn_list_perceived):
                if c == p:
                    mispro_label.append(0)  # = correct
                elif p == "<sil>" and c != "<sil>":
                    mispro_label.append(2)  # D deletion
                elif p != "<sil>" and c == "<sil>":
                    mispro_label.append(3)  # I insertion
                else:
                    mispro_label.append(1)  # S substitution
            
            mispro_label = torch.LongTensor(mispro_label)
            yield mispro_label

            # ===== Generate interleaved sequence =====
            # Format: [can_1, perc_1, err_1, can_2, perc_2, err_2, ...]
            error_map = {0: "=", 1: "S", 2: "D", 3: "I"}
            interleaved_tokens = []
            
            for can, perc, err_id in zip(phn_list_canonical, phn_list_perceived, mispro_label):
                interleaved_tokens.append(can)
                interleaved_tokens.append(perc)
                interleaved_tokens.append(error_map[int(err_id.item())])
            
            phn_list_target_interleaved = interleaved_tokens
            yield phn_list_target_interleaved
            
            phn_encoded_list_target_interleaved = self.label_encoder.encode_sequence(phn_list_target_interleaved)
            yield phn_encoded_list_target_interleaved
            
            phn_encoded_target_interleaved = torch.LongTensor(phn_encoded_list_target_interleaved)
            yield phn_encoded_target_interleaved

        return text_pipeline_interleaved
    
    def prepare(self):
        """Prepare datasets for LLM with interleaved sequences."""
        train_data, valid_data, test_data = self._prepare_datasets()
        datasets = [train_data, valid_data, test_data]
        
        # Add audio pipeline
        audio_pipeline = self._create_audio_pipeline()
        sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)
        
        # Add text pipeline with interleaved sequences
        text_pipeline = self._create_text_pipelines()
        sb.dataio.dataset.add_dynamic_item([train_data], text_pipeline)
        sb.dataio.dataset.add_dynamic_item([valid_data, test_data], text_pipeline)

        # Setup label encoder
        self._setup_label_encoder(datasets)

        # Set output keys - including new interleaved sequence fields
        output_keys = [
            "id", "sig",
            # Base sequences
            "phn_encoded_target",
            "phn_list_target",
            "phn_encoded_canonical",
            "phn_list_canonical",
            "phn_encoded_perceived",
            "phn_list_perceived",
            # BOS/EOS variants (optional, can be removed if not needed)
            "phn_list_target_bos", "phn_list_target_eos",
            "phn_encoded_target_bos", "phn_encoded_target_eos",
            "phn_list_canonical_bos", "phn_list_canonical_eos",
            "phn_encoded_canonical_bos", "phn_encoded_canonical_eos",
            "phn_list_perceived_bos", "phn_list_perceived_eos",
            "phn_encoded_perceived_bos", "phn_encoded_perceived_eos",
            # NEW: Error labels and interleaved sequence
            "mispro_label",
            "phn_list_target_interleaved",
            "phn_encoded_target_interleaved",
            "wrd",
        ]
        
        sb.dataio.dataset.set_output_keys([train_data], output_keys)
        sb.dataio.dataset.set_output_keys([valid_data, test_data], output_keys)

        logger.info(f"[Interleaved Data] Train: {len(train_data)}, Valid: {len(valid_data)}, Test: {len(test_data)}")
        
        return train_data, valid_data, test_data, self.label_encoder
