import os
import sys
import torch
import torch.nn as nn
import logging
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from mpd_eval_v4 import MpdStats
import librosa
import json
import wandb
import time
import torchaudio
import csv
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
from torch.nn.functional import kl_div
from speechbrain.nnet.loss.guidedattn_loss import GuidedAttentionLoss
import re
from utils.EncoderManager import EncoderManager
from utils.LossManager import CTCLossManager
from utils.context_phone_metrics import make_context_phone_ind2lab, decode_ids_to_phone_string


# ============================================================================
# Result Data Classes for Flexible Output
# ============================================================================

@dataclass
class InferenceResult:
    """Single inference result with optional metrics"""
    id: str
    prediction: str  # Decoded phoneme sequence
    canonical: Optional[str] = None
    perceived: Optional[str] = None
    target: Optional[str] = None
    per: Optional[float] = None  # Phoneme Error Rate
    mpd_result: Optional[Dict[str, Any]] = None  # MPD metrics


@dataclass
class TestResults:
    """Collection of test results with summary statistics"""
    results: List[InferenceResult] = field(default_factory=list)
    has_reference: bool = False
    has_canonical: bool = False
    has_perceived: bool = False
    
    # Summary metrics (only computed if reference available)
    overall_per: Optional[float] = None
    overall_mpd_f1: Optional[float] = None
    overall_mpd_precision: Optional[float] = None
    overall_mpd_recall: Optional[float] = None
    
    def add_result(self, result: InferenceResult):
        self.results.append(result)
        if result.target is not None:
            self.has_reference = True
        if result.canonical is not None:
            self.has_canonical = True
        if result.perceived is not None:
            self.has_perceived = True


# ============================================================================
# CSV Result Writer
# ============================================================================

class ResultWriter:
    """Flexible CSV writer that adapts columns based on available data"""
    
    def __init__(self, output_path: str):
        self.output_path = output_path
        self.results: List[InferenceResult] = []
    
    def add_result(self, result: InferenceResult):
        self.results.append(result)
    
    def add_results(self, results: List[InferenceResult]):
        self.results.extend(results)
    
    def write(self, test_results: Optional[TestResults] = None):
        """
        Write results to CSV with flexible columns based on available data.
        
        Columns:
        - Always: id, prediction
        - If has_canonical: canonical
        - If has_perceived: perceived  
        - If has_reference: target, per
        - If MPD computed: mpd_correct, mpd_precision, mpd_recall, mpd_f1
        """
        if test_results is not None:
            results = test_results.results
            has_reference = test_results.has_reference
            has_canonical = test_results.has_canonical
            has_perceived = test_results.has_perceived
        else:
            results = self.results
            has_reference = any(r.target is not None for r in results)
            has_canonical = any(r.canonical is not None for r in results)
            has_perceived = any(r.perceived is not None for r in results)
        
        if not results:
            logging.warning("No results to write")
            return
        
        # Determine columns based on available data
        columns = ['id', 'prediction']
        
        if has_canonical:
            columns.append('canonical')
        if has_perceived:
            columns.append('perceived')
        if has_reference:
            columns.extend(['target', 'per'])
        
        # Check if MPD results are available
        has_mpd = any(r.mpd_result is not None for r in results)
        if has_mpd:
            columns.extend(['mpd_correct', 'mpd_precision', 'mpd_recall', 'mpd_f1'])
        
        # Write CSV
        os.makedirs(os.path.dirname(self.output_path) if os.path.dirname(self.output_path) else '.', exist_ok=True)
        
        with open(self.output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            
            for result in results:
                row = {
                    'id': result.id,
                    'prediction': result.prediction
                }
                
                if has_canonical:
                    row['canonical'] = result.canonical or ''
                if has_perceived:
                    row['perceived'] = result.perceived or ''
                if has_reference:
                    row['target'] = result.target or ''
                    row['per'] = f"{result.per:.4f}" if result.per is not None else ''
                
                if has_mpd and result.mpd_result:
                    row['mpd_correct'] = result.mpd_result.get('correct', '')
                    row['mpd_precision'] = f"{result.mpd_result.get('precision', 0):.4f}" if result.mpd_result.get('precision') is not None else ''
                    row['mpd_recall'] = f"{result.mpd_result.get('recall', 0):.4f}" if result.mpd_result.get('recall') is not None else ''
                    row['mpd_f1'] = f"{result.mpd_result.get('f1', 0):.4f}" if result.mpd_result.get('f1') is not None else ''
                
                writer.writerow(row)
        
        logging.info(f"✅ Results written to: {self.output_path}")
        
        # Write summary if available
        if test_results and has_reference:
            summary_path = self.output_path.replace('.csv', '_summary.txt')
            with open(summary_path, 'w') as f:
                f.write("=" * 50 + "\n")
                f.write("Test Results Summary\n")
                f.write("=" * 50 + "\n")
                f.write(f"Total samples: {len(results)}\n")
                if test_results.overall_per is not None:
                    f.write(f"Overall PER: {test_results.overall_per:.4f}\n")
                if test_results.overall_mpd_f1 is not None:
                    f.write(f"Overall MPD F1: {test_results.overall_mpd_f1:.4f}\n")
                    f.write(f"Overall MPD Precision: {test_results.overall_mpd_precision:.4f}\n")
                    f.write(f"Overall MPD Recall: {test_results.overall_mpd_recall:.4f}\n")
            logging.info(f"✅ Summary written to: {summary_path}")

# ============================================================================
# Base Model with Unified Architecture
# ============================================================================

class PhnMonoSSLModel_IF(sb.Brain):
    """
    Unified base model for phoneme-level mispronunciation detection.
    
    Architecture: SSL -> Encoder -> CTC
    
    Supports:
    - Encoder types: None, Linear, Conformer, Zipformer, RVQ
    - CTC loss types: Vanilla, Label Prior, OTTC, CR-CTC
    - Configurable via hparams without code changes
    """
    
    def __init__(self, *args, patience=20, **kwargs):
        super().__init__(*args, **kwargs)
        self.patience = patience
        self.no_improve_epochs = 0
        self.best_per_list = []
        self.best_mpd_f1_list = []
        self.best_per = float('inf')
        self.best_mpd_f1 = float('-inf')
        self.last_improved_epoch = 0
        self.best_valid_loss = float('inf')
        self.best_valid_loss_list = []
        
        # CR-CTC loss tracking (will be reset each epoch)
        self.cr_loss_sum = 0.0
        self.cr_loss_count = 0
        self.ctc_loss_sum = 0.0
        self.ctc_loss_count = 0
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if getattr(self.modules, "perceived_ssl", None) is not None:
            self.modules.perceived_ssl.to(self.device)
        if getattr(self.modules, "canonical_ssl", None) is not None:
            self.modules.canonical_ssl.to(self.device)
        
        # Initialize unified components
        self._init_encoder_manager()
        self._init_loss_manager()
        self._init_mpd_loss_manager()
        
        # MPD loss tracking (will be reset each epoch)
        self.mpd_binary_loss_sum = 0.0
        self.mpd_binary_loss_count = 0
        self.mpd_cls_loss_sum = 0.0
        self.mpd_cls_loss_count = 0
        
    def _init_encoder_manager(self):
        """Initialize encoder manager based on hparams"""
        encoder_type = getattr(self.hparams, 'encoder_type', None)
        
        # Auto-detect encoder type if not specified
        if encoder_type is None:
            if getattr(self.modules, 'RVQ', None) is not None:
                encoder_type = 'rvq'
            elif getattr(self.modules, 'ConformerEncoder', None) is not None:
                encoder_type = 'conformer'
            elif getattr(self.modules, 'ZipformerEncoder', None) is not None:
                encoder_type = 'zipformer'
            elif getattr(self.modules, 'enc', None) is not None:
                encoder_type = 'linear'
            else:
                encoder_type = None
        
        self.encoder_manager = EncoderManager(
            encoder_type=encoder_type,
            modules=self.modules,
            hparams=self.hparams,
            device=self.device
        )
        
    def _init_loss_manager(self):
        """Initialize loss manager based on hparams"""
        loss_type = getattr(self.hparams, 'ctc_loss_type', 'vanilla')
        
        # import pdb; pdb.set_trace()
        
        self.loss_manager = CTCLossManager(
            loss_type=loss_type,
            ctc_cost=self.hparams.ctc_cost,
            hparams=self.hparams,
            blank_index=self.hparams.blank_index
        )
    
    def _init_mpd_loss_manager(self):
        """Initialize MPD loss manager"""
        from utils.MPDLossManager import MPDLossManager
        
        self.mpd_loss_manager = MPDLossManager(
            hparams=self.hparams,
            device=self.device
        )
    
    def create_attention_mask_from_input_sequence(self, input_sequence):
        """Create attention mask from input sequence lengths (optimized for GPU)"""
        batch_size = input_sequence.size(0)
        max_len = input_sequence.max().item()  # Single sync point
        # Create arange directly on GPU
        indices = torch.arange(max_len, device=self.device, dtype=input_sequence.dtype).unsqueeze(0)
        attention_mask = (indices < input_sequence.unsqueeze(1)).float()
        return attention_mask
    
    def check_gradients(self, loss):
        """Check if gradients are finite"""
        if not torch.isfinite(loss):
            print("Warning: loss is not finite, skipping step")
            return False
        return True
    
    def compute_forward(self, batch, stage, pseudo_labels=None, pseudo_label_lens=None):
        """
        Unified forward pass supporting all configurations.
        
        Args:
            batch: Input batch
            stage: Training stage
            pseudo_labels: Optional pseudo labels for unlabeled data (for OTTC)
            pseudo_label_lens: Optional pseudo label lengths (for OTTC)
        
        Returns:
            Depending on configuration, returns:
            - (p_ctc, wav_lens) for vanilla
            - (p_ctc, wav_lens, commitment_loss, codebook_loss) for RVQ
            - (p_ctc, logits, weights_logits, weights_labels, wav_lens) for OTTC
            - (p_ctc, wav_lens, time_mask, is_crctc_mode) for CR-CTC
        """
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        
        # check if OTTC mode is enabled
        use_ottc = (
            self.loss_manager.loss_type == 'ottc' and 
            stage == sb.Stage.TRAIN and
            getattr(self.hparams, "use_ottc", True)
        )
        
        # Check if CR-CTC mode is enabled (requires augmentation)
        use_crctc = (
            self.loss_manager.loss_type == 'crctc' and 
            stage == sb.Stage.TRAIN and
            getattr(self.hparams, "use_crctc", True) and
            hasattr(self.hparams, "augmentation")  # CR-CTC requires augmentation
        )

        # Check if CT-OTTC mode is enabled
        use_crottc = (
            self.loss_manager.loss_type == 'crottc' and
            stage == sb.Stage.TRAIN and
            getattr(self.hparams, "use_crottc", True) and
            hasattr(self.hparams, "augmentation")  # OTTC requires augmentation
        )
        
        # Apply augmentation in training,
        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "speed_augmentation"):
                wavs = self.hparams.speed_augmentation(wavs)
            
            # Essential Augmentation for for CR-CTC / CR-OTTC
            if use_crctc or use_crottc:
                # CR-CTC: Apply augmentation twice to get two different views
                wavs_1, wav_lens_1 = self.hparams.augmentation.forward(wavs, lengths=wav_lens)
                wavs_2, wav_lens_2 = self.hparams.augmentation.forward(wavs, lengths=wav_lens)
                
                # Extract SSL features for both augmented versions
                feats_1 = self.modules.perceived_ssl(wavs_1)
                feats_2 = self.modules.perceived_ssl(wavs_2)
                
                # Concatenate both versions: [2*B, T, D]
                feats = torch.cat([feats_1, feats_2], dim=0)
                wav_lens = wav_lens.repeat(2)
            else:
                # Standard augmentation (non-CR-CTC)
                if hasattr(self.hparams, "augmentation"):
                    wavs = self.hparams.augmentation(wavs)
                
                # Extract SSL features
                feats = self.modules.perceived_ssl(wavs)
        else:
            # Extract SSL features (no augmentation)
            feats = self.modules.perceived_ssl(wavs)
        
        # Encode features
        x, encoder_extras = self.encoder_manager(feats, wav_lens)
        
        # apply fusenet
        cano_phn, cano_lens = batch.phn_encoded_canonical if hasattr(batch, 'phn_encoded_canonical') else (None, None)
        Cano_emb = self.modules.phn_emb(cano_phn) if cano_phn is not None else None

        # fuse cano_emb with x if crottc, use half batch of x for crottc/crctc
        if use_crottc or use_crctc:
            x_half = x[: x.shape[0] // 2, :, :]
        else:
            x_half = x
        
        memory = x_half
        if self.hparams.post_encoder_reduction_factor >= 1:
            memory = self.modules.projector(memory)

        from utils.layers.utils import make_pad_mask
        from speechbrain.nnet.attention import RelPosEncXL, RelPosMHAXL, RoPEMHA 
        # import pdb; pdb.set_trace()

        fuse_feat, _,  fuse_attn = self.modules.fuse_net(
            tgt=Cano_emb,
            memory=memory,
            tgt_key_padding_mask=make_pad_mask(Cano_emb.shape[1] * cano_lens, maxlen=Cano_emb.shape[1]).to(self.device),
            pos_embs_tgt=RelPosEncXL(emb_dim=self.hparams.dnn_neurons)(Cano_emb).to(self.device),
            pos_embs_src=RelPosEncXL(emb_dim=self.hparams.dnn_neurons)(memory).to(self.device)
        )
            # memory_key_padding_mask=make_pad_mask(memory.shape[1] * wav_lens, maxlen=memory.shape[1]).to(self.device),
        
        # import pdb; pdb.set_trace()

        h_mispro = self.hparams.mispro_head(fuse_feat.transpose(1, 2))
        # for binary detection, 
        h_mispro_bin = self.hparams.mispro_head_binary_out(h_mispro)
        h_mispro_bin = h_mispro_bin.transpose(1, 2)  # [B, T_c, 1]
        # for multi-class detection, 4 classes
        h_mispro_cls = self.hparams.mispro_head_class_out(h_mispro.transpose(1, 2)) #[B, T_c, 4]

        # [B, T_p, D]
            # pos_embs_tgt=RelPosEncXL(emb_dim=self.hparams.dnn_neurons)(Cano_emb).to(self.device),
            # pos_embs_src=RelPosEncXL(emb_dim=self.hparams.dnn_neurons)(memory).to(self.device)

        # import pdb; pdb.set_trace()

        # CTC output layer
        logits = self.modules.ctc_lin(x)
        p_ctc = self.hparams.log_softmax(logits)
        
        # Handle OTTC-specific outputs
        if use_crottc or use_ottc:
            # 支持 labeled 数据（真实标签）和 unlabeled 数据（伪标签）
            if hasattr(self.modules, "lm_weight") and stage != sb.Stage.TEST:
                # 获取标签（真实或伪标签）
                if hasattr(batch, 'phn_encoded_target') and batch.phn_encoded_target is not None:
                    # Labeled data: 使用真实标签
                    targets, target_lens = batch.phn_encoded_target
                elif pseudo_labels is not None:
                    # Unlabeled data: 使用伪标签（从 teacher model 生成）
                    targets = pseudo_labels
                    target_lens = pseudo_label_lens
                else:
                    # 没有任何标签，跳过 OTTC 权重计算
                    targets = None
                
                if targets is not None:
                    labels_mask = (targets != self.hparams.blank_index).float()
                    
                    weights_logits = self.modules.lm_weight(x)
                    lens_abs = (wav_lens * feats.shape[-2]).int()
                    output_mask = self.create_attention_mask_from_input_sequence(lens_abs)
                    
                    import torch.nn.functional as F
                    # Optimize: use in-place operations and avoid unnecessary copies
                    weights_logits = weights_logits.squeeze()
                    weights_logits = weights_logits.masked_fill(output_mask == 0, -torch.inf)
                    weights_logits = F.softmax(weights_logits, dim=-1)
                    
                    # Optimize: use clamp to avoid division by zero without sync
                    label_sums = labels_mask.sum(dim=1, keepdim=True).clamp(min=1e-8)
                    weights_labels = labels_mask / label_sums
                    
                    return p_ctc, logits, weights_logits, weights_labels, wav_lens, h_mispro_bin, h_mispro_cls
        
        # Handle RVQ outputs
        if 'commitment_loss' in encoder_extras:
            return (p_ctc, wav_lens, 
                    encoder_extras['commitment_loss'], 
                    encoder_extras['codebook_loss'])
        
        # Handle CR-CTC outputs
        if use_crctc:
            return p_ctc, wav_lens, None, True  # True indicates CR-CTC mode
        
        # Standard output
        return p_ctc, wav_lens
    
    def compute_objectives(self, predictions, batch, stage):
        """Unified objective computation with flexible reference handling"""
        # Parse predictions based on type
        if len(predictions) == 2:
            p_ctc, wav_lens = predictions
            extras = {}
        elif len(predictions) == 4:
            if isinstance(predictions[3], bool):  # CR-CTC
                p_ctc, wav_lens, time_mask, is_crctc_mode = predictions
                extras = {'time_mask': time_mask, 'is_crctc_mode': is_crctc_mode}
            else:  # RVQ
                p_ctc, wav_lens, commitment_loss, codebook_loss = predictions
                extras = {'commitment_loss': commitment_loss, 'codebook_loss': codebook_loss}
        elif len(predictions) == 7:  # OTTC or CR-OTTC
            p_ctc, logits, weights_logits, weights_labels, wav_lens, h_mispro_bin, h_mispro_cls = predictions
            extras = {'logits': logits, 'weights_logits': weights_logits, 'weights_labels': weights_labels, 'h_mispro_bin': h_mispro_bin, 'h_mispro_cls': h_mispro_cls}
            if self.loss_manager.loss_type == 'crottc':
                extras['is_crottc_mode'] = True
        else:
            raise ValueError(f"Unexpected predictions format: {len(predictions)} elements")
    
        # Get batch IDs
        ids = batch.id
        
        # Check what reference data is available
        has_target = hasattr(batch, 'phn_encoded_target')
        has_canonical = hasattr(batch, 'phn_encoded_canonical')
        has_perceived = hasattr(batch, 'phn_encoded_perceived')
        
        # Get targets (required for loss computation in training)
        targets = None
        target_lens = None
        canonicals = None
        canonical_lens = None
        perceiveds = None
        perceived_lens = None
        
        if has_target:
            targets, target_lens = batch.phn_encoded_target
        if has_canonical:
            canonicals, canonical_lens = batch.phn_encoded_canonical
        if has_perceived:
            perceiveds, perceived_lens = batch.phn_encoded_perceived
        
        # Handle target selection (canonical/perceived/target)
        if stage != sb.Stage.TRAIN and has_target:
            training_target = getattr(self.hparams, 'training_target', 'target')
            if training_target == "canonical" and has_canonical:
                targets = canonicals
                target_lens = canonical_lens
            elif training_target == "perceived" and has_perceived:
                targets = perceiveds
                target_lens = perceived_lens
        
        # Compute loss (only if targets available)
        if targets is not None:
            loss, loss_dict = self.loss_manager.compute_loss(
                p_ctc, targets, wav_lens, target_lens, stage, extras
            )
        else:
            # No targets available - return dummy loss for inference-only mode
            loss = torch.tensor(0.0, device=self.device)
            loss_dict = {}
        
        # Track CR-CTC / CR-OTTC losses for epoch-level logging (keep on GPU)
        if stage == sb.Stage.TRAIN:
            if 'cr_loss' in loss_dict:
                self.cr_loss_sum += loss_dict['cr_loss'].item()
                self.cr_loss_count += 1
            if 'ctc_loss' in loss_dict:
                self.ctc_loss_sum += loss_dict['ctc_loss'].item()
                self.ctc_loss_count += 1
            if 'ottc_loss' in loss_dict:
                self.ctc_loss_sum += loss_dict['ottc_loss'].item()
                self.ctc_loss_count += 1
        
        # Add RVQ losses if present
        if 'commitment_loss' in extras:
            loss = loss + extras['commitment_loss'] + extras['codebook_loss']
            loss_dict['commitment_loss'] = extras['commitment_loss'].detach()
            loss_dict['codebook_loss'] = extras['codebook_loss'].detach()
        
        # ===== Compute MPD losses (mispronunciation detection) =====
        if stage == sb.Stage.TRAIN and 'h_mispro_bin' in extras and 'h_mispro_cls' in extras:
            h_mispro_bin = extras['h_mispro_bin']
            h_mispro_cls = extras['h_mispro_cls']
            
            # Get canonical lengths for masking if available
            cano_lens = None
            if canonicals is not None and canonical_lens is not None:
                cano_lens = canonical_lens
            
            # Compute MPD losses
            
            mpd_loss, mpd_loss_dict = self.mpd_loss_manager.compute_loss_with_weight_mask(
                h_mispro_bin, h_mispro_cls, batch, cano_lens=cano_lens, stage=stage
            )
            
            # Add MPD losses to total loss and loss dict
            loss_dict.update(mpd_loss_dict)
            
            # Get MPD loss weights from hparams
            mpd_loss_weight = getattr(self.hparams, 'mpd_loss_weight', 0.1)
            loss = loss + mpd_loss_weight * mpd_loss
            
            # Track MPD losses for logging
            # import pdb; pdb.set_trace()
            if mpd_loss_dict['binary_loss'].item() > 0:
                self.mpd_binary_loss_sum += mpd_loss_dict['binary_loss'].item()
                self.mpd_binary_loss_count += 1
            if mpd_loss_dict['cls_loss'].item() > 0:
                self.mpd_cls_loss_sum += mpd_loss_dict['cls_loss'].item()
                self.mpd_cls_loss_count += 1
            
        
        # Evaluation metrics (validation/test stage)
        if stage != sb.Stage.TRAIN:
            # For CR-CTC, use first half for evaluation
            if extras.get('is_crctc_mode', False):
                p_ctc_eval = p_ctc[:len(ids)]
                wav_lens_eval = wav_lens[:len(ids)]
            else:
                p_ctc_eval = p_ctc
                wav_lens_eval = wav_lens
            
            # Decode predictions
            sequence = sb.decoders.ctc_greedy_decode(
                p_ctc_eval, wav_lens_eval, blank_id=self.hparams.blank_index
            )
            
            # CTC metrics (only if targets available)
            if has_target and targets is not None:
                from utils.losses.CTCLossWithLabelPriors import CTCLossWithLabelPriors
                if isinstance(self.hparams.ctc_cost, CTCLossWithLabelPriors):
                    try:
                        self.ctc_metrics.append(
                            ids,
                            log_probs=p_ctc_eval.permute(1, 0, 2),
                            targets=targets,
                            input_lengths=(wav_lens_eval * p_ctc_eval.shape[1]).to(torch.int32),
                            target_lengths=(target_lens * targets.shape[1]).to(torch.int32)
                        )
                    except:
                        self.ctc_metrics.append(ids, p_ctc_eval, targets, wav_lens_eval, target_lens)
                else:
                    self.ctc_metrics.append(ids, p_ctc_eval, targets, wav_lens_eval, target_lens)
                
                # PER metrics
                # Remove token ID 70 from sequences
                sequence = [[token for token in seq if token != 70] for seq in sequence]
                        
                self.per_metrics.append(
                    ids=ids,
                    predict=sequence,
                    target=targets,
                    predict_len=None,
                    target_len=target_lens,
                    ind2lab=self._metric_ind2lab(),
                )
            
            # MPD metrics (only if canonical and perceived available)
            if has_canonical and has_perceived and canonicals is not None and perceiveds is not None:
                self.mpd_metrics.append(
                    ids=ids,
                    predict=sequence,
                    canonical=canonicals,
                    perceived=perceiveds,
                    predict_len=None,
                    canonical_len=canonical_lens,
                    perceived_len=perceived_lens,
                    ind2lab=self._metric_ind2lab(),
                )
            
            # Collect results for CSV output in TEST stage
            if stage == sb.Stage.TEST:
                if not hasattr(self, 'test_results_for_csv'):
                    self.test_results_for_csv = TestResults()
                
                for i, (seq_id, seq) in enumerate(zip(ids, sequence)):
                    pred_str = self._decode_sequence(seq)
                    
                    result = InferenceResult(
                        id=seq_id,
                        prediction=pred_str
                    )
                    
                    if has_canonical and canonicals is not None:
                        result.canonical = self._decode_tensor(canonicals[i], canonical_lens[i] if canonical_lens is not None else None)
                    
                    if has_perceived and perceiveds is not None:
                        result.perceived = self._decode_tensor(perceiveds[i], perceived_lens[i] if perceived_lens is not None else None)
                    
                    if has_target and targets is not None:
                        result.target = self._decode_tensor(targets[i], target_lens[i] if target_lens is not None else None)
                        result.per = self._compute_per(pred_str, result.target)
                    
                    self.test_results_for_csv.add_result(result)
        
        return loss
    
    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        predictions = self.compute_forward(batch, stage=stage)
        loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()
    
    # ========================================================================
    # Inference Methods
    # ========================================================================
    
    def inference(
        self,
        audio_path: str,
        canonical: Optional[str] = None,
        return_details: bool = False
    ) -> Union[str, InferenceResult]:
        """
        Run inference on a single audio file.
        
        Args:
            audio_path: Path to the audio file
            canonical: Optional canonical phoneme sequence (for MPD evaluation)
            return_details: If True, return InferenceResult with full details
        
        Returns:
            Predicted phoneme sequence string, or InferenceResult if return_details=True
        """
        self.modules.eval()
        
        with torch.no_grad():
            # Load and preprocess audio
            wavs, sr = torchaudio.load(audio_path)
            
            # Resample if needed
            target_sr = getattr(self.hparams, 'sample_rate', 16000)
            if sr != target_sr:
                resampler = torchaudio.transforms.Resample(sr, target_sr)
                wavs = resampler(wavs)
            
            # Ensure mono
            if wavs.shape[0] > 1:
                wavs = wavs.mean(dim=0, keepdim=True)
            
            # Add batch dimension and move to device
            wavs = wavs.to(self.device)
            if wavs.dim() == 2:
                wavs = wavs.unsqueeze(0)  # [1, C, T] -> [B, C, T]
            if wavs.dim() == 3 and wavs.shape[1] == 1:
                wavs = wavs.squeeze(1)  # [B, 1, T] -> [B, T]
            
            wav_lens = torch.tensor([1.0], device=self.device)
            
            # Forward pass
            feats = self.modules.perceived_ssl(wavs)
            x, _ = self.encoder_manager(feats, wav_lens)
            logits = self.modules.ctc_lin(x)
            p_ctc = self.hparams.log_softmax(logits)
            
            # Decode
            sequence = sb.decoders.ctc_greedy_decode(
                p_ctc, wav_lens, blank_id=self.hparams.blank_index
            )
            
            # Convert to string
            pred_str = self._decode_sequence(sequence[0])
        
        if not return_details:
            return pred_str
        
        # Build detailed result
        result = InferenceResult(
            id=os.path.basename(audio_path),
            prediction=pred_str,
            canonical=canonical
        )
        
        # print out
        return result
    
    def inference_batch(
        self,
        batch,
        compute_metrics: bool = True
    ) -> List[InferenceResult]:
        """
        Run inference on a batch and return detailed results.
        
        Args:
            batch: SpeechBrain batch object
            compute_metrics: Whether to compute metrics if reference available
        
        Returns:
            List of InferenceResult objects
        """
        self.modules.eval()
        
        with torch.no_grad():
            batch = batch.to(self.device)
            wavs, wav_lens = batch.sig
            ids = batch.id
            
            # Forward pass
            feats = self.modules.perceived_ssl(wavs)
            x, _ = self.encoder_manager(feats, wav_lens)
            logits = self.modules.ctc_lin(x)
            p_ctc = self.hparams.log_softmax(logits)
            
            # Decode
            sequences = sb.decoders.ctc_greedy_decode(
                p_ctc, wav_lens, blank_id=self.hparams.blank_index
            )
        
        results = []
        
        # Check what reference data is available
        has_target = hasattr(batch, 'phn_encoded_target')
        has_canonical = hasattr(batch, 'phn_encoded_canonical')
        has_perceived = hasattr(batch, 'phn_encoded_perceived')
        
        # Get reference data if available
        targets = None
        target_lens = None
        canonicals = None
        canonical_lens = None
        perceiveds = None
        perceived_lens = None
        
        if has_target:
            targets, target_lens = batch.phn_encoded_target
        if has_canonical:
            canonicals, canonical_lens = batch.phn_encoded_canonical
        if has_perceived:
            perceiveds, perceived_lens = batch.phn_encoded_perceived
        
        for i, (seq_id, seq) in enumerate(zip(ids, sequences)):
            pred_str = self._decode_sequence(seq)
            
            result = InferenceResult(
                id=seq_id,
                prediction=pred_str
            )
            
            # Add reference data if available
            if has_canonical and canonicals is not None:
                result.canonical = self._decode_tensor(canonicals[i], canonical_lens[i] if canonical_lens is not None else None)
            
            if has_perceived and perceiveds is not None:
                result.perceived = self._decode_tensor(perceiveds[i], perceived_lens[i] if perceived_lens is not None else None)
            
            if has_target and targets is not None:
                result.target = self._decode_tensor(targets[i], target_lens[i] if target_lens is not None else None)
                
                # Compute PER if target available and metrics requested
                if compute_metrics:
                    result.per = self._compute_per(pred_str, result.target)
            
            # Compute MPD if canonical and perceived available
            if compute_metrics and has_canonical and has_perceived:
                result.mpd_result = self._compute_mpd_single(
                    pred_str,
                    result.canonical,
                    result.perceived
                )
            
            results.append(result)
        
        return results
    
    def inference_dataset(
        self,
        dataloader,
        output_csv: Optional[str] = None,
        compute_metrics: bool = True,
        show_progress: bool = True
    ) -> TestResults:
        """
        Run inference on entire dataset and optionally save to CSV.
        
        Args:
            dataloader: PyTorch DataLoader
            output_csv: Optional path to save CSV results
            compute_metrics: Whether to compute metrics if reference available
            show_progress: Whether to show progress bar
        
        Returns:
            TestResults object with all results and summary statistics
        """
        self.modules.eval()
        test_results = TestResults()
        
        from tqdm import tqdm
        iterator = tqdm(dataloader, desc="Inference") if show_progress else dataloader
        
        for batch in iterator:
            batch_results = self.inference_batch(batch, compute_metrics=compute_metrics)
            for result in batch_results:
                test_results.add_result(result)
        
        # Compute summary metrics if reference data available
        if test_results.has_reference and compute_metrics:
            pers = [r.per for r in test_results.results if r.per is not None]
            if pers:
                test_results.overall_per = sum(pers) / len(pers)
        
        if test_results.has_canonical and test_results.has_perceived and compute_metrics:
            mpd_results = [r.mpd_result for r in test_results.results if r.mpd_result is not None]
            if mpd_results:
                # Aggregate MPD metrics
                test_results.overall_mpd_precision = sum(m.get('precision', 0) for m in mpd_results) / len(mpd_results)
                test_results.overall_mpd_recall = sum(m.get('recall', 0) for m in mpd_results) / len(mpd_results)
                test_results.overall_mpd_f1 = sum(m.get('f1', 0) for m in mpd_results) / len(mpd_results)
        
        # Save to CSV if path provided
        if output_csv:
            writer = ResultWriter(output_csv)
            writer.write(test_results)
        
        return test_results
    
    def _decode_sequence(self, sequence: List[int]) -> str:
        """Decode a sequence of token indices to string"""
        if hasattr(self, 'label_encoder') and self.label_encoder is not None:
            try:
                return decode_ids_to_phone_string(
                    self.label_encoder.decode_ndim,
                    sequence,
                    getattr(self.hparams, "context_phone_mode", "mono"),
                )
            except:
                pass
        return ' '.join(str(idx) for idx in sequence)

    def _metric_ind2lab(self):
        """Return ind2lab for metrics, optionally projecting context phones."""
        return make_context_phone_ind2lab(
            self.label_encoder.decode_ndim,
            getattr(self.hparams, "context_phone_mode", "mono"),
        )
    
    def _decode_tensor(self, tensor: torch.Tensor, length: Optional[torch.Tensor] = None) -> str:
        """Decode a tensor of token indices to string"""
        if length is not None:
            actual_len = int(length.item() * tensor.shape[0]) if length.item() <= 1.0 else int(length.item())
            tensor = tensor[:actual_len]
        
        indices = tensor.cpu().tolist()
        # Remove padding (typically 0)
        indices = [idx for idx in indices if idx != 0]
        return self._decode_sequence(indices)
    
    def _compute_per(self, prediction: str, target: str) -> float:
        """Compute Phoneme Error Rate between prediction and target"""
        pred_tokens = prediction.split()
        target_tokens = target.split()
        
        if len(target_tokens) == 0:
            return 0.0 if len(pred_tokens) == 0 else 1.0
        
        # Simple edit distance
        m, n = len(pred_tokens), len(target_tokens)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if pred_tokens[i-1] == target_tokens[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        return dp[m][n] / n
    
    def _compute_mpd_single(
        self,
        prediction: str,
        canonical: str,
        perceived: str
    ) -> Dict[str, Any]:
        """
        Compute MPD metrics for a single sample.
        
        Returns dict with: correct, precision, recall, f1
        """
        pred_tokens = set(prediction.split())
        canon_tokens = set(canonical.split())
        perc_tokens = set(perceived.split())
        
        # Mispronunciations are tokens in perceived but not in canonical
        # (simplified version - actual MPD may be more complex)
        gt_mispro = perc_tokens - canon_tokens
        pred_mispro = pred_tokens - canon_tokens
        
        if len(gt_mispro) == 0 and len(pred_mispro) == 0:
            return {'correct': True, 'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
        
        if len(gt_mispro) == 0:
            return {'correct': False, 'precision': 0.0, 'recall': 1.0, 'f1': 0.0}
        
        if len(pred_mispro) == 0:
            return {'correct': False, 'precision': 1.0, 'recall': 0.0, 'f1': 0.0}
        
        tp = len(pred_mispro & gt_mispro)
        precision = tp / len(pred_mispro) if pred_mispro else 0.0
        recall = tp / len(gt_mispro) if gt_mispro else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'correct': pred_mispro == gt_mispro,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    # ========================================================================
    # Flexible Test Stage Handler
    # ========================================================================
    
    def test_with_flexible_output(
        self,
        test_set,
        output_folder: str,
        min_key: str = "PER",
        max_key: str = "mpd_f1"
    ):
        """
        Run test stage with flexible output based on available reference data.
        
        This method:
        1. Automatically detects if reference/canonical/perceived data is available
        2. Computes metrics only when reference data exists
        3. Generates CSV with appropriate columns
        4. Saves summary statistics if metrics computed
        
        Args:
            test_set: Test dataset/dataloader
            output_folder: Folder to save results
            min_key: Metric to minimize for checkpoint selection
            max_key: Metric to maximize for checkpoint selection
        """
        # Ensure output folder exists
        os.makedirs(output_folder, exist_ok=True)
        
        # Load best checkpoint
        if self.checkpointer is not None:
            self.checkpointer.recover_if_possible(
                min_key=min_key,
                max_key=max_key
            )
        
        # Prepare for test
        self.on_stage_start(sb.Stage.TEST, epoch=None)
        self.modules.eval()
        
        test_results = TestResults()
        total_loss = 0.0
        batch_count = 0
        
        # Check first batch to determine data availability
        first_batch = next(iter(test_set))
        has_target = hasattr(first_batch, 'phn_encoded_target')
        has_canonical = hasattr(first_batch, 'phn_encoded_canonical')
        has_perceived = hasattr(first_batch, 'phn_encoded_perceived')
        
        logging.info(f"📊 Test data availability:")
        logging.info(f"   - Target reference: {'✓' if has_target else '✗'}")
        logging.info(f"   - Canonical: {'✓' if has_canonical else '✗'}")
        logging.info(f"   - Perceived: {'✓' if has_perceived else '✗'}")
        
        from tqdm import tqdm
        
        with torch.no_grad():
            for batch in tqdm(test_set, desc="Testing"):
                batch = batch.to(self.device)
                
                # Get predictions
                predictions = self.compute_forward(batch, stage=sb.Stage.TEST)
                
                # Compute loss only if we have targets
                if has_target:
                    loss = self.compute_objectives(predictions, batch, stage=sb.Stage.TEST)
                    total_loss += loss.item()
                    batch_count += 1
                
                # Get batch results
                batch_results = self.inference_batch(
                    batch,
                    compute_metrics=(has_target or (has_canonical and has_perceived))
                )
                
                for result in batch_results:
                    test_results.add_result(result)
        
        # Update test_results flags
        test_results.has_reference = has_target
        test_results.has_canonical = has_canonical
        test_results.has_perceived = has_perceived
        
        # Compute summary statistics
        if has_target:
            test_results.overall_per = self.per_metrics.summarize("error_rate")
            avg_loss = total_loss / max(1, batch_count)
        
        if has_canonical and has_perceived:
            test_results.overall_mpd_f1 = self.mpd_metrics.summarize("mpd_f1")
            mpd_summary = self.mpd_metrics.summarize()
            if isinstance(mpd_summary, dict):
                test_results.overall_mpd_precision = mpd_summary.get('precision', None)
                test_results.overall_mpd_recall = mpd_summary.get('recall', None)
        
        # Write CSV results
        csv_path = os.path.join(output_folder, "test_results.csv")
        writer = ResultWriter(csv_path)
        writer.write(test_results)
        
        # Log results
        logging.info("=" * 50)
        logging.info("Test Results Summary")
        logging.info("=" * 50)
        logging.info(f"Total samples: {len(test_results.results)}")
        
        if has_target:
            logging.info(f"Average Loss: {avg_loss:.4f}")
            logging.info(f"Overall PER: {test_results.overall_per:.4f}")
        
        if has_canonical and has_perceived and test_results.overall_mpd_f1 is not None:
            logging.info(f"Overall MPD F1: {test_results.overall_mpd_f1:.4f}")
        
        logging.info(f"Results saved to: {csv_path}")
        
        return test_results

    def _load_token_names(self):
        """Load token names from label encoder file"""
        token_names = {}
        label_encoder_path = os.path.join(self.hparams.save_folder, "label_encoder.txt")
        if os.path.exists(label_encoder_path):
            with open(label_encoder_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and '=>' in line:
                        try:
                            name, idx = line.split('=>')
                            name = name.strip().strip("'\"")
                            idx = idx.strip()
                            token_names[int(idx)] = name
                        except:
                            pass
        return token_names
    
    def on_stage_start(self, stage, epoch):
        """Gets called when a stage starts"""
        self.ctc_metrics = self.hparams.ctc_stats()
        if hasattr(self.hparams, "augmentation"):
            self.modules.perceived_ssl.model.config.apply_spec_augment = True
        
        # Reset CR-CTC loss tracking at the start of each training epoch
        if stage == sb.Stage.TRAIN:
            self.cr_loss_sum = 0.0
            self.cr_loss_count = 0
            self.ctc_loss_sum = 0.0
            self.ctc_loss_count = 0
            
            # Reset MPD loss tracking at the start of each training epoch
            self.mpd_binary_loss_sum = 0.0
            self.mpd_binary_loss_count = 0
            self.mpd_cls_loss_sum = 0.0
            self.mpd_cls_loss_count = 0
            self.ctc_loss_sum = 0.0
            self.ctc_loss_count = 0
        
        if stage != sb.Stage.TRAIN:
            self.per_metrics = self.hparams.per_stats()
            self.mpd_metrics = MpdStats()
        
        # Initialize test results collector for CSV output
        if stage == sb.Stage.TEST:
            self.test_results_for_csv = TestResults()
    
    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch"""
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
            
            # Compute average CR-CTC losses for the epoch
            if self.loss_manager.loss_type == 'crctc' or self.loss_manager.loss_type == 'crottc':
                # import pdb; pdb.set_trace()
                self.avg_cr_loss = (self.cr_loss_sum / max(1, self.cr_loss_count)
                                   if self.cr_loss_count > 0 else 0.0)
                self.avg_ctc_loss_train = (self.ctc_loss_sum / max(1, self.ctc_loss_count)
                                          if self.ctc_loss_count > 0 else 0.0)
        else:
            per = self.per_metrics.summarize("error_rate")
            mpd_f1 = self.mpd_metrics.summarize("mpd_f1")
        
        if stage == sb.Stage.VALID:
            # Prepare train stats
            train_stats = {"loss": self.train_loss}
            if self.loss_manager.loss_type == 'crctc' or self.loss_manager.loss_type == 'crottc':
                train_stats["ctc_loss"] = getattr(self, 'avg_ctc_loss_train', 0.0)
                train_stats["cr_loss"] = getattr(self, 'avg_cr_loss', 0.0)
            
            # Add MPD losses to train stats if available
            if self.mpd_binary_loss_count > 0:
                train_stats["mpd_binary_loss"] = self.mpd_binary_loss_sum / self.mpd_binary_loss_count
            if self.mpd_cls_loss_count > 0:
                train_stats["mpd_cls_loss"] = self.mpd_cls_loss_sum / self.mpd_cls_loss_count
            
            # Prepare valid stats
            valid_stats = {
                "loss": stage_loss,
                "ctc_loss": self.ctc_metrics.summarize("average"),
                "PER": per,
                "mpd_f1": mpd_f1
            }
            
            self.hparams.train_logger.log_stats(
                stats_meta={
                    "epoch": epoch,
                    "lr_adam": self.adam_optimizer.param_groups[0]["lr"],
                    "lr_pretrained": self.pretrained_opt_class.param_groups[0]["lr"],
                },
                train_stats=train_stats,
                valid_stats=valid_stats,
            )
            
            # Save checkpoints
            improved = False
            if per < self.best_per:
                self.best_per = per
                improved = True
            
            if mpd_f1 > self.best_mpd_f1:
                self.best_mpd_f1 = mpd_f1
                improved = True
            
            ckpt_name = f"{epoch:03d}_PER_{per:.4f}_F1_{mpd_f1:.4f}.ckpt"
            max_save_models = getattr(self.hparams, 'max_save_models', 3)
            self.checkpointer.save_and_keep_only(
                meta={"PER": per, "mpd_f1": mpd_f1, "epoch": epoch},
                name=ckpt_name,
                num_to_keep=max_save_models,
                importance_keys=[lambda ckpt: (-ckpt.meta["PER"], ckpt.meta["mpd_f1"])]
            )
            
            # Early stopping
            if stage_loss < self.best_valid_loss or len(self.best_valid_loss_list) < 10:
                if stage_loss < self.best_valid_loss:
                    self.best_valid_loss = stage_loss
                    improved = True
                self.best_valid_loss_list.append((stage_loss, epoch, ckpt_name))
                self.best_valid_loss_list.sort(key=lambda x: x[0])
                self.best_valid_loss_list = self.best_valid_loss_list[:10]
            
            if improved:
                self.no_improve_epochs = 0
                self.last_improved_epoch = epoch
            else:
                self.no_improve_epochs += 1
            
            # Wandb logging
            wandb_dict = {
                "epoch": epoch,
                "train_loss": self.train_loss,
                "valid_loss": stage_loss,
                "valid_ctc_loss": self.ctc_metrics.summarize("average"),
                "PER": per,
                "mpd_f1": mpd_f1,
            }
            
            # Add CR-CTC specific metrics
            if self.loss_manager.loss_type == 'crctc' or self.loss_manager.loss_type == 'crottc':
                wandb_dict["train_ctc_loss"] = getattr(self, 'avg_ctc_loss_train', 0.0)
                wandb_dict["train_cr_loss"] = getattr(self, 'avg_cr_loss', 0.0)
            
            # Add MPD losses to wandb logging
            if self.mpd_binary_loss_count > 0:
                wandb_dict["train_mpd_binary_loss"] = self.mpd_binary_loss_sum / self.mpd_binary_loss_count
            if self.mpd_cls_loss_count > 0:
                wandb_dict["train_mpd_cls_loss"] = self.mpd_cls_loss_sum / self.mpd_cls_loss_count
            
            wandb.log(wandb_dict, step=epoch)
            
            # Early stop if patience exceeded
            if self.no_improve_epochs >= self.patience:
                print(f"Early stopping at epoch {epoch}, no improvement for {self.patience} epochs.")
                raise KeyboardInterrupt("Early stopping triggered")
        
        if stage == sb.Stage.TEST:
            # Flexible test output based on available metrics
            test_stats = {"loss": stage_loss}
            
            # Check if per_metrics and mpd_metrics were populated
            has_per_metrics = hasattr(self, 'per_metrics') and len(self.per_metrics.scores) > 0
            has_mpd_metrics = hasattr(self, 'mpd_metrics') and len(self.mpd_metrics.scores) > 0
            
            if has_per_metrics:
                per = self.per_metrics.summarize("error_rate")
                test_stats["PER"] = per
            
            if has_mpd_metrics:
                mpd_f1 = self.mpd_metrics.summarize("mpd_f1")
                test_stats["mpd_f1"] = mpd_f1
            
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=test_stats,
            )
            
            # Write detailed stats files if metrics available
            if has_per_metrics:
                per_file = getattr(self.hparams, 'per_file', os.path.join(self.hparams.output_folder, 'per_stats.txt'))
                with open(per_file, "w") as w:
                    self.per_metrics.write_stats(w)
                logging.info(f"✅ PER stats written to: {per_file}")
            
            if has_mpd_metrics:
                mpd_file = getattr(self.hparams, 'mpd_file', os.path.join(self.hparams.output_folder, 'mpd_stats.txt'))
                with open(mpd_file, "w") as m:
                    self.mpd_metrics.write_stats(m)
                logging.info(f"✅ MPD stats written to: {mpd_file}")
            
            # Generate CSV results
            if hasattr(self, 'test_results_for_csv') and self.test_results_for_csv:
                csv_path = getattr(self.hparams, 'test_csv_file', 
                                   os.path.join(self.hparams.output_folder, 'test_results.csv'))
                writer = ResultWriter(csv_path)
                writer.write(self.test_results_for_csv)
    
    def fit_batch(self, batch):
        """Fit one batch"""
        if self.hparams.auto_mix_prec:
            self.pretrained_opt_class.zero_grad()
            self.adam_optimizer.zero_grad()
            
            with torch.amp.autocast("cuda"):
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            
            self.scaler.scale(loss / self.hparams.gradient_accumulation).backward()
            self.scaler.unscale_(self.pretrained_opt_class)
            self.scaler.unscale_(self.adam_optimizer)
            
            if self.check_gradients(loss):
                if any(p.requires_grad for p in self.pretrained_opt_class.param_groups[0]['params']):
                    self.scaler.step(self.pretrained_opt_class)
                if any(p.requires_grad for p in self.adam_optimizer.param_groups[0]['params']):
                    self.scaler.step(self.adam_optimizer)
            
            self.scaler.update()
        else:
            outputs = self.compute_forward(batch, sb.Stage.TRAIN)
            loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            (loss / self.hparams.gradient_accumulation).backward()
            
            if self.step % self.hparams.gradient_accumulation == 0:
                if self.check_gradients(loss):
                    self.pretrained_opt_class.step()
                    self.adam_optimizer.step()
                
                self.pretrained_opt_class.zero_grad()
                self.adam_optimizer.zero_grad()
        
        return loss.detach().cpu()
    
    def init_optimizers(self):
        """Initialize optimizers"""
        self.adam_optimizer = self.hparams.adam_opt_class(
            self.hparams.model.parameters(),
        )
        self.pretrained_opt_class = self.hparams.pretrained_opt_class(
            self.modules.perceived_ssl.parameters(),
        )
        
        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("adam_opt", self.adam_optimizer)
            self.checkpointer.add_recoverable("pretrained_opt", self.pretrained_opt_class)
            self.checkpointer.add_recoverable("tokenizer", self.label_encoder)
    
    def on_fit_start(self):
        """Gets called at the beginning of fit()"""
        self._compile()
        self._wrap_distributed()
        self.init_optimizers()
        
        # Load pretrained components if specified
        # import pdb; pdb.set_trace()
        resume_from_pretrainer = getattr(self.hparams, 'resume_from_pretrainer', None)
        if resume_from_pretrainer is not None:
            resume_from_paths = resume_from_pretrainer.collect_files(default_source=self.hparams.resume_from)
            resume_from_pretrainer.load_collected()
            logging.info(f"✅ Resumed from pretrainer: {self.hparams.resume_from}")
        
        # if getattr(self.hparams, 'load_pretrained_components', False):
        #     pretrained_path = getattr(self.hparams, 'pretrained_model_path', '')
        #     components = getattr(self.hparams, 'components_to_load', ['ssl', 'enc', "ctc_head"])
        #     freeze_loaded = getattr(self.hparams, 'freeze_loaded_components', True)
            
        #     if pretrained_path and os.path.exists(pretrained_path):
        #         try:
        #             self.load_pretrained_components(
        #                 checkpoint_path=pretrained_path,
        #                 components_to_load=components,
        #                 freeze_loaded=freeze_loaded
        #             )
        #         except Exception as e:
        #             print(f"❌ Failed to load pretrained components: {e}")
        
        elif self.checkpointer is not None:
            # self.checkpointer.recover_if_possible(min_key="PER")
            self.checkpointer.recover_if_possible(min_key="PER", max_key="mpd_f1"
            )
    
        
    def load_pretrained_components(self, checkpoint_path, components_to_load=None, freeze_loaded=True):
        """Load specific components from a pretrained model"""
        if components_to_load is None:
            components_to_load = ['ssl']
        
        print(f"\n🔄 Loading pretrained components from: {checkpoint_path}")
        print(f"   Components to load: {components_to_load}")
        
        from speechbrain.utils.parameter_transfer import Pretrainer
        
        pretrainer = Pretrainer(
            collect_in=self.hparams.pretrained_model_path,
            loadables={
                "perceived_ssl": self.modules.perceived_ssl,
                "model": self.hparams.model,
            },
            paths={
                "perceived_ssl": "perceived_ssl.ckpt",
                "model": "model.ckpt",
            },
        )
        
        paths = pretrainer.collect_files(default_source=self.hparams.pretrained_model_path)
        pretrainer.load_collected()
        
        # Freeze loaded components if requested
        if freeze_loaded:
            for component in components_to_load:
                if component == 'ssl':
                    for param in self.modules.perceived_ssl.parameters():
                        param.requires_grad = False
                    print("   🔒 SSL model frozen")
                elif component == 'enc':
                    if hasattr(self.modules, 'enc'):
                        for param in self.modules.enc.parameters():
                            param.requires_grad = False
                        print("   🔒 Encoder projection frozen")
                elif component == 'ctc_head':
                    for param in self.modules.ctc_lin.parameters():
                        param.requires_grad = False
                    print("   🔒 CTC head frozen")
