import os
import sys
import torch
import torch.nn
import logging
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
# from mpd_eval_v3 import MpdStats
from mpd_eval_v4 import MpdStats
import librosa
import json
import wandb
import time
import torchaudio
from speechbrain.inference.text import GraphemeToPhoneme
from torch.nn.functional import kl_div
from speechbrain.nnet.loss.guidedattn_loss import GuidedAttentionLoss
import re

# Placeholder: gather CTC-aligned representations for each token
def gather_ctc_aligned_reps(encoded, targets, target_lens):
    # Placeholder: ideally, implement alignment logic here
    # For now, simply average pool the encoded features into L segments
    B, T, D = encoded.size()
    L = targets.size(1)
    avg_len = T // L
    reps = []
    for i in range(L):
        reps.append(encoded[:, i * avg_len : (i + 1) * avg_len].mean(dim=1))
    return torch.stack(reps, dim=1)  # [B, L, D]

def simplify_phoneme(p):
    return re.sub(r"\d", "", p).lower()  # e.g., 'AA1' -> 'aa'

class PhnMonoSSLModel(sb.Brain):
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
        if getattr(self.modules, "perceived_ssl", None) is not None:
            self.modules.perceived_ssl.to(self.device)
        if getattr(self.modules, "canonical_ssl", None) is not None:
            self.modules.canonical_ssl.to(self.device)
            
        self.best_valid_loss = float('inf')
        self.best_valid_loss_list = []  # List of (valid_loss, epoch, ckpt_name)
    
    # Batch = namedtuple("Batch", ["features", "feature_lengths", "targets", "target_lengths"])
    
    # def align(self, batch: Batch, tokens: List[int]):
    #     # Implemented according to:
    #     # https://pytorch.org/audio/main/tutorials/ctc_forced_alignment_api_tutorial.html
    #     assert batch.features.size(0) == 1, "Only single utterance is supported for forced alignment"

    #     emission = self.forward(batch)
    #     if emission is None:
    #         return None
        
    #     targets = torch.tensor([tokens], dtype=torch.int32, device="cpu")
    #     alignments, scores = torchaudio.functional.forced_align(emission, targets, blank=self.blank_idx)

    #     alignments, scores = alignments[0], scores[0]  # remove batch dimension for simplicity
    #     scores = scores.exp()  # convert back to probability
    #     return alignments, scores
    def create_attention_mask_from_input_sequence(self, input_sequence):
        ''' create attention mask from input sequence 
                input_sequence: tensor of shape (batch_size) containing the length of each sequence in the batch
            output:
                attention_mask: tensor of shape (batch_size, max_length in input_sequence) 
        '''
        tampon = (torch.arange(1,max(input_sequence)+1)).repeat(len(input_sequence), 1).to('cuda')

        attention_mask = (tampon <= input_sequence.unsqueeze(1)).float()

        return attention_mask
    
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
        # if self.modules.perceived_ssl.feature_extractor.return_attention_mask:
        #     attn_mask = make_attn_mask(wavs, wav_lens)
        # else:
        #     attn_mask = None
        
        feats = self.modules.perceived_ssl(wavs)
        
        x = self.modules.enc(feats)
        
        if getattr(self.modules, "ConformerEncoder", None) is not None:
            from speechbrain.nnet.attention import RelPosEncXL, RelPosMHAXL, RoPEMHA 
            pos_emb = RelPosEncXL(emb_dim=self.hparams.dnn_neurons)(x).to(self.device)
            # import pdb; pdb.set_trace()
            x, _ = self.modules.ConformerEncoder(x, pos_embs=pos_emb)
        # import pdb; pdb.set_trace()
        if getattr(self.modules, "ZipformerEncoder", None) is not None:
            # from speechbrain.nnet.attention import RelPosEncXL, RelPosMHAXL, RoPEMHA 
            # pos_emb = RelPosEncXL(emb_dim=self.hparams.dnn_neurons)(x.permute(1, 0, 2)).to(self.device)
            x = self.modules.ZipformerEncoder(x.permute(1, 0, 2)) # [T, B, D]
            x = x.permute(1, 0, 2) # [B, T, D]
        
        # import pdb; pdb.set_trace()
        # Get RVQ if exists
        if getattr(self.modules, "RVQ", None) is not None:
            # Expect [B, C, T]
            x = x.transpose(1, 2)  # [B, T, C] -> [B, C, T]
            discrete_embeddings, codes, latents, commitment_loss, codebook_loss = self.modules.RVQ(x)
            x = discrete_embeddings.transpose(1, 2)  # [B, C, T] -> [B, T, C]
        # output layer for ctc log-probabilities
        logits = self.modules.ctc_lin(x)
        p_ctc = self.hparams.log_softmax(logits) # (B, T, C)
        
        # OTTC implementation for lm_weight:
        # import pdb; pdb.set_trace()
        if hasattr(self.modules, "lm_weight"):
            if stage != sb.Stage.TEST:
                targets, target_lens = batch.phn_encoded_target
                labels_mask = (targets != self.hparams.blank_index).float()  # (B, L)
                weights_logits = self.modules.lm_weight(x)
                # make mask with wav_lens
                lens_abs = (wav_lens * feats.shape[-2]).int()
                # import pdb; pdb.set_trace()
                output_mask = self.create_attention_mask_from_input_sequence(lens_abs) # torch.Size([64, 389])
                # output_mask = self.create_attention_mask_from_input_sequence(input_lengths)   
                import torch.nn.functional as F
                weights_logits = F.softmax(weights_logits.squeeze().masked_fill_(output_mask == 0, - torch.inf ))
                weights_labels =   labels_mask /  labels_mask.sum(axis = 1)[:,None] #weights_logits.sum(axis = 1)[:,None] *         

                # get label mask TODO
                # labels_mask = (labels >= 0).float()

                # import pdb; pdb.set_trace()
                return p_ctc, logits, weights_logits, weights_labels, wav_lens
        
        # Support decoding and alignment monitoring for VALID and TEST stages
        if stage == sb.Stage.VALID or stage == sb.Stage.TEST:
            from torchaudio.functional import forced_align, merge_tokens
            from pathlib import Path
            import matplotlib.pyplot as plt
            import json
            
            # For VALID: monitor a fixed sample across epochs
            # For TEST: save alignments for all samples
            if stage == sb.Stage.VALID:
                # Initialize fixed sample ID if not exists (use first batch's first ID)
                if not hasattr(self, '_fixed_sample_id'):
                    self._fixed_sample_id = batch.id[0]
                    print(f"📌 VALID 固定监控样本 ID: {self._fixed_sample_id}")
                
                # Only process if current batch contains the fixed sample ID
                if self._fixed_sample_id not in batch.id:
                    if getattr(self.modules, "RVQ", None) is not None:
                        return p_ctc, wav_lens, commitment_loss, codebook_loss
                    return p_ctc, wav_lens
                
                sample_indices = [batch.id.index(self._fixed_sample_id)]
                epoch = self.hparams.epoch_counter.current
                alignment_dir = os.path.join(self.hparams.output_folder, "alignment_monitoring")
                output_base = os.path.join(alignment_dir, f"epoch_{epoch:03d}")
                stage_label = f"VALID-Epoch{epoch}"
                
            else:  # TEST stage
                # Process all samples in the batch
                sample_indices = list(range(len(batch.id)))
                alignment_dir = os.path.join(self.hparams.output_folder, "test_decoding")
                output_base = alignment_dir
                stage_label = "TEST"
            
            # Create output directory
            os.makedirs(output_base, exist_ok=True)
            
            # Initialize list for collecting all predictions (for TEST stage)
            if stage == sb.Stage.TEST:
                if not hasattr(self, '_test_predictions'):
                    self._test_predictions = []
            
            # Process selected samples
            for loop_idx, real_batch_idx in enumerate(sample_indices):
                sample_id = batch.id[real_batch_idx]
                output_dir = output_base
                
                # Extract single sample
                p_ctc_sample = p_ctc[real_batch_idx:real_batch_idx+1]  # [1, T, C]
                wav_lens_sample = wav_lens[real_batch_idx:real_batch_idx+1]  # [1]
                
                # Align targeting tokens
                targets, target_lens = batch.phn_encoded_target
                targets_sample = targets[real_batch_idx:real_batch_idx+1]
                target_lens_sample = target_lens[real_batch_idx:real_batch_idx+1]
                
                # Remove padding from targets
                actual_target_len = int(target_lens_sample[0].item() * targets_sample.shape[-1])
                targets_sample_no_pad = targets_sample[:, :actual_target_len]
                
                # Calculate actual input length
                actual_input_len = int(wav_lens_sample[0].item() * p_ctc_sample.shape[1])
                
                # Greedy decode prediction
                predict_target = sb.decoders.ctc_greedy_decode(
                    p_ctc_sample, wav_lens_sample, blank_id=self.hparams.blank_index
                )
                
                # Prepare output filename
                sample_stem = Path(sample_id).stem
                
                # Ground truth forced alignment
                try:
                    forced_alignments, scores = forced_align(
                        log_probs=p_ctc_sample,
                        targets=targets_sample_no_pad,
                        target_lengths=torch.tensor([actual_target_len], dtype=torch.int32, device=self.device),
                        input_lengths=torch.tensor([actual_input_len], dtype=torch.int32, device=self.device),
                        blank=self.hparams.blank_index
                    )
                    forced_alignments = forced_alignments[0]
                    scores = scores[0].exp()
                    aligned_tokens_gt = merge_tokens(forced_alignments, scores)
                    
                except Exception as e:
                    if stage == sb.Stage.TEST:
                        print(f"⚠️  [{sample_id}] Forced alignment failed: {e}")
                    aligned_tokens_gt = None
                    scores = None
                
                # Predicted forced alignment
                predict_target_sample = predict_target[0] if predict_target else []
                aligned_tokens_pred = None
                p_scores = None
                
                if len(predict_target_sample) > 0:
                    predict_target_tensor = torch.tensor([predict_target_sample], dtype=torch.int32, device=self.device)
                    try:
                        p_forced_alignments, p_scores = forced_align(
                            log_probs=p_ctc_sample,
                            targets=predict_target_tensor,
                            target_lengths=torch.tensor([len(predict_target_sample)], dtype=torch.int32, device=self.device),
                            input_lengths=torch.tensor([actual_input_len], dtype=torch.int32, device=self.device),
                            blank=self.hparams.blank_index
                        )
                        p_forced_alignments = p_forced_alignments[0]
                        p_scores = p_scores[0].exp()
                        aligned_tokens_pred = merge_tokens(p_forced_alignments, p_scores)
                    except Exception as e:
                        if stage == sb.Stage.TEST:
                            print(f"⚠️  [{sample_id}] Predicted alignment failed: {e}")
                
                # Save alignment plots
                if stage == sb.Stage.VALID:
                    # VALID stage: separate plots for GT and Pred
                    if aligned_tokens_gt is not None and scores is not None:
                        try:
                            fig_gt = self.plot_scores(aligned_tokens_gt, scores)
                            fig_gt.suptitle(f"GT Alignment - {sample_stem} ({stage_label})")
                            gt_path = os.path.join(output_dir, f"gt_alignment_{sample_stem}.png")
                            fig_gt.savefig(gt_path, dpi=150, bbox_inches='tight')
                            plt.close(fig_gt)
                        except Exception as e:
                            print(f"⚠️  [{sample_id}] Failed to save GT plot: {e}")
                    
                    if aligned_tokens_pred is not None and p_scores is not None:
                        try:
                            fig_pred = self.plot_scores(aligned_tokens_pred, p_scores)
                            fig_pred.suptitle(f"Pred Alignment - {sample_stem} ({stage_label})")
                            pred_path = os.path.join(output_dir, f"pred_alignment_{sample_stem}.png")
                            fig_pred.savefig(pred_path, dpi=150, bbox_inches='tight')
                            plt.close(fig_pred)
                        except Exception as e:
                            print(f"⚠️  [{sample_id}] Failed to save pred plot: {e}")
                
                elif stage == sb.Stage.TEST:
                    # TEST stage: combined comparison plot
                    if aligned_tokens_gt is not None and scores is not None:
                        try:
                            # Extract speaker from sample_id (e.g., "/path/L2-ARCTIC/TLV/..." -> "TLV")
                            speaker_id = "unknown"
                            if "/" in sample_id:
                                parts = sample_id.split("/")
                                # Look for known patterns: L2-ARCTIC/SPEAKER_ID or similar
                                for i, part in enumerate(parts):
                                    if part in ["L2-ARCTIC", "L2ARCTIC"] and i+1 < len(parts):
                                        speaker_id = parts[i+1]
                                        break
                            
                            comparison_title = f"🎤 {speaker_id} - {sample_stem}\n GT vs Predicted Alignment"
                            
                            fig_compare = self.plot_alignment_comparison(
                                aligned_tokens_gt, scores,
                                aligned_tokens_pred if aligned_tokens_pred is not None else [],
                                p_scores if p_scores is not None else None,
                                title=comparison_title
                            )
                            compare_path = os.path.join(output_dir, f"alignment_compare_{sample_stem}.png")
                            fig_compare.savefig(compare_path, dpi=150, bbox_inches='tight')
                            plt.close(fig_compare)
                            print(f"💾 Saved comparison plot: {compare_path}")
                        except Exception as e:
                            print(f"⚠️  [{sample_id}] Failed to save comparison plot: {e}")
                
                # For TEST stage: save decoded sequence and metadata
                if stage == sb.Stage.TEST:
                    # Decode sequence
                    decoded_seq = self.label_encoder.decode_ndim(predict_target_sample) if len(predict_target_sample) > 0 else ""
                    
                    # Load target and canonical phonemes if available
                    try:
                        canonicals, canonical_lens = batch.phn_encoded_canonical
                        perceiveds, perceived_lens = batch.phn_encoded_perceived
                        canonical_sample = canonicals[real_batch_idx:real_batch_idx+1]
                        perceived_sample = perceiveds[real_batch_idx:real_batch_idx+1]
                        canonical_len = canonical_lens[real_batch_idx].item()
                        perceived_len = perceived_lens[real_batch_idx].item()
                        
                        canonical_decoded = self.label_encoder.decode_ndim(canonical_sample[0, :int(canonical_len*canonical_sample.shape[-1])].tolist())
                        perceived_decoded = self.label_encoder.decode_ndim(perceived_sample[0, :int(perceived_len*perceived_sample.shape[-1])].tolist())
                    except:
                        canonical_decoded = ""
                        perceived_decoded = ""
                    
                    # Save as JSON
                    test_record = {
                        "sample_id": sample_id,
                        "predicted": decoded_seq,
                        "canonical": canonical_decoded,
                        "perceived": perceived_decoded,
                        "num_predicted": len(predict_target_sample),
                    }
                    
                    json_path = os.path.join(output_dir, f"decode_{sample_stem}.json")
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(test_record, f, ensure_ascii=False, indent=2)
                    
                    self._test_predictions.append(test_record)
        
        if getattr(self.modules, "RVQ", None) is not None:
            return p_ctc, wav_lens, commitment_loss, codebook_loss
        return p_ctc, wav_lens
    
    def plot_scores(self, word_spans, scores):
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots()
        span_xs, span_hs = [], []
        ax.axvspan(word_spans[0].start - 0.05, word_spans[-1].end + 0.05, facecolor="paleturquoise", edgecolor="none", zorder=-1)
        # for t_span in word_spans:
        for span in word_spans:
            for t in range(span.start, span.end):
                span_xs.append(t + 0.5)
                span_hs.append(scores[t].item())
            ax.annotate(self.label_encoder.decode_ndim(span.token), (span.start, -0.07))
            ax.axvspan(span.start - 0.05, span.end + 0.05, facecolor="mistyrose", edgecolor="none", zorder=-1)
        ax.bar(span_xs, span_hs, color="lightsalmon", edgecolor="coral")
        ax.set_title("Frame-level scores and word segments")
        ax.set_ylim(-0.1, None)
        ax.grid(True, axis="y")
        ax.axhline(0, color="black")
        fig.tight_layout()
        return fig
    
    def plot_alignment_comparison(self, word_spans_gt, scores_gt, word_spans_pred, scores_pred, title="GT vs Predicted"):
        """
        Plot GT and Predicted alignment side by side for comparison.
        
        Arguments
        ---------
        word_spans_gt : list of namedtuples
            Ground truth word/token spans
        scores_gt : tensor
            Ground truth alignment scores
        word_spans_pred : list of namedtuples or None
            Predicted word/token spans
        scores_pred : tensor or None
            Predicted alignment scores
        title : str
            Title for the figure
            
        Returns
        -------
        fig : matplotlib figure
            Figure with comparison plots
        """
        from matplotlib import pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        
        # Determine layout
        if word_spans_pred is not None and len(word_spans_pred) > 0:
            fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        else:
            fig, axes = plt.subplots(1, 1, figsize=(14, 4))
            axes = [axes]
        
        # Plot GT alignment
        ax = axes[0]
        span_xs, span_hs = [], []
        if len(word_spans_gt) > 0:
            ax.axvspan(word_spans_gt[0].start - 0.05, word_spans_gt[-1].end + 0.05, 
                       facecolor="paleturquoise", edgecolor="none", zorder=-1)
            for span in word_spans_gt:
                for t in range(span.start, span.end):
                    span_xs.append(t + 0.5)
                    span_hs.append(scores_gt[t].item())
                token_name = self.label_encoder.decode_ndim(span.token)
                ax.annotate(token_name, (span.start, -0.07), fontsize=8)
                ax.axvspan(span.start - 0.05, span.end + 0.05, facecolor="mistyrose", edgecolor="none", zorder=-1)
        ax.bar(span_xs, span_hs, color="lightsalmon", edgecolor="coral", alpha=0.8)
        ax.set_title("🎯 Ground Truth Alignment", fontsize=11, fontweight='bold')
        ax.set_ylabel("Score", fontsize=10)
        ax.set_ylim(-0.1, None)
        ax.grid(True, axis="y", alpha=0.3)
        ax.axhline(0, color="black", linewidth=0.8)
        
        # Plot Predicted alignment (if available)
        if len(axes) > 1:
            ax = axes[1]
            if word_spans_pred is not None and len(word_spans_pred) > 0 and scores_pred is not None:
                span_xs_pred, span_hs_pred = [], []
                ax.axvspan(word_spans_pred[0].start - 0.05, word_spans_pred[-1].end + 0.05, 
                           facecolor="lightgreen", edgecolor="none", zorder=-1, alpha=0.5)
                for span in word_spans_pred:
                    for t in range(span.start, span.end):
                        span_xs_pred.append(t + 0.5)
                        span_hs_pred.append(scores_pred[t].item())
                    token_name = self.label_encoder.decode_ndim(span.token)
                    ax.annotate(token_name, (span.start, -0.07), fontsize=8)
                    ax.axvspan(span.start - 0.05, span.end + 0.05, facecolor="lightcyan", edgecolor="none", zorder=-1)
                ax.bar(span_xs_pred, span_hs_pred, color="lightgreen", edgecolor="green", alpha=0.8)
                ax.set_title("📊 Predicted Alignment", fontsize=11, fontweight='bold')
                ax.set_ylabel("Score", fontsize=10)
                ax.set_xlabel("Frame Index", fontsize=10)
                ax.set_ylim(-0.1, None)
                ax.grid(True, axis="y", alpha=0.3)
                ax.axhline(0, color="black", linewidth=0.8)
            else:
                ax.text(0.5, 0.5, "❌ Predicted alignment unavailable\n(empty or failed)", 
                       ha='center', va='center', fontsize=12, transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
        
        fig.suptitle(title, fontsize=13, fontweight='bold', y=0.995)
        plt.tight_layout()
        return fig
    
    def compute_objectives(self, predictions, batch, stage):
        "Given the network predictions and targets computed the NLL loss."
        if getattr(self.modules, "RVQ", None) is not None:
            p_ctc, wav_lens, commitment_loss, codebook_loss = predictions
        if getattr(self.modules, "lm_weight", None) is not None:
            if stage != sb.Stage.TEST:
                p_ctc, logits, weights_logits, weights_labels, wav_lens = predictions
            else:
                p_ctc, wav_lens = predictions
        else:
            p_ctc, wav_lens = predictions
        ids = batch.id
        targets, target_lens = batch.phn_encoded_target
        
        if stage != sb.Stage.TRAIN:
            canonicals, canonical_lens = batch.phn_encoded_canonical
            perceiveds, perceived_lens = batch.phn_encoded_perceived
        
            # Additional: BCE loss on binary mispronunciation prediction
            if self.hparams.training_target == "target":
                targets = targets
                target_lens = target_lens
            elif self.hparams.training_target == "canonical":
                targets = canonicals
                target_lens = canonical_lens
            elif self.hparams.training_target == "perceived":
                targets = perceiveds
                target_lens = perceived_lens        
        
        # support CTCLossWithLabelPriors, OTTC loss
        from utils.losses.CTCLossWithLabelPriors import CTCLossWithLabelPriors
        from utils.losses.ot_loss import batched_ottc_loss_bucketized
        
        # import pdb; pdb.set_trace()
        if type(self.hparams.ctc_cost) == CTCLossWithLabelPriors:
            # feature [T, B, D]
            # targets: [B, T]
            # src_length: [B], int
            # batch.target_lengths: [B], int
            
            p_ctc_ctclp = p_ctc.permute(1, 0, 2)  # (B, T, D) -> (T, B, C)
            abs_wav_lens = (wav_lens * p_ctc.shape[-2] ).to(torch.int32)
            abs_target_lens = (target_lens*targets.shape[-1]).to(torch.int32)
            step_type_dict = {
                sb.Stage.TRAIN: "train",
                sb.Stage.VALID: "val",
                sb.Stage.TEST: "test"
            }
            loss_ctc = self.hparams.ctc_cost(log_probs=p_ctc_ctclp,
                                             targets=targets,
                                            input_lengths=abs_wav_lens,
                                            target_lengths=abs_target_lens,
                                            step_type=step_type_dict[stage]
                                            )
        
        elif self.hparams.ctc_cost == batched_ottc_loss_bucketized:
            # import pdb; pdb.set_trace()
            if getattr(self.modules, "lm_weight", None) is not None:
                if stage != sb.Stage.TEST:
                    # OTTC loss computation
                    # from utils.losses.ot_loss import 
                    labels_mask = (targets != self.hparams.blank_index).float()  # (B, L)
                    one_hot_labels = torch.nn.functional.one_hot(targets, num_classes=self.hparams.output_neurons)
                    loss_ctc, _, _, _ = self.hparams.ctc_cost(x = logits,
                                                    y = one_hot_labels,
                                                    a = weights_logits,
                                                    b = weights_labels,
                                                    amask = None,
                                                    bmask = labels_mask,
                                                    euclidian = False,
                                                    jsd = False,
                                                    )
                else:
                    # vanilla CTC loss for decode
                    from speechbrain.nnet.losses import ctc_loss
                    loss_ctc = ctc_loss(p_ctc, targets, wav_lens, target_lens, blank_index=self.hparams.blank_index)
                    # loss_ctc = self.hparams.ctc_cost(p_ctc, targets, wav_lens, target_lens)

        else:
            # vanilla CTC loss
            loss_ctc = self.hparams.ctc_cost(p_ctc, targets, wav_lens, target_lens)
            
        
        if getattr(self.modules, "RVQ", None) is not None:
            loss = loss_ctc + (commitment_loss + codebook_loss)
        elif getattr(self.modules, "lm_weight", None) is not None:
            # TODO: add label smoothing loss?
            loss = loss_ctc
        else:
            loss = loss_ctc

        # Record losses for posterity
        if stage != sb.Stage.TRAIN:
            # Note: sb.decoders.ctc_greedy_decode will also remove padded tokens
            # that is, it return a list of list with different lengths
            sequence = sb.decoders.ctc_greedy_decode(
                p_ctc, wav_lens, blank_id=self.hparams.blank_index
            )
            if type(self.hparams.ctc_cost) == CTCLossWithLabelPriors:
                # Use standard CTC loss for logging purposes
                try:
                    self.ctc_metrics.append(ids, 
                        log_probs=p_ctc.permute(1, 0, 2),
                        targets=targets,
                        input_lengths=(wav_lens * p_ctc.shape[-2] ).to(torch.int32),
                        target_lengths=(target_lens*targets.shape[-1]).to(torch.int32)
                    )
                except:
                    self.ctc_metrics.append(ids, p_ctc, targets, wav_lens, target_lens)

            else:
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

    def _load_token_names(self):
        """Load token names from label encoder file."""
        token_names = {}
        label_encoder_path = os.path.join(self.hparams.save_folder, "label_encoder.txt")
        if os.path.exists(label_encoder_path):
            with open(label_encoder_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                                # Format: 'token' => index
                                if '=>' in line:
                                    name, idx = line.split('=>')
                                    name = name.strip().strip("'\"")
                                    idx = idx.strip()
                                    token_names[int(idx)] = name
                        except:
                            pass
        return token_names
    
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
            
            # Update label priors after each training epoch (if using CTCLossWithLabelPriors)
            from utils.CTCLossWithLabelPriors import CTCLossWithLabelPriors
            if isinstance(self.hparams.ctc_cost, CTCLossWithLabelPriors):
                if self.hparams.ctc_cost.log_priors_sum is not None and self.hparams.ctc_cost.num_samples > 0:
                    # Compute average log priors from accumulated statistics
                    new_log_priors = (
                        self.hparams.ctc_cost.log_priors_sum 
                        - torch.log(torch.tensor(float(self.hparams.ctc_cost.num_samples), device=self.hparams.ctc_cost.log_priors_sum.device))
                    )
                    
                    # ===== Detailed Prior Logging (inspired by TorchAudio implementation) =====
                    print(f"\n{'='*80}")
                    print(f"📊 Label Priors Update - Epoch {epoch}")
                    print(f"{'='*80}")
                    print(f"Total frames processed: {self.hparams.ctc_cost.num_samples}")
                    print(f"Prior scaling factor (α): {self.hparams.ctc_cost.prior_scaling_factor}")
                    
                    # Convert to probability space for easier interpretation
                    new_priors_prob = new_log_priors.exp()
                    
                    # Load token names for better readability
                    token_names = self._load_token_names()
                    
                    # Create dictionary: phoneme -> prior
                    priors_dict = {}
                    log_priors_dict = {}
                    priors_list = new_priors_prob[0].tolist()
                    log_priors_list = new_log_priors[0].tolist()
                    
                    for idx, (prior, log_prior) in enumerate(zip(priors_list, log_priors_list)):
                        token_name = token_names.get(idx, f"token_{idx}")
                        priors_dict[token_name] = prior
                        log_priors_dict[token_name] = log_prior
                    
                    # Print formatted priors (probability) as dictionary
                    print(f"\n📋 New priors (probability) - Dictionary format:")
                    # Sort by prior value (descending) for better visibility
                    sorted_priors = sorted(priors_dict.items(), key=lambda x: x[1], reverse=True)
                    for token_name, prior in sorted_priors[:10]:  # Show top 10
                        print(f"  {token_name:>10s}: {prior:.4f}")
                    print(f"  ... (showing top 10 of {len(priors_dict)} tokens)")
                    
                    # Also print the compact format (for backward compatibility)
                    print(f"\n📊 New priors (probability) - Compact format:")
                    print("  " + ", ".join([f"{p:.4f}" for p in priors_list]))
                    
                    # Print formatted log-priors
                    print(f"\n📊 New log-priors - Compact format:")
                    print("  " + ", ".join([f"{lp:.2f}" for lp in log_priors_list]))
                    
                    # If we have previous priors, show the change
                    if self.hparams.ctc_cost.log_priors is not None:
                        old_priors_prob = self.hparams.ctc_cost.log_priors.exp()
                        diff_percent = ((new_priors_prob - old_priors_prob) / old_priors_prob * 100)[0].tolist()
                        print(f"\nChange from previous epoch (%):")
                        print("  " + ", ".join([f"{d:+.2f}" for d in diff_percent]))
                        
                        # Highlight the most changed tokens
                        abs_diff = [abs(d) for d in diff_percent]
                        top_changed_indices = sorted(range(len(abs_diff)), key=lambda i: abs_diff[i], reverse=True)[:5]
                        print(f"\nTop 5 most changed tokens:")
                        
                        # Load token names from label encoder
                        try:
                            token_names = self._load_token_names()
                            for idx in top_changed_indices:
                                token_name = token_names.get(idx, f"token_{idx}")
                                print(f"  {token_name:>10s} (idx={idx:2d}): {old_priors_prob[0][idx].item():.4f} → {new_priors_prob[0][idx].item():.4f} ({diff_percent[idx]:+.2f}%)")
                        except Exception as e:
                            # Fallback if token names can't be loaded
                            for idx in top_changed_indices:
                                print(f"  Token {idx:2d}: {old_priors_prob[0][idx].item():.4f} → {new_priors_prob[0][idx].item():.4f} ({diff_percent[idx]:+.2f}%)")
                    
                    # Highlight blank token
                    blank_idx = self.hparams.blank_index
                    blank_prior = new_priors_prob[0][blank_idx].item()
                    print(f"\n🎯 Blank token (idx={blank_idx}) prior: {blank_prior:.4f} ({blank_prior*100:.2f}%)")
                    
                    # Apply threshold to prevent very small priors (following TorchAudio)
                    prior_threshold = -12.0
                    new_log_priors = torch.where(
                        new_log_priors < prior_threshold,
                        torch.tensor(prior_threshold, device=new_log_priors.device),
                        new_log_priors
                    )
                    num_clipped = (new_log_priors == prior_threshold).sum().item()
                    if num_clipped > 0:
                        print(f"⚠️  Clipped {num_clipped} priors to threshold {prior_threshold}")
                    
                    print(f"{'='*80}\n")
                    
                    # ==============================================
                    # EMA Update for Stable Priors
                    # ==============================================
                    if self.hparams.ctc_cost.ema_log_priors is None:
                        # First epoch: initialize EMA with current priors
                        self.hparams.ctc_cost.ema_log_priors = new_log_priors.clone()
                        print(f"📊 Initialized EMA priors (Epoch {epoch})")
                        print(f"   Momentum: {self.hparams.ctc_cost.prior_momentum}")
                    else:
                        # Subsequent epochs: apply exponential moving average
                        momentum = self.hparams.ctc_cost.prior_momentum
                        old_ema = self.hparams.ctc_cost.ema_log_priors
                        
                        # Compute EMA: new_ema = momentum * old_ema + (1 - momentum) * current
                        self.hparams.ctc_cost.ema_log_priors = (
                            momentum * old_ema + (1 - momentum) * new_log_priors
                        )
                        
                        # Report the smoothing effect
                        old_blank_prior = torch.exp(old_ema[0, blank_idx]).item()
                        current_blank_prior = new_priors_prob[0, blank_idx].item()
                        ema_blank_prior = torch.exp(self.hparams.ctc_cost.ema_log_priors[0, blank_idx]).item()
                        
                        print(f"\n📊 EMA Prior Update (Epoch {epoch}):")
                        print(f"   Blank Prior Evolution:")
                        print(f"     Current epoch:    {current_blank_prior:.6f} ({current_blank_prior*100:.2f}%)")
                        print(f"     Previous EMA:     {old_blank_prior:.6f} ({old_blank_prior*100:.2f}%)")
                        print(f"     New EMA:          {ema_blank_prior:.6f} ({ema_blank_prior*100:.2f}%)")
                        print(f"     Change:           {abs(ema_blank_prior - old_blank_prior):.6f} ({(ema_blank_prior - old_blank_prior)/old_blank_prior*100:+.2f}%)")
                        print(f"     Momentum:         {momentum} (keeps {momentum*100:.0f}% old + {(1-momentum)*100:.0f}% new)")
                        
                        # Warn if current epoch differs significantly from EMA
                        diff_ratio = abs(current_blank_prior - ema_blank_prior) / ema_blank_prior
                        if diff_ratio > 0.3:  # More than 30% difference
                            print(f"   ⚠️  Warning: Current epoch blank prior differs {diff_ratio*100:.1f}% from EMA")
                            print(f"       This suggests training instability or data issue")
                    
                    # Use EMA priors (not current epoch priors) for loss computation
                    self.hparams.ctc_cost.log_priors = self.hparams.ctc_cost.ema_log_priors
                    
                    # Reset accumulators for next epoch
                    self.hparams.ctc_cost.log_priors_sum = None
                    self.hparams.ctc_cost.num_samples = 0
                    
                    # Save priors to checkpoint directory (optional but recommended)
                    try:
                        priors_dir = os.path.join(self.hparams.save_folder, "label_priors")
                        os.makedirs(priors_dir, exist_ok=True)
                        
                        # Save EMA priors (used for training)
                        ema_priors_path = os.path.join(priors_dir, f"log_priors_ema_epoch_{epoch}.pt")
                        torch.save(self.hparams.ctc_cost.ema_log_priors, ema_priors_path)
                        print(f"💾 Saved EMA priors to: {ema_priors_path}")
                        
                        # Also save current epoch priors (for comparison)
                        current_priors_path = os.path.join(priors_dir, f"log_priors_current_epoch_{epoch}.pt")
                        torch.save(new_log_priors, current_priors_path)
                        print(f"💾 Saved current epoch priors to: {current_priors_path}")
                        
                        # Save priors as JSON dictionary for easy inspection
                        import json
                        
                        # Prepare EMA priors dict
                        ema_priors_prob = torch.exp(self.hparams.ctc_cost.ema_log_priors).squeeze()
                        token_names = self._load_token_names()
                        ema_priors_dict = {
                            token_names.get(idx, f"token_{idx}"): float(prior) 
                            for idx, prior in enumerate(ema_priors_prob.tolist())
                        }
                        ema_log_priors_dict = {
                            token_names.get(idx, f"token_{idx}"): float(log_prior) 
                            for idx, log_prior in enumerate(self.hparams.ctc_cost.ema_log_priors.squeeze().tolist())
                        }
                        
                        json_path = os.path.join(priors_dir, f"priors_epoch_{epoch}.json")
                        json_data = {
                            "epoch": epoch,
                            "num_samples": int(self.hparams.ctc_cost.num_samples) if hasattr(self.hparams.ctc_cost, 'num_samples') else 0,
                            "prior_scaling_factor": float(self.hparams.ctc_cost.prior_scaling_factor),
                            "prior_momentum": float(self.hparams.ctc_cost.prior_momentum),
                            "current_epoch_priors": priors_dict,
                            "current_epoch_log_priors": log_priors_dict,
                            "ema_priors": ema_priors_dict,
                            "ema_log_priors": ema_log_priors_dict
                        }
                        with open(json_path, 'w') as f:
                            json.dump(json_data, f, indent=2)
                        print(f"💾 Saved priors JSON to: {json_path}")
                        
                        # Plot priors distribution
                        try:
                            import matplotlib.pyplot as plt
                            import matplotlib
                            matplotlib.use('Agg')  # Non-interactive backend
                            
                            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
                            
                            # Sort tokens by prior value for better visualization
                            sorted_items = sorted(priors_dict.items(), key=lambda x: x[1], reverse=True)
                            tokens_sorted = [item[0] for item in sorted_items]
                            priors_sorted = [item[1] for item in sorted_items]
                            
                            # Plot 1: Bar chart of all priors (sorted)
                            # Identify blank token (usually highest prior, or explicitly named)
                            blank_token_names = ['<blank>', 'blank', 'starting_index', 'token_0']
                            colors = ['red' if any(bn in name.lower() for bn in blank_token_names) or idx == 0
                                     else 'steelblue' 
                                     for idx, name in enumerate(tokens_sorted)]
                            bars = ax1.bar(range(len(tokens_sorted)), priors_sorted, color=colors)
                            ax1.set_xlabel('Token Index (sorted by prior)', fontsize=12)
                            ax1.set_ylabel('Prior (Probability)', fontsize=12)
                            ax1.set_title(f'Label Priors Distribution - Epoch {epoch}', fontsize=14, fontweight='bold')
                            ax1.grid(True, alpha=0.3, axis='y')
                            
                            # Highlight blank token
                            blank_idx_sorted = tokens_sorted.index('<blank>') if '<blank>' in tokens_sorted else (
                                tokens_sorted.index('blank') if 'blank' in tokens_sorted else 0
                            )
                            ax1.axhline(y=0.4, color='red', linestyle='--', alpha=0.5, linewidth=1, label='Target ~0.4')
                            ax1.axhline(y=0.3, color='red', linestyle='--', alpha=0.5, linewidth=1)
                            ax1.legend()
                            
                            # Add value labels for top 10
                            for i in range(min(10, len(priors_sorted))):
                                ax1.text(i, priors_sorted[i], f'{priors_sorted[i]:.3f}', 
                                        ha='center', va='bottom', fontsize=8)
                            
                            # Plot 2: Top 20 tokens with names
                            top_n = min(20, len(tokens_sorted))
                            tokens_top = tokens_sorted[:top_n]
                            priors_top = priors_sorted[:top_n]
                            colors_top = ['red' if any(bn in name.lower() for bn in blank_token_names) 
                                         else 'steelblue' 
                                         for name in tokens_top]
                            
                            bars2 = ax2.barh(range(top_n), priors_top, color=colors_top)
                            ax2.set_yticks(range(top_n))
                            ax2.set_yticklabels(tokens_top, fontsize=10)
                            ax2.set_xlabel('Prior (Probability)', fontsize=12)
                            ax2.set_ylabel('Token', fontsize=12)
                            ax2.set_title(f'Top {top_n} Tokens by Prior', fontsize=14, fontweight='bold')
                            ax2.grid(True, alpha=0.3, axis='x')
                            ax2.invert_yaxis()  # Highest at top
                            
                            # Add value labels
                            for i, (token, prior) in enumerate(zip(tokens_top, priors_top)):
                                ax2.text(prior, i, f' {prior:.4f}', 
                                        va='center', fontsize=9)
                            
                            plt.tight_layout()
                            plot_path = os.path.join(priors_dir, f"priors_distribution_epoch_{epoch}.png")
                            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                            plt.close(fig)
                            print(f"� Saved priors distribution plot to: {plot_path}")
                        except Exception as plot_e:
                            print(f"⚠️  Failed to plot priors distribution: {plot_e}")
                            
                    except Exception as e:
                        print(f"⚠️  Failed to save priors: {e}\n")
                        
        else:
            per = self.per_metrics.summarize("error_rate")
            mpd_f1 = self.mpd_metrics.summarize("mpd_f1")

        if stage == sb.Stage.VALID:
            # Log stats
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
            # Save best Models (only keep the single best for each metric)
            improved = False
            # Save best PER model (lower is better)
            if per < self.best_per:
                ckpt_name = f"best_per_{epoch:03d}_{per:.4f}.ckpt"
                self.checkpointer.save_and_keep_only(
                    meta={"PER": per, "mpd_f1": mpd_f1, "epoch": epoch},
                    name=ckpt_name,
                    num_to_keep=self.hparams.max_save_models,
                    min_keys=["PER"]
                )
                self.best_per = per
                improved = True
            
            # Save best mpd_f1 model (higher is better)
            if mpd_f1 > self.best_mpd_f1:
                ckpt_name = f"best_mpdf1_{epoch:03d}_{mpd_f1:.4f}.ckpt"
                self.checkpointer.save_and_keep_only(
                    meta={"PER": per, "mpd_f1": mpd_f1, "epoch": epoch},
                    name=ckpt_name,
                    num_to_keep=self.hparams.max_save_models,
                    max_keys=["mpd_f1"]
                )
                self.best_mpd_f1 = mpd_f1
                improved = True

            # Early stopping logic: only track best valid loss, do not save checkpoint for valid loss
            if stage_loss < self.best_valid_loss or len(self.best_valid_loss_list) < 10:
                ckpt_name = f"best_valid_loss_{epoch:03d}_{stage_loss:.4f}.ckpt"
                # Do NOT save checkpoint for valid loss (just update stats)
                self.best_valid_loss_list.append((stage_loss, epoch, ckpt_name))
                self.best_valid_loss_list = sorted(self.best_valid_loss_list, key=lambda x: x[0])[:10]
                self.best_valid_loss = self.best_valid_loss_list[0][0]
                improved = True

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
            with open(self.hparams.per_file, "w") as w:
                w.write("CTC loss stats:\n")
                self.ctc_metrics.write_stats(w)
                w.write("\nPER stats:\n")
                self.per_metrics.write_stats(w)
                print(
                    "CTC and PER stats written to file",
                    self.hparams.per_file,
                )
            with open(self.hparams.mpd_file, "w") as m:
                m.write("MPD results and stats:\n")
                self.mpd_metrics.write_stats(m)
                print(
                    "MPD results and stats written to file",
                    self.hparams.mpd_file,
                )
            
            # Save aggregated test decoding results
            if hasattr(self, '_test_predictions') and len(self._test_predictions) > 0:
                test_decoding_dir = os.path.join(self.hparams.output_folder, "test_decoding")
                os.makedirs(test_decoding_dir, exist_ok=True)
                
                # Save aggregated results as JSON
                test_summary_path = os.path.join(test_decoding_dir, "test_decoding_summary.json")
                test_summary = {
                    "total_samples": len(self._test_predictions),
                    "predictions": self._test_predictions
                }
                
                import json
                with open(test_summary_path, 'w', encoding='utf-8') as f:
                    json.dump(test_summary, f, ensure_ascii=False, indent=2)
                
                print(f"💾 Test decoding results saved to: {test_decoding_dir}")
                print(f"   Total samples: {len(self._test_predictions)}")
                print(f"   Summary file: {test_summary_path}")

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
                if any(p.requires_grad for p in self.pretrained_opt_class.param_groups[0]['params']):
                    self.scaler.step(self.pretrained_opt_class)
                if any(p.requires_grad for p in self.adam_optimizer.param_groups[0]['params']):
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
            # import pdb; pdb.set_trace()
            self.checkpointer.add_recoverable("tokenizer", self.label_encoder)  
    
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
        
        # Partial load pretrained components if specified
        
        if getattr(self.hparams, 'load_pretrained_components', False):
            pretrained_path = getattr(self.hparams, 'pretrained_model_path', '')
            components = getattr(self.hparams, 'components_to_load', ['ssl', 'enc', "ctc_head"])
            freeze_loaded = getattr(self.hparams, 'freeze_loaded_components', True)
            
            if pretrained_path and os.path.exists(pretrained_path):
                try:
                    self.load_pretrained_components(
                        checkpoint_path=pretrained_path,
                        components_to_load=components,
                        freeze_loaded=freeze_loaded
                    )
                except Exception as e:
                    print(f"❌ Failed to load pretrained components: {e}")
                    print("   Continuing with random initialization...")
            else:
                print(f"⚠️  Pretrained model path not found: {pretrained_path}")
                print("   Continuing with random initialization...")
        # Load latest checkpoint to resume training if interrupted
        ## NOTE: make sure to use the "best" model to continual training
        ## so we set the `min_key` argument
        elif self.checkpointer is not None:
            # TODO: support recover best on PER or mpd_f1 or averaged model of best PER and mpd_f1
            self.checkpointer.recover_if_possible(
                min_key="PER",
                # max_key="mpd_f1",
            )
        
    def load_pretrained_components(self, checkpoint_path, components_to_load=None, freeze_loaded=True):
        """
        Load specific components from a pretrained model checkpoint
        
        Args:
            checkpoint_path (str): Path to the checkpoint directory or file
            components_to_load (list): List of components to load. 
                                    Options: ['ssl', 'encoder', 'ctc_head', 'decoder', 'enc_projection']
                                    If None, loads ['ssl', 'encoder'] by default
            freeze_loaded (bool): Whether to freeze the loaded components
        """
        if components_to_load is None:
            components_to_load = ['ssl']  # Default: load SSL 
        
        print(f"\n🔄 Loading pretrained components from: {checkpoint_path}")
        print(f"   Components to load: {components_to_load}")
        # pdb.set_trace()
                
        from speechbrain.utils.parameter_transfer import Pretrainer

        pretrainer = Pretrainer(
            collect_in=self.hparams.pretrained_model_path,      # 把文件收集到这个目录（用软链或拷贝）
            loadables={
                "perceived_ssl":     self.modules.perceived_ssl,
                "model":     self.hparams.model,
            },
            paths={
                # 只写文件名，后面用 default_source 指定“仓库/目录”
                "perceived_ssl":     "perceived_ssl.ckpt",
                "model":   "model.ckpt",
            },
        )
        # import pdb; pdb.set_trace()
        paths = pretrainer.collect_files(default_source=self.hparams.pretrained_model_path)
        # before = self.modules.perceived_ssl.state_dict()["model.encoder.layers.23.final_layer_norm.weight"]
        pretrainer.load_collected()

        # Freeze loaded components if requested
        if freeze_loaded:
            for component in components_to_load:
                if component == 'ssl':
                    for param in self.modules.perceived_ssl.parameters():
                        param.requires_grad = False
                    self.ssl_frozen = True
                    print("   🔒 SSL model frozen")
                    
                elif component == 'encoder':
                    for param in self.modules.TransASR.encoder.parameters():
                        param.requires_grad = False
                    if hasattr(self.modules.TransASR, 'custom_src_module'):
                        for param in self.modules.TransASR.custom_src_module.parameters():
                            param.requires_grad = False
                    self.encoder_frozen = True
                    print("   🔒 Encoder frozen")
                    
                elif component == 'enc':
                    if hasattr(self.modules, 'enc'):
                        for param in self.modules.enc.parameters():
                            param.requires_grad = False
                        print("   🔒 Encoder projection frozen")
                        
                elif component == 'ctc_head':
                    for param in self.modules.ctc_lin.parameters():
                        param.requires_grad = False
                    print("   🔒 CTC head frozen")
    
        # print(f"   ✅ Successfully loaded components: {loaded_components}")
        # return loaded_components

class PhnMonoSSLModel_misproBCE(PhnMonoSSLModel):
    def __init__(self, *args,):
        super().__init__(*args)
        self_best_mispro_ce = float('inf')
        # if self.modules.BCEloss is not None:
        #     self.modules.BCEloss.to(self.device)
            
    def compute_objectives(self, predictions, batch, stage):
        "Given the network predictions and targets computed the NLL loss."
        loss = 0
        p_ctc, wav_lens = predictions
        ids = batch.id
        targets, target_lens = batch.phn_encoded_target
        mis_pro_labels, mis_pro_lens = batch.mispro_label
        canonicals, canonical_lens = batch.phn_encoded_canonical
        perceiveds, perceived_lens = batch.phn_encoded_perceived
        if self.hparams.training_target == "target":
            targets = targets
            target_lens = target_lens
        elif self.hparams.training_target == "canonical":
            targets = canonicals
            target_lens = canonical_lens
        elif self.hparams.training_target == "perceived":
            targets = perceiveds
            target_lens = perceived_lens        

        sequence = sb.decoders.ctc_greedy_decode(
            p_ctc, wav_lens, blank_id=self.hparams.blank_index
        )
        
        loss_ctc = self.hparams.ctc_cost(p_ctc, targets, wav_lens, target_lens)
        loss_mirpro_BCElike_ctc = self.hparams.ctc_cost_mispro(
            p_ctc, mis_pro_labels, wav_lens, mis_pro_lens
        )
        alpha = 0.8
        loss += (alpha * loss_ctc + (1-alpha)*loss_mirpro_BCElike_ctc)
        # loss += loss_ctc
        
        # Compute BCE loss for mispronunciation detection
        
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Canonical Phoneme Embedding, LSTM + Linear, LSTM out as key, Linear out as value
        self.modules.CanonicalPhonemeEmbedding = torch.nn.Embedding(
            num_embeddings= 42,
            embedding_dim=384  # Use the same dimension as the encoder output
        ).to(self.device)

        self.modules.CanonicalPhonemeLSTM = torch.nn.LSTM(
            input_size=384,
            hidden_size=384 // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        ).to(self.device)

        self.modules.CanonicalPhonemeLinear = torch.nn.Linear(
            in_features=384,
            out_features=384
        ).to(self.device)
        
        # Cross-attention mechanism
        # # Assuming x is the query and canonical_out is the key-value pair
        self.modules.cross_attention = sb.nnet.attention.MultiheadAttention(
            nhead=4,
            d_model=384,
            dropout=0.0
        )

        self.modules.attn_proj = sb.nnet.linear.Linear(
            input_size=384,  # Assuming the output dimension of cross-attention
            n_neurons=42
        )
        self.modules.out_sequence = sb.nnet.linear.Linear(
            input_size=42 *2,  # Concatenation of CTC and attention outputs
            n_neurons=42
        )
        
        # wrap all the added modules in a ModuleList
        self.modules.canoPhnEmb_Hybrid_CTC_Attention = torch.nn.ModuleList([
            self.modules.CanonicalPhonemeEmbedding,
            self.modules.CanonicalPhonemeLSTM,
            self.modules.CanonicalPhonemeLinear,
            self.modules.cross_attention,
            self.modules.attn_proj,
            self.modules.out_sequence
        ]).to(self.device)
        
    def compute_forward(self, batch, stage):
        "Given an input batch it computes the phoneme probabilities."
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        canonicals, canonical_lens = batch.phn_encoded_canonical

        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs)

        feats = self.modules.perceived_ssl(wavs)
        x = self.modules.enc(feats)

        # output layer for ctc log-probabilities
        logits = self.modules.ctc_lin(x)  # [B, T, 42]
        p_ctc = self.hparams.log_softmax(logits)  # [B, T, 42]

        # Get canonical phoneme embeddings
        canonical_embeds = self.modules.CanonicalPhonemeEmbedding(canonicals)  # [B, T, D]
        # Pass through LSTM
        canonical_lstm_out, _ = self.modules.CanonicalPhonemeLSTM(canonical_embeds)  # [B, T, D]
        # Apply linear transformation
        canonical_out = self.modules.CanonicalPhonemeLinear(canonical_lstm_out)  # [B, T, D]

        # Cross-attention mechanism
        # Q: Acoustic x
        # K: Canonical LSTM output
        # V: Canonical Linear output
        # relpostion embedding
        pos_emb = torch.randn(1,
                              x.size(1)*2-1,
                              x.size(2)).to(self.device)
        attn_output, attn_map = self.modules.cross_attention(
            query=x,
            key=canonical_lstm_out,
            value=canonical_out,
        )  # [B, T, D]
        attn_output_logits = self.modules.attn_proj(attn_output)  # [B, T, 42]
        p_attn = self.hparams.log_softmax(attn_output_logits)  # [B, T, 42]

        # Concatenate CTC and attention outputs (both [B, T, 42]) along last dim
        concat_hidden = torch.cat((logits, attn_output_logits), dim=-1)  # [B, T, 768]
        out_logits = self.modules.out_sequence(concat_hidden)  # [B, T, 42]
        p_out = self.hparams.log_softmax(out_logits)  # [B, T, 42]

        return p_out, p_ctc, p_attn, wav_lens, attn_map
        
    def compute_objectives(self, predictions, batch, stage):
        "Given the network predictions and targets computed the NLL loss."

        p_out, p_ctc, p_attn, wav_lens, attn_map = predictions

        ids = batch.id
        targets, target_lens = batch.phn_encoded_target
        canonicals, canonical_lens = batch.phn_encoded_canonical 
        perceiveds, perceived_lens = batch.phn_encoded_perceived 
        # len(perceiveds) == len(canonicals) != len(targets) 
        
        # if stage != sb.Stage.TRAIN:
        #     canonicals, canonical_lens = batch.phn_encoded_canonical
        # loss_ctc = self.hparams.ctc_cost(p_ctc, targets, wav_lens, target_lens)
        loss_ctc_ssl_ctc = self.hparams.ctc_cost(p_ctc, targets, wav_lens, target_lens)
        loss_ctc_attn = self.hparams.ctc_cost(p_attn, targets, wav_lens, target_lens)
        loss_ctc_out = self.hparams.ctc_cost(p_out, targets, wav_lens, target_lens)
        
        # Tobe fix
        loss = loss_ctc_attn
        # Log both CTC losses to wandb

        # Record losses for posterity
        if stage != sb.Stage.TRAIN:

            # Note: sb.decoders.ctc_greedy_decode will also remove padded tokens
            # that is, it return a list of list with different lengths
            sequence_ssl_ctc  = sb.decoders.ctc_greedy_decode(
                p_ctc, wav_lens, blank_id=self.hparams.blank_index
            )
            sequence_attn = sb.decoders.ctc_greedy_decode(
                p_attn, wav_lens, blank_id=self.hparams.blank_index
            )
            sequence_out = sb.decoders.ctc_greedy_decode(
                p_out, wav_lens, blank_id=self.hparams.blank_index
            )
            # self.ctc_metrics.append(ids, p_ctc, targets, wav_lens, target_lens)
            # dump first attention map to file
            import matplotlib.pyplot as plt
            # output_dir = self.hparams.output_dir
            # create attn dir
            attn_dir = os.path.join(self.hparams.output_folder, "attention_maps")
            if not os.path.exists(attn_dir):
                os.makedirs(attn_dir, exist_ok=True)
            # get current epoch
            epoch = self.hparams.epoch_counter.current
            if epoch %5 == 0 or epoch==1:
                for attn_id, attn in enumerate(attn_map[0: 1]):
                    plt.figure(figsize=(5, 5))
                    # apply log
                    plt.imshow(torch.log(attn).cpu().detach().numpy(), aspect='auto', origin='lower')
                    
                    plt.title(f"Attention Map for ID {ids[attn_id]}")
                    plt.xlabel("Canonical Phoneme Index")
                    plt.ylabel("Acoustic Feature Index")
                    plt.tight_layout()
                    attn_file = os.path.join(attn_dir, f"{ids[attn_id].split('/')[-1]}_epoch{epoch}.png")
                    plt.savefig(attn_file)
                    plt.close()
                    print(f"Saved attention map to {attn_file}")
                
            self.ctc_metrics.append(ids, p_out, targets, wav_lens, target_lens)

            self.per_metrics.append(
                ids=ids,
                predict=sequence_out,
                target=targets,
                predict_len=None,
                target_len=target_lens,
                ind2lab=self.label_encoder.decode_ndim,
            )
            
            self.mpd_metrics.append(
                ids=ids,
                predict=sequence_out,
                canonical=canonicals,
                perceived=perceiveds,
                predict_len=None,
                canonical_len=canonical_lens,
                perceived_len=perceived_lens,
                ind2lab=self.label_encoder.decode_ndim,
            )

        return loss

    def init_optimizers(self):
        # merge two models parameters
        modules_to_be_trained = torch.nn.ModuleList([
            self.hparams.model,
            self.modules.canoPhnEmb_Hybrid_CTC_Attention
        ])
        
        self.adam_optimizer = self.hparams.adam_opt_class(
            modules_to_be_trained.parameters()
        )
        self.pretrained_opt_class = self.hparams.pretrained_opt_class(
            self.modules.perceived_ssl.parameters(), 
        )

        if self.checkpointer is not None:
            # if self.hparams.perceived_ssl is not None and not self.hparams.perceived_ssl.freeze:
            self.checkpointer.add_recoverable("adam_opt", self.adam_optimizer)
            self.checkpointer.add_recoverable("pretrained_opt", self.pretrained_opt_class)
            self.checkpointer.add_recoverable("canoPhnEmb_Hybrid_CTC_Attention", self.modules.canoPhnEmb_Hybrid_CTC_Attention)

class PhnMonoSSLModel_withcanoPhnEmb_HMA_CTC(PhnMonoSSLModel):
    """PhnMonoSSLModel_withcanoPhnEmb_Hybrid_CTC_Attention_Ver2
    "https://arxiv.org/abs/2110.07274"
    Args:
        [Attn, SSL] -> Linear -> CTC 
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Canonical Phoneme Embedding, LSTM + Linear, LSTM out as key, Linear out as value
        # if modules has these given module below, skip defining them
        
        if self.modules.CanonicalPhonemeEmbedding:
            self.modules.CanonicalPhonemeEmbedding.to(self.device)
        if self.modules.CanonicalPhonemeLSTM:
            self.modules.CanonicalPhonemeLSTM.to(self.device)
        if self.modules.CanonicalPhonemeLinear:
            self.modules.CanonicalPhonemeLinear.to(self.device)

        if self.modules.cross_attention is not None:
            self.modules.cross_attention.to(self.device)
        if self.modules.attn_proj is not None:
            self.modules.attn_proj.to(self.device)
        
        if self.modules.cnn is not None:
            self.modules.cnn.to(self.device)
        if self.modules.out_sequence is not None:
            self.modules.out_sequence.to(self.device)


    def compute_forward(self, batch, stage):
        "Given an input batch it computes the phoneme probabilities."
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        
        canonicals, canonical_lens = batch.phn_encoded_canonical # [B, T_c]
        perceiveds, perceived_lens = batch.phn_encoded_perceived # [B, T_p]

        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs)
        feats = self.modules.perceived_ssl(wavs) # [B, T, ENC_DIM]
        x = self.modules.enc(feats) # [B, T, d_model]

        # output layer for ctc log-probabilities
        logits = self.modules.ctc_lin(x) # [B, T, D]
        p_ctc = self.hparams.log_softmax(logits) # [B, T, D]

        # Get canonical phoneme embeddings
        canonical_embeds = self.modules.CanonicalPhonemeEmbedding(canonicals) # [B, T_c, d_model]
        # Pass through LSTM
        canonical_lstm_out, _ = self.modules.CanonicalPhonemeLSTM(canonical_embeds) # [B, T_c, d_model]
        # Apply linear transformation
        canonical_out = self.modules.CanonicalPhonemeLinear(canonical_lstm_out) # [B, T_c, d_model]
        # pdb.set_trace()
        # ---- Build key padding mask for canonical sequence (True = pad) ----
        
        # SpeechBrain length tensors are ratios in [0,1]; convert to absolute token lengths
        max_S = canonical_lstm_out.size(1)
        canon_token_lens = (canonical_lens.to(self.device).float() * max_S).round().clamp(max=max_S).long()  # [B]
        key_padding_mask = torch.arange(max_S, device=self.device).unsqueeze(0) >= canon_token_lens.unsqueeze(1)  # [B, S], True = pad

        # ---- Query padding mask (True = pad) ----
        T = feats.size(1)
        q_tok_lens = (wav_lens.to(self.device).float() * T).round().clamp(max=T).long()  # [B]
        query_pad_mask = torch.arange(T, device=self.device).unsqueeze(0) >= q_tok_lens.unsqueeze(1)  # [B, T]

        # # ---- Build query from SSL feats (pre-CTC encoder) ----
        # q_raw = feats  # [B, T, ENC_DIM], keep richer temporal structure
        # q_proj = self.modules.q_proj(q_raw)  # [B, T, d_model]

        # # zero-out padded query positions
        # q_masked = q_proj.masked_fill(query_pad_mask.unsqueeze(-1), 0.0)
        # 零掉 query 的 pad 位置
        x_masked = x.masked_fill(query_pad_mask.unsqueeze(-1), 0.0)

        # ---- Cross-attention with padding mask ----
        attn_output, attn_map = self.modules.cross_attention(
            query=x_masked,
            key=canonical_lstm_out,
            value=canonical_out,
            attn_mask=None,
            key_padding_mask=key_padding_mask,
        )  # [B, T, d_model]
        # Concatenate CTC and attention outputs
        concat_hidden = torch.cat((feats, attn_output), dim=-1)
        concat_hidden = self.modules.cnn(concat_hidden.transpose(1, 2)).transpose(1, 2)
        # 
        out_logits = self.modules.out_sequence(concat_hidden)
        
        p_ctc = self.hparams.log_softmax(out_logits)
        # attn_output --> TransformerDecoder, 
        p_attn = self.hparams.log_softmax(self.modules.attn_proj(attn_output))
        
        return p_ctc, p_attn, wav_lens, attn_map
    
    def compute_objectives(self, predictions, batch, stage):
        "Given the network predictions and targets computed the NLL loss."

        p_ctc, p_attn, wav_lens, attn_map = predictions

        ids = batch.id
        targets, target_lens = batch.phn_encoded_target
        canonicals, canonical_lens = batch.phn_encoded_canonical 
        perceiveds, perceived_lens = batch.phn_encoded_perceived 
        # len(perceiveds) == len(canonicals) != len(targets) 

        # main CTC loss
        loss_ctc = self.hparams.ctc_cost(p_ctc, targets, wav_lens, target_lens)
        # attention CTC loss for canonical phoneme
        loss_attn_ctc = self.hparams.ctc_cost(p_attn, targets, wav_lens, target_lens)
        
        if hasattr(self.hparams, "loss_lambda"):
            lam = self.hparams.loss_lambda
        else:
            lam = 0.5  # default value if not specified
        loss = lam * loss_ctc + (1 - lam) * loss_attn_ctc
        # loss = loss_attn_ctc  # use attention CTC loss as main loss

        # Record losses for posterity
        if stage != sb.Stage.TRAIN:

            # Note: sb.decoders.ctc_greedy_decode will also remove padded tokens
            # that is, it return a list of list with different lengths
            sequence = sb.decoders.ctc_greedy_decode(
                p_ctc, wav_lens, blank_id=self.hparams.blank_index
            )
            sequence_attn = sb.decoders.ctc_greedy_decode(
                p_attn, wav_lens, blank_id=self.hparams.blank_index
            )
            # self.ctc_metrics.append(ids, p_ctc, targets, wav_lens, target_lens)
            # dump first attention map to file
            import matplotlib.pyplot as plt
            # output_dir = self.hparams.output_dir
            # create attn dir
            attn_dir = os.path.join(self.hparams.output_folder, "attention_maps")
            if not os.path.exists(attn_dir):
                os.makedirs(attn_dir, exist_ok=True)
            # get current epoch
            epoch = self.hparams.epoch_counter.current
            if epoch %10 == 0 or epoch==1:
                for attn_id, attn in enumerate(attn_map[0: 1]):
                    plt.figure(figsize=(5, 5))
                    # apply log
                    plt.imshow(attn.cpu().detach().numpy(), aspect='auto', origin='lower')
                    plt.title(f"Attention Map for ID {ids[attn_id]}")
                    plt.xlabel("Canonical Phoneme Index")
                    plt.ylabel("Acoustic Feature Index")
                    plt.tight_layout()
                    attn_file = os.path.join(attn_dir, f"{ids[attn_id].split('/')[-1]}_epoch{epoch}.png")
                    plt.savefig(attn_file)
                    plt.close()
                    print(f"Saved attention map to {attn_file}")
                
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

        # Log to wandb if available (VALID stage only)
        if stage == sb.Stage.VALID:
            try:
                import wandb
                wandb.log({
                    "loss_ctc_head": loss_ctc.item(),
                    "loss_ctc_attention": loss_attn_ctc.item(),
                }, step=self.hparams.epoch_counter.current)
            except Exception:
                pass

        return loss

    def init_optimizers(self):
        # merge two models parameters
        modules_to_be_trained = torch.nn.ModuleList([
            self.hparams.model,
        ])
        
        self.adam_optimizer = self.hparams.adam_opt_class(
            modules_to_be_trained.parameters()
        )
        self.pretrained_opt_class = self.hparams.pretrained_opt_class(
            self.modules.perceived_ssl.parameters(), 
        )
        if self.checkpointer is not None:
            # if self.hparams.perceived_ssl is not None and not self.hparams.perceived_ssl.freeze:
            self.checkpointer.add_recoverable("adam_opt", self.adam_optimizer)
            self.checkpointer.add_recoverable("pretrained_opt", self.pretrained_opt_class)

class HMA_attn_ctc_to_canonical(PhnMonoSSLModel_withcanoPhnEmb_HMA_CTC):
    def compute_objectives(self, predictions, batch, stage):
        return super().compute_objectives(predictions, batch, stage)

class HMA_attn_ctc_to_mispro(PhnMonoSSLModel_withcanoPhnEmb_HMA_CTC):
    def ctc_segmentation_align(self, p_ctc, targets, wav_len, target_len, blank_id=0, char_list=None, frame_duration=1):
        """
        使用ctc_segmentation库对齐声学特征和标签。
        参数：
            p_ctc: [B, T, C]，log概率（或概率）
            targets: [B, L]，标签索引
            wav_len: [B]，每个batch的音频长度占maxlen的比例
            target_len: [B]，每个batch的目标长度和maxlen的比例
            blank_id: CTC blank的索引
            char_list: 标签id到字符的映射（如['a','b',...,'<blank>']）
            frame_duration: 每帧对应的秒数
        返回：
            alignments: list，每个batch元素是[(start_frame, end_frame, label), ...]
        """
        import numpy as np
        from ctc_segmentation import ctc_segmentation, CtcSegmentationParameters, prepare_text, determine_utterance_segments, prepare_tokenized_text, prepare_token_list
        alignments = []
        timings_array = []
        char_probs_array = []
        state_list_array = []
        B = p_ctc.shape[0]
        for b in range(B):
            # 1. 获取概率（如果是log概率则exp）
            # get valid target length and wav length
            wav_len_b = int(wav_len[b] * p_ctc.shape[1])  if isinstance(wav_len[b], torch.Tensor) else wav_len[b]
            target_len_b = int(target_len[b] * targets.shape[1]) if isinstance(target_len[b], torch.Tensor) else target_len[b]
            # 计算实际帧数
            probs = p_ctc[b][:wav_len_b].detach().cpu().numpy()

            if np.max(probs) <= 0:
                probs = np.exp(probs)
            if probs.ndim == 1:
                probs = probs[None, :]  # 保证至少2维
            # 2. 标签转字符
            if char_list is None:
                # 默认用id字符串
                char_list = [str(i) for i in range(probs.shape[1])]
            # 3. 目标序列转字符
            target_ids = targets[b].detach().cpu().numpy().tolist()
            # 去除padding/blank
            target_ids = [i for i in target_ids if i != blank_id]
            ground_truth = ' '.join([char_list[i] for i in target_ids])
            # 4. 配置参数
            config = CtcSegmentationParameters()
            config.blank = blank_id
            config.index_duration = frame_duration
            config.char_list = char_list
            # 5. 预处理文本
            ground_truth_list = [ground_truth]
            char_list_proc, utt_begin_indices = prepare_tokenized_text(config, ground_truth_list)
            # 6. 运行ctc_segmentation
            # import pdb; pdb.set_trace()
            timings, char_probs, state_list = ctc_segmentation(config, probs, char_list_proc)
            # 7. 解析对齐区间
            # import pdb; pdb.set_trace()
            segments = determine_utterance_segments(config, utt_begin_indices, probs, timings, ground_truth_list)

            # segments: list of dicts with 'label', 'start', 'end' (帧)
            alignments.append(segments)
            timings_array.append(timings)
            char_probs_array.append(char_probs)
            state_list_array.append(state_list) 
        # return as BatchTensor
        timings_tensor_list = [torch.tensor(t, dtype=torch.float16) for t in timings_array]
        char_probs_tensor_list = [torch.tensor(c, dtype=torch.float16) for c in char_probs_array]
        #state_list_tensor_list = [torch.tensor(s, dtype=torch.float16) for s in state_list_array]
        
        from torch.nn.utils.rnn import pad_sequence

        timings_padded = pad_sequence(timings_tensor_list, batch_first=True, padding_value=0.0)      # [B, max_T]
        char_probs_padded = pad_sequence(char_probs_tensor_list, batch_first=True, padding_value=0.0) # [B, max_T, C]
        # state_list_padded = pad_sequence(state_list_tensor_list, batch_first=True, padding_value=0.0) # [B, max_T]

        batch_dict = {
            "timings": timings_padded,
            "char_probs": char_probs_padded,
            "state_list": state_list_array,
        }
        return alignments, batch_dict
        # import pdb;pdb.set_trace()
        
        # import pdb;pdb.set_trace()

        # return alignments, timings_array
    """PhnMonoSSLModel_withcanoPhnEmb_Hybrid_CTC_Attention_Ver2
    "https://arxiv.org/abs/2110.07274"
    Args:
        [Attn, SSL] -> Linear -> CTC 
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Canonical Phoneme Embedding, LSTM + Linear, LSTM out as key, Linear out as value
        # if modules has these given module below, skip defining them

        if self.modules.mispro_head is not None:
            self.modules.mispro_head.to(self.device)

        if self.modules.lab_sequence_lin is not None:
            self.modules.lab_sequence_lin.to(self.device)

    def compute_forward(self, batch, stage):
        "Given an input batch it computes the phoneme probabilities."
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        
        canonicals, canonical_lens = batch.phn_encoded_canonical # [B, T_c]
        perceiveds, perceived_lens = batch.phn_encoded_perceived # [B, T_p]

        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs)
        feats = self.modules.perceived_ssl(wavs) # [B, T, ENC_DIM]
        x = self.modules.enc(feats) # [B, T, d_model]

        # output layer for ctc log-probabilities
        logits = self.modules.ctc_lin(x) # [B, T, D]
        p_ctc = self.hparams.log_softmax(logits) # [B, T, D]

        # Get canonical phoneme embeddings
        canonical_embeds = self.modules.CanonicalPhonemeEmbedding(canonicals) # [B, T_c, d_model]
        # Pass through LSTM
        canonical_lstm_out, _ = self.modules.CanonicalPhonemeLSTM(canonical_embeds) # [B, T_c, d_model]
        # Apply linear transformation
        canonical_out = self.modules.CanonicalPhonemeLinear(canonical_lstm_out) # [B, T_c, d_model]
        # pdb.set_trace()
        # ---- Build key padding mask for canonical sequence (True = pad) ----
        # SpeechBrain length tensors are ratios in [0,1]; convert to absolute token lengths
        max_S = canonical_lstm_out.size(1)
        canon_token_lens = (canonical_lens.to(self.device).float() * max_S).round().clamp(max=max_S).long()  # [B]
        key_padding_mask = torch.arange(max_S, device=self.device).unsqueeze(0) >= canon_token_lens.unsqueeze(1)  # [B, S], True = pad

        # ---- Query padding mask (True = pad) ----
        T = feats.size(1)
        q_tok_lens = (wav_lens.to(self.device).float() * T).round().clamp(max=T).long()  # [B]
        query_pad_mask = torch.arange(T, device=self.device).unsqueeze(0) >= q_tok_lens.unsqueeze(1)  # [B, T]

        # # ---- Build query from SSL feats (pre-CTC encoder) ----
        # q_raw = feats  # [B, T, ENC_DIM], keep richer temporal structure
        # q_proj = self.modules.q_proj(q_raw)  # [B, T, d_model]

        # # zero-out padded query positions
        # q_masked = q_proj.masked_fill(query_pad_mask.unsqueeze(-1), 0.0)
        # 零掉 query 的 pad 位置
        x_masked = x.masked_fill(query_pad_mask.unsqueeze(-1), 0.0)

        # ---- Cross-attention with padding mask ----
        attn_output, attn_map = self.modules.cross_attention(
            query=x_masked,
            key=canonical_lstm_out,
            value=canonical_out,
            attn_mask=None,
            key_padding_mask=key_padding_mask,
        )  # [B, T, d_model]
        # Concatenate CTC and attention outputs
        concat_hidden = torch.cat((feats, attn_output), dim=-1)
        concat_hidden = self.modules.cnn(concat_hidden.transpose(1, 2)).transpose(1, 2)
        # 
        out_logits = self.modules.out_sequence(concat_hidden)
        
        p_ctc_attn = self.hparams.log_softmax(out_logits)
        # mispronunciation detection head
        out_mispro = self.modules.mispro_head(concat_hidden)  # [B, T, D]

        # labeling sequence
        lab_sequence_logits = self.modules.lab_sequence_lin(concat_hidden)  # [B, T, D]
        
        p_label_seq = self.hparams.log_softmax(lab_sequence_logits)
        # attn_output --> TransformerDecoder, 
        p_attn = self.hparams.log_softmax(self.modules.attn_proj(attn_output))
        
        return p_ctc, p_attn, p_ctc_attn, p_label_seq, out_mispro, wav_lens, attn_map
    
    def compute_objectives(self, predictions, batch, stage):
        "Given the network predictions and targets computed the NLL loss."

        p_ctc, p_attn, p_ctc_attn, p_label_seq, out_mispro, wav_lens, attn_map = predictions

        ids = batch.id
        targets, target_lens = batch.phn_encoded_target
        canonicals, canonical_lens = batch.phn_encoded_canonical 
        perceiveds, perceived_lens = batch.phn_encoded_perceived 

        mispronunciation_labels, _ = batch.mispro_label_framewise
        phn_seq_labels, phn_seq_lens = batch.phn_encoded_target_bin
        
        # Downsample frame-wise mispronunciation labels to match out_mispro time dimension
        B, T_out, _ = out_mispro.shape
        # Assume labels shape is [B, T_mis]
        # Convert to float and add channel dimension
        labels_unsq = mispronunciation_labels.float().unsqueeze(1)  # [B, 1, T_mis]
        # Adaptive average pooling to T_out
        pool = torch.nn.AdaptiveAvgPool1d(T_out)
        labels_ds = pool(labels_unsq).squeeze(1)  # [B, T_out]
        # Binarize: any positive in the original window yields 1
        mispronunciation_labels = (labels_ds >= 0.5).float()
        # Adjust lengths to new resolution
        orig_len = mispronunciation_labels.shape[1]
        # Compute new lengths proportionally
        # main CTC loss
        # Downsample the bin-wise phoneme sequence labels to match p_label_seq time dimension
        import torch.nn.functional as F
        import matplotlib.pyplot as plt
        # plt.figure(figsize=(10, 10))        
        # plt.imshow(phn_seq_labels.cpu().detach().numpy(), aspect='auto', origin='lower', interpolation='nearest')
        # plt.colorbar()
        # plt.savefig("phn_seq_labels.png")
        def downsample_pool_argmax(labels: torch.LongTensor, T_out: int, C: int):
            """
            labels: Tensor[B, L]
            T_out: 目标时间步长度
            C:    类别数（如 40 多）
            返回 new_labels: Tensor[B, T_out]
            """
            B, L = labels.shape
            # 1) 变成 one-hot：[B, L] -> [B, L, C] -> [B, C, L]
            onehot = F.one_hot(labels, num_classes=C).permute(0, 2, 1).float()
            # 2) 自适应平均池化到长度 T_out -> [B, C, T_out]
            pooled = F.adaptive_avg_pool1d(onehot, output_size=T_out)
            # 3) 通道上取最大 -> [B, T_out]
            new_labels = pooled.argmax(dim=1)
            return new_labels
        phn_seq_labels = downsample_pool_argmax(phn_seq_labels, T_out=out_mispro.shape[1], C=self.hparams.output_neurons)
        # plt.imshow(phn_seq_labels.cpu().detach().numpy(), aspect='auto', origin='lower')
        # plt.savefig("phn_seq_labels_ds.png")
        # plt.xlabel("Time")
        # import pdb; pdb.set_trace()

        loss_ctc = self.hparams.ctc_cost(p_ctc, targets, wav_lens, target_lens)
        
        # CTC loss for canonical phonemes prediction out of attention layers
        loss_ctc_attn = self.hparams.ctc_cost(p_attn, canonicals, wav_lens, canonical_lens)
        
        # CTC loss for perceived phonemes prediction
        loss_ctc_fuse = self.hparams.ctc_cost(p_ctc_attn, targets, wav_lens, target_lens)
        
        # sequence mispronunciation loss
        loss_mispro = sb.nnet.losses.bce_loss(
            inputs=out_mispro,
            targets=mispronunciation_labels,
            length=wav_lens,
        )

        # sequence labeling loss
        CE_loss = torch.nn.CrossEntropyLoss(ignore_index=self.label_encoder.lab2ind["<blank>"])
        p_label_seq = p_label_seq.view(-1, self.hparams.output_neurons)  # [B*T, D]
        phn_seq_labels = phn_seq_labels.view(-1)  # [B*T]

        loss_label_seq = CE_loss(
            input=p_label_seq,
            target=phn_seq_labels,
        )
        
        # CTC segmentation alignment to do later
        # char_list = [self.label_encoder.ind2lab[i] for i in range(len(self.label_encoder.ind2lab))]
        # _, timings = self.ctc_segmentation_align(p_ctc, targets, wav_len=wav_lens, target_len=target_lens, blank_id=self.hparams.blank_index, char_list=char_list)
        # predict_phn_timings, predict_phn_probs, _ = timings["timings"], timings["char_probs"], timings["state_list"]
        
        # import pdb; pdb.set_trace()


        loss = loss_ctc + 5 * loss_mispro + loss_label_seq

        # Record losses for posterity
        if stage != sb.Stage.TRAIN:
            # Note: sb.decoders.ctc_greedy_decode will also remove padded tokens
            # that is, it return a list of list with different lengths

            sequence = sb.decoders.ctc_greedy_decode(
                p_ctc, wav_lens, blank_id=self.hparams.blank_index
            )
            import matplotlib.pyplot as plt
            attn_dir = os.path.join(self.hparams.output_folder, "attention_maps")
            if not os.path.exists(attn_dir):
                os.makedirs(attn_dir, exist_ok=True)
            epoch = self.hparams.epoch_counter.current
            if epoch %10 == 0 or epoch==1:
                for attn_id, attn in enumerate(attn_map[0: 1]):
                    plt.figure(figsize=(5, 5))
                    plt.imshow(attn.cpu().detach().numpy(), aspect='auto', origin='lower')
                    plt.title(f"Attention Map for ID {ids[attn_id]}")
                    plt.xlabel("Canonical Phoneme Index")
                    plt.ylabel("Acoustic Feature Index")
                    plt.tight_layout()
                    attn_file = os.path.join(attn_dir, f"{ids[attn_id].split('/')[-1]}_epoch{epoch}.png")
                    plt.savefig(attn_file)
                    plt.close()
                    print(f"Saved attention map to {attn_file}")
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

        # Log to wandb if available (VALID stage only)
        if stage == sb.Stage.VALID:
            try:
                import wandb
                wandb.log({
                    "loss_ctc_head": loss_ctc.item(),
                    "loss_ctc_attn": loss_ctc_attn.item(),
                    "loss_ctc_fuse": loss_ctc_fuse.item(),
                    "loss_mispro": loss_mispro.item(),
                    "loss_label_seq": loss_label_seq.item(),
                }, step=self.hparams.epoch_counter.current)
            except Exception:
                pass

        return loss
    
    class HMA_attn_ctc_to_canonical(PhnMonoSSLModel_withcanoPhnEmb_HMA_CTC):
        def compute_objectives(self, predictions, batch, stage):
            return super().compute_objectives(predictions, batch, stage)
        
        def compute_objectives(self, predictions, batch, stage):
            "Given the network predictions and targets computed the NLL loss."

            p_ctc, p_attn, wav_lens, attn_map = predictions

            ids = batch.id
            targets, target_lens = batch.phn_encoded_target
            canonicals, canonical_lens = batch.phn_encoded_canonical 
            perceiveds, perceived_lens = batch.phn_encoded_perceived 
            mispro_labels, mispro_lens = batch.mispro_label # canonical phoneme length
            phn_starts, phn_ends = batch.phone_starts
            phn_ends, phn_ends_lens = batch.phn_ends
            
            # generate a batch form mispronunciation labels on p_ctc's size
            mispro_labels_framewise = torch.zeros_like(p_ctc, dtype=torch.float)
            for i, (start, end) in enumerate(zip(phn_starts, phn_ends)):
                # fill the mispronunciation labels with 1s in the range of start to end
                mispro_labels_framewise[i, start:end] = 1.0
            # get frame-level mispronunciation labels
            # import pdb; pdb.set_trace()
            # main CTC loss
            loss_ctc = self.hparams.ctc_cost(p_ctc, targets, wav_lens, target_lens)
            # attention CTC loss for canonical phoneme
            loss_attn_ctc = self.hparams.ctc_cost(p_attn, canonicals, wav_lens, canonicals)
        
            if hasattr(self.hparams, "loss_lambda"):
                lam = self.hparams.loss_lambda
            else:
                lam = 0.5  # default value if not specified
            loss = lam * loss_ctc + (1 - lam) * loss_attn_ctc
            # loss = loss_attn_ctc  # use attention CTC loss as main loss

            # Record losses for posterity
            if stage != sb.Stage.TRAIN:

                # Note: sb.decoders.ctc_greedy_decode will also remove padded tokens
                # that is, it return a list of list with different lengths
                sequence = sb.decoders.ctc_greedy_decode(
                    p_ctc, wav_lens, blank_id=self.hparams.blank_index
                )
                # self.ctc_metrics.append(ids, p_ctc, targets, wav_lens, target_lens)
                # dump first attention map to file
                import matplotlib.pyplot as plt
                # output_dir = self.hparams.output_dir
                # create attn dir
                attn_dir = os.path.join(self.hparams.output_folder, "attention_maps")
                if not os.path.exists(attn_dir):
                    os.makedirs(attn_dir, exist_ok=True)
                # get current epoch
                epoch = self.hparams.epoch_counter.current
                if epoch %10 == 0 or epoch==1:
                    for attn_id, attn in enumerate(attn_map[0: 1]):
                        plt.figure(figsize=(5, 5))
                        # apply log
                        plt.imshow(attn.cpu().detach().numpy(), aspect='auto', origin='lower')
                        plt.title(f"Attention Map for ID {ids[attn_id]}")
                        plt.xlabel("Canonical Phoneme Index")
                        plt.ylabel("Acoustic Feature Index")
                        plt.tight_layout()
                        attn_file = os.path.join(attn_dir, f"{ids[attn_id].split('/')[-1]}_epoch{epoch}.png")
                        plt.savefig(attn_file)
                        plt.close()
                        print(f"Saved attention map to {attn_file}")
                    
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

            # Log to wandb if available (VALID stage only)
            if stage == sb.Stage.VALID:
                try:
                    import wandb
                    wandb.log({
                        "loss_ctc_head": loss_ctc.item(),
                        "loss_ctc_attention": loss_attn_ctc.item(),
                    }, step=self.hparams.epoch_counter.current)
                except Exception:
                    pass

            return loss
          
class PhnMonoSSLModel_withcanoPhnEmb_MHA_Guided_Attention_CTC(PhnMonoSSLModel_withcanoPhnEmb_HMA_CTC):
    """PhnMonoSSLModel_withcanoPhnEmb_Hybrid_CTC_Attention_Ver2
    "https://arxiv.org/abs/2110.07274"
    Args:
        [Attn, SSL] -> Linear -> CTC 
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.guided_attn_loss = GuidedAttentionLoss(
           sigma=getattr(self.hparams, "ga_sigma", 0.2)
        )
        
    def compute_objectives(self, predictions, batch, stage):
        "Given the network predictions and targets computed the NLL loss."

        p_ctc, p_attn, wav_lens, attn_map = predictions

        ids = batch.id
        targets, target_lens = batch.phn_encoded_target
        canonicals, canonical_lens = batch.phn_encoded_canonical 
        perceiveds, perceived_lens = batch.phn_encoded_perceived 
        # len(perceiveds) == len(canonicals) != len(targets) 

        # main CTC losses
        loss_ctc = self.hparams.ctc_cost(p_ctc, targets, wav_lens, target_lens)
        loss_attn_ctc = self.hparams.ctc_cost(p_attn, targets, wav_lens, target_lens)

        # ---- Guided Attention Loss ----
        # attn_map: [B,H,T_q,T_k] or [B,T_q,T_k]
        if attn_map.dim() == 4:
            A = attn_map.mean(dim=1)          # average heads -> [B,T_q,T_k]
        elif attn_map.dim() == 3:
            A = attn_map
        else:
            raise ValueError(f"Unexpected attn_map shape {attn_map.shape}")

        # GuidedAttentionLoss expects (B, targets, inputs).
        # 我们把“targets”当作 canonical tokens (K)， “inputs”当作 acoustic frames (Q)。
        A_ga = A.permute(0, 2, 1)             # [B, T_k, T_q]

        Bsz, T_q_max, T_k_max = A.shape
        # 语音帧长度
        if wav_lens.dtype.is_floating_point:
            in_lens_abs = (wav_lens * T_q_max).round().long().clamp(1, T_q_max)
        else:
            in_lens_abs = wav_lens.long().clamp(1, T_q_max)
        # canonical token 长度
        if canonical_lens.dtype.is_floating_point:
            tgt_lens_abs = (canonical_lens * T_k_max).round().long().clamp(1, T_k_max)
        else:
            tgt_lens_abs = canonical_lens.long().clamp(1, T_k_max)

        loss_ga = self.guided_attn_loss(
            A_ga, input_lengths=in_lens_abs, target_lengths=tgt_lens_abs
        )

        lam_main = getattr(self.hparams, "loss_lambda", 0.5)
        lam_ga   = getattr(self.hparams, "ga_lambda", 0.1)

        loss = lam_main * loss_ctc + (1 - lam_main) * loss_attn_ctc + lam_ga * loss_ga
        # import pdb; pdb.set_trace()
        # Record losses for posterity
        if stage != sb.Stage.TRAIN:

            # Note: sb.decoders.ctc_greedy_decode will also remove padded tokens
            # that is, it return a list of list with different lengths
            sequence = sb.decoders.ctc_greedy_decode(
                p_ctc, wav_lens, blank_id=self.hparams.blank_index
            )
            # self.ctc_metrics.append(ids, p_ctc, targets, wav_lens, target_lens)
            # dump first attention map to file
            import matplotlib.pyplot as plt
            # output_dir = self.hparams.output_dir
            # create attn dir
            attn_dir = os.path.join(self.hparams.output_folder, "attention_maps")
            if not os.path.exists(attn_dir):
                os.makedirs(attn_dir, exist_ok=True)
            # get current epoch
            epoch = self.hparams.epoch_counter.current
            if epoch %5 == 0 or epoch==1:
                for attn_id, attn in enumerate(attn_map[0: 1]):
                    plt.figure(figsize=(5, 5))
                    # apply log
                    plt.imshow(attn.cpu().detach().numpy(), aspect='auto', origin='lower')
                    plt.title(f"Attention Map for ID {ids[attn_id]}")
                    plt.xlabel("Canonical Phoneme Index")
                    plt.ylabel("Acoustic Feature Index")
                    plt.tight_layout()
                    attn_file = os.path.join(attn_dir, f"{ids[attn_id].split('/')[-1]}_epoch{epoch}.png")
                    plt.savefig(attn_file)
                    plt.close()
                    print(f"Saved attention map to {attn_file}")
                
            self.ctc_metrics.append(ids, p_ctc, targets, wav_lens, target_lens)
            # remove 41 from sequence if element == 41
            
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

        # Log to wandb if available (VALID stage only)
        if stage == sb.Stage.VALID:
            try:
                import wandb
                wandb.log({
                    "loss_ctc_head": loss_ctc.item(),
                    "loss_ctc_attention": loss_attn_ctc.item(),
                    "loss_ga": loss_ga.item(),
                }, step=self.hparams.epoch_counter.current)
            except Exception:
                pass

        return loss
    
    def init_optimizers(self):
        # merge two models parameters
        modules_to_be_trained = torch.nn.ModuleList([
            self.hparams.model,
        ])
        
        self.adam_optimizer = self.hparams.adam_opt_class(
            modules_to_be_trained.parameters()
        )
        self.pretrained_opt_class = self.hparams.pretrained_opt_class(
            self.modules.perceived_ssl.parameters(), 
        )
        if self.checkpointer is not None:
            # if self.hparams.perceived_ssl is not None and not self.hparams.perceived_ssl.freeze:
            self.checkpointer.add_recoverable("adam_opt", self.adam_optimizer)
            self.checkpointer.add_recoverable("pretrained_opt", self.pretrained_opt_class)

class PhnMonoSSLModel_RVQforCano(PhnMonoSSLModel):
    def compute_forward(self, batch, stage):
        "Given an input batch it computes the phoneme probabilities."
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        # phns_bos, _ = batch.phn_encoded_bos

        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs)

        # some wav2vec models (e.g. large-lv60) needs attention_mask
        # if self.modules.perceived_ssl.feature_extractor.return_attention_mask:
        #     attn_mask = make_attn_mask(wavs, wav_lens)
        # else:
        #     attn_mask = None

        feats = self.modules.perceived_ssl(wavs)
        if feats.dim() == 4:
            feats = feats[self.hparams.preceived_ssl_emb_layer]

        x = self.modules.enc(feats)
        if getattr(self.modules, "ConformerEncoder", None) is not None:
            from speechbrain.nnet.attention import RelPosEncXL, RelPosMHAXL, RoPEMHA 
            # import pdb; pdb.set_trace()
            if self.modules.ConformerEncoder.attention_type == "RelPosMHAXL":
                pos_emb = RelPosEncXL(emb_dim=self.hparams.dnn_neurons)(x).to(self.device)
                x, _ = self.modules.ConformerEncoder(x, pos_embs=pos_emb)
            else:
                x, _ = self.modules.ConformerEncoder(x)
            
        # Get RVQ if exists
        if getattr(self.modules, "RVQ", None) is not None:
            # Expect [B, C, T]
            x = x.transpose(1, 2)  # [B, T, C] -> [B, C, T]
            discrete_embeddings, codes, latents, commitment_loss, codebook_loss = self.modules.RVQ(x)
            import pdb; pdb.set_trace()
            # Use continuous embeddings for perceived phoneme CTC, while discrete codes for canonical phoneme CTC
            perc_x = x.transpose(1, 2)  # [B, C, T] -> [B, T, C] 
            cano_x = discrete_embeddings.transpose(1, 2)  # [B, C, T] -> [B, T, C]
        # else:
        # output layer for ctc log-probabilities
        logits = self.modules.ctc_lin(perc_x)
        p_ctc = self.hparams.log_softmax(logits) # (B, T, C)
        # Canonical 
        cano_logits = self.modules.ctc_cano_lin(cano_x)
        p_cano_ctc = self.hparams.log_softmax(cano_logits) # (
        
        if self.modules.RVQ is not None:
            return p_ctc, p_cano_ctc, wav_lens, commitment_loss, codebook_loss
        return p_ctc, wav_lens

    def compute_objectives(self, predictions, batch, stage):
        "Given the network predictions and targets computed the NLL loss."
        if self.modules.RVQ is not None:
            p_ctc, p_cano_ctc, wav_lens, commitment_loss, codebook_loss = predictions
        else:
            p_ctc, wav_lens = predictions
        ids = batch.id
        targets, target_lens = batch.phn_encoded_target
        
        canonicals, canonical_lens = batch.phn_encoded_canonical
        perceiveds, perceived_lens = batch.phn_encoded_perceived

        loss_ctc = self.hparams.ctc_cost(p_ctc, targets, wav_lens, target_lens)
        loss_cano_ctc = self.hparams.ctc_cost(p_cano_ctc, canonicals, wav_lens, canonical_lens) 
        
        if self.modules.RVQ is not None:
            loss = loss_ctc + loss_cano_ctc + (commitment_loss + codebook_loss)
        else:
            loss = loss_ctc

        # Record losses for posterity
        if stage != sb.Stage.TRAIN:
            # Note: sb.decoders.ctc_greedy_decode will also remove padded tokens
            # that is, it return a list of list with different lengths
            sequence = sb.decoders.ctc_greedy_decode(
                p_ctc, wav_lens, blank_id=self.hparams.blank_index
            )
            sequence_cano = sb.decoders.ctc_greedy_decode(
                p_cano_ctc, wav_lens, blank_id=self.hparams.blank_index
            )
            
            self.ctc_metrics.append(ids, p_ctc, targets, wav_lens, target_lens)
            self.ctc_cano_metrics.append(ids, p_cano_ctc, canonicals, wav_lens, canonical_lens)
            
            self.per_metrics.append(
                ids=ids,
                predict=sequence,
                target=targets,
                predict_len=None,
                target_len=target_lens,
                ind2lab=self.label_encoder.decode_ndim,
            )
            
            self.per_cano_metrics.append(
                ids=ids,
                predict=sequence_cano,
                target=canonicals,
                predict_len=None,
                target_len=canonical_lens,
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
    
    def on_stage_start(self, stage, epoch):
        "Gets called when a stage (either training, validation, test) starts."
        self.ctc_metrics = self.hparams.ctc_stats()
        self.ctc_cano_metrics = self.hparams.ctc_stats()
        if hasattr(self.hparams, "augmentation"):
            self.modules.perceived_ssl.model.config.apply_spec_augment = True

        if stage != sb.Stage.TRAIN:
  
            self.per_metrics = self.hparams.per_stats()
            self.per_cano_metrics = self.hparams.per_stats()
            self.mpd_metrics = MpdStats()
    
    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        else:
            per = self.per_metrics.summarize("error_rate")
            per_cano = self.per_cano_metrics.summarize("error_rate")        
            mpd_f1 = self.mpd_metrics.summarize("mpd_f1")

        if stage == sb.Stage.VALID:
            # Log stats
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
                    "ctc_cano_loss": self.ctc_cano_metrics.summarize("average"),
                    "PER": per,
                    "PER_cano": per_cano,
                    "mpd_f1": mpd_f1
                },
            )
            # Save best Models (only keep the single best for each metric)
            improved = False
            # Save best PER model (lower is better)
            # if per < self.best_per:
            #     ckpt_name = f"best_per_{epoch:03d}_{per:.4f}.ckpt"
            #     self.checkpointer.save_and_keep_only(
            #         meta={"PER": per, "mpd_f1": mpd_f1, "epoch": epoch},
            #         name=ckpt_name,
            #         num_to_keep=2,
            #         min_keys=["PER", "PER_cano"]
            #     )
            #     self.best_per = per
            #     improved = True
            # # Save best mpd_f1 model (higher is better)
            # if mpd_f1 > self.best_mpd_f1:
            #     ckpt_name = f"best_mpdf1_{epoch:03d}_{mpd_f1:.4f}.ckpt"
            #     self.checkpointer.save_and_keep_only(
            #         meta={"PER": per, "mpd_f1": mpd_f1, "epoch": epoch},
            #         name=ckpt_name,
            #         num_to_keep=2,
            #         max_keys=["mpd_f1"]
            #     )
            #     self.best_mpd_f1 = mpd_f1
            #     improved = True
            ckpt_name = f"{epoch:03d}_PER_{per:.4f}_PER_Cano_{per_cano:.4f}_F1_{mpd_f1:.4f}.ckpt"

            self.checkpointer.save_and_keep_only(
                meta={"PER": per, "PER_cano": per_cano, "mpd_f1": mpd_f1, "epoch": epoch},
                name=ckpt_name,
                num_to_keep=3,
                importance_keys=[
                    lambda ckpt: (-ckpt.meta["PER"], -ckpt.meta["PER_cano"], ckpt.meta["mpd_f1"]),  # lower PER, lower PER_cano, higher mpd_f1
                ]
            )

            # Early stopping logic: only track best valid loss, do not save checkpoint for valid loss
            if stage_loss < self.best_valid_loss or len(self.best_valid_loss_list) < 10:
                ckpt_name = f"best_valid_loss_{epoch:03d}_{stage_loss:.4f}.ckpt"
                # Do NOT save checkpoint for valid loss (just update stats)
                self.best_valid_loss_list.append((stage_loss, epoch, ckpt_name))
                self.best_valid_loss_list = sorted(self.best_valid_loss_list, key=lambda x: x[0])[:10]
                self.best_valid_loss = self.best_valid_loss_list[0][0]
                improved = True

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
                "ctc_cano_loss": self.ctc_cano_metrics.summarize("average"),
                "PER": per,
                "PER_cano": per_cano,
                "mpd_f1": mpd_f1,
            }, step=epoch)
            # Early stop if patience exceeded
            if self.no_improve_epochs >= self.patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {self.patience} epochs)")
                raise StopIteration

        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats={"loss": stage_loss, "PER": per, "mpd_f1": mpd_f1, "PER_cano": per_cano},
            )
            # 
            with open(self.hparams.per_file, "w") as w:
                w.write("CTC loss stats:\n")
                self.ctc_metrics.write_stats(w)
                w.write("\nPER stats:\n")
                self.per_metrics.write_stats(w)
                print(
                    "CTC and PER stats written to file",
                    self.hparams.per_file,
                )
            if getattr(self.hparams, "ctc_cano_file", None) is None:
                self.hparams.ctc_cano_file = self.hparams.per_file + ".cano.txt"
                with open(self.hparams.ctc_cano_file, "w") as w:
                    w.write("Canonical CTC loss stats:\n")
                    self.ctc_cano_metrics.write_stats(w)
                    w.write("\nCanonical PER stats:\n")
                    self.per_cano_metrics.write_stats(w)
                    print(
                        "Canonical CTC and PER stats written to file",
                        self.hparams.ctc_cano_file,
                    )
            with open(self.hparams.mpd_file, "w") as m:
                m.write("MPD results and stats:\n")
                self.mpd_metrics.write_stats(m)
                print(
                    "MPD results and stats written to file",
                    self.hparams.mpd_file,
                )
                
class PhnMonoSSLModel_DualCTCHead(PhnMonoSSLModel):
    """PhnMonoSSLModel with dual CTC heads for perceived and canonical phonemes.
        Perceived feature from middle layers, Canonical from Last layer.
        Ver 1: seperate enc after SSL, expecially for different SSL Layers for Canonical and Perceived features.
        Ver 2: share enc if Canonical and Perceived features are from the same layer, and allow shareenc=True
    """
    def compute_forward(self, batch, stage):
        "Given an input batch it computes the phoneme probabilities."
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig

        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs)

        feats = self.modules.perceived_ssl(wavs)
        assert feats.dim() == 4  # (B, L, T, D)
        feats_cano = feats[self.hparams.preceived_ssl_emb_layer]
        feats_perc = feats[self.hparams.canonical_ssl_emb_layer]
        # If Canonical Feature and Perceived Feature are from the same layer and allow share encoder, 
        if self.hparams.preceived_ssl_emb_layer == self.hparams.canonical_ssl_emb_layer and self.hparams.shareenc and getattr(self.modules, "enc_cano", None) == None:
            feats_perc = feats_cano
            x_perc = self.modules.enc(feats_perc)
            x_cano = x_perc
            
        else:
            # Use a separate encoder for Canonical feature
            try:
                x_cano = self.modules.enc_cano(feats_cano)
                x_perc = self.modules.enc(feats_perc)
            except:
                if self.hparams.preceived_ssl_emb_layer != self.hparams.canonical_ssl_emb_layer:
                    raise ValueError("Please define a separate encoder for Canonical feature as it is from a different SSL layer.")

        
        if getattr(self.modules, "ConformerEncoder", None) is not None:
            from speechbrain.nnet.attention import RelPosEncXL, RelPosMHAXL, RoPEMHA 
            pos_emb = RelPosEncXL(emb_dim=self.hparams.dnn_neurons)(x_perc).to(self.device)
            x_perc, _ = self.modules.ConformerEncoder(x_perc, pos_embs=pos_emb)

        # output layer for ctc log-probabilities
        logits = self.modules.ctc_lin(x_cano)
        p_ctc = self.hparams.log_softmax(logits) # (B, T, C)
        
        # Canonical 
        cano_logits = self.modules.ctc_cano_lin(x_cano)
        p_cano_ctc = self.hparams.log_softmax(cano_logits) # (B, T, C)
        
        return p_ctc, p_cano_ctc, wav_lens

    def compute_objectives(self, predictions, batch, stage):
        "Given the network predictions and targets computed the NLL loss."

        p_ctc, p_cano_ctc, wav_lens = predictions
        ids = batch.id
        targets, target_lens = batch.phn_encoded_target
        
        canonicals, canonical_lens = batch.phn_encoded_canonical
        perceiveds, perceived_lens = batch.phn_encoded_perceived

        loss_ctc = self.hparams.ctc_cost(p_ctc, targets, wav_lens, target_lens)
        loss_cano_ctc = self.hparams.ctc_cost(p_cano_ctc, canonicals, wav_lens, canonical_lens) 
        
        loss = loss_ctc + loss_cano_ctc

        # Record losses for posterity
        if stage != sb.Stage.TRAIN:
            # Note: sb.decoders.ctc_greedy_decode will also remove padded tokens
            # that is, it return a list of list with different lengths
            sequence = sb.decoders.ctc_greedy_decode(
                p_ctc, wav_lens, blank_id=self.hparams.blank_index
            )
            sequence_cano = sb.decoders.ctc_greedy_decode(
                p_cano_ctc, wav_lens, blank_id=self.hparams.blank_index
            )
            
            self.ctc_metrics.append(ids, p_ctc, targets, wav_lens, target_lens)
            self.ctc_cano_metrics.append(ids, p_cano_ctc, canonicals, wav_lens, canonical_lens)
            
            self.per_metrics.append(
                ids=ids,
                predict=sequence,
                target=targets,
                predict_len=None,
                target_len=target_lens,
                ind2lab=self.label_encoder.decode_ndim,
            )
            
            self.per_cano_metrics.append(
                ids=ids,
                predict=sequence_cano,
                target=canonicals,
                predict_len=None,
                target_len=canonical_lens,
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
    
    def on_stage_start(self, stage, epoch):
        "Gets called when a stage (either training, validation, test) starts."
        self.ctc_metrics = self.hparams.ctc_stats()
        self.ctc_cano_metrics = self.hparams.ctc_stats()
        if hasattr(self.hparams, "augmentation"):
            self.modules.perceived_ssl.model.config.apply_spec_augment = True

        if stage != sb.Stage.TRAIN:
  
            self.per_metrics = self.hparams.per_stats()
            self.per_cano_metrics = self.hparams.per_stats()
            self.mpd_metrics = MpdStats()
    
    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        else:
            per = self.per_metrics.summarize("error_rate")
            per_cano = self.per_cano_metrics.summarize("error_rate")        
            mpd_f1 = self.mpd_metrics.summarize("mpd_f1")

        if stage == sb.Stage.VALID:
            # Log stats
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
                    "ctc_cano_loss": self.ctc_cano_metrics.summarize("average"),
                    "PER": per,
                    "PER_cano": per_cano,
                    "mpd_f1": mpd_f1
                },
            )
            # Save best Models (only keep the single best for each metric)
            improved = False

            ckpt_name = f"{epoch:03d}_PER_{per:.4f}_PER_Cano_{per_cano:.4f}_F1_{mpd_f1:.4f}.ckpt"

            self.checkpointer.save_and_keep_only(
                meta={"PER": per, "PER_cano": per_cano, "mpd_f1": mpd_f1, "epoch": epoch},
                name=ckpt_name,
                num_to_keep=3,
                importance_keys=[
                    lambda ckpt: (-ckpt.meta["PER"], -ckpt.meta["PER_cano"], ckpt.meta["mpd_f1"]),  # lower PER, lower PER_cano, higher mpd_f1
                ]
            )

            # Early stopping logic: only track best valid loss, do not save checkpoint for valid loss
            if stage_loss < self.best_valid_loss or len(self.best_valid_loss_list) < 10:
                ckpt_name = f"best_valid_loss_{epoch:03d}_{stage_loss:.4f}.ckpt"
                # Do NOT save checkpoint for valid loss (just update stats)
                self.best_valid_loss_list.append((stage_loss, epoch, ckpt_name))
                self.best_valid_loss_list = sorted(self.best_valid_loss_list, key=lambda x: x[0])[:10]
                self.best_valid_loss = self.best_valid_loss_list[0][0]
                improved = True

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
                "ctc_cano_loss": self.ctc_cano_metrics.summarize("average"),
                "PER": per,
                "PER_cano": per_cano,
                "mpd_f1": mpd_f1,
            }, step=epoch)
            # Early stop if patience exceeded
            if self.no_improve_epochs >= self.patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {self.patience} epochs)")
                raise StopIteration

        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats={"loss": stage_loss, "PER": per, "mpd_f1": mpd_f1, "PER_cano": per_cano},
            )
            # 
            with open(self.hparams.per_file, "w") as w:
                w.write("CTC loss stats:\n")
                self.ctc_metrics.write_stats(w)
                w.write("\nPER stats:\n")
                self.per_metrics.write_stats(w)
                print(
                    "CTC and PER stats written to file",
                    self.hparams.per_file,
                )
            if getattr(self.hparams, "ctc_cano_file", None) is None:
                self.hparams.ctc_cano_file = self.hparams.per_file + ".cano.txt"
                with open(self.hparams.ctc_cano_file, "w") as w:
                    w.write("Canonical CTC loss stats:\n")
                    self.ctc_cano_metrics.write_stats(w)
                    w.write("\nCanonical PER stats:\n")
                    self.per_cano_metrics.write_stats(w)
                    print(
                        "Canonical CTC and PER stats written to file",
                        self.hparams.ctc_cano_file,
                    )
            with open(self.hparams.mpd_file, "w") as m:
                m.write("MPD results and stats:\n")
                self.mpd_metrics.write_stats(m)
                print(
                    "MPD results and stats written to file",
                    self.hparams.mpd_file,
                )
                
class PhnMonoSSLModel_RVQforBoth(PhnMonoSSLModel_DualCTCHead):
    def compute_forward(self, batch, stage):
        "Given an input batch it computes the phoneme probabilities."
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        # phns_bos, _ = batch.phn_encoded_bos

        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs)

        # some wav2vec models (e.g. large-lv60) needs attention_mask
        # if self.modules.perceived_ssl.feature_extractor.return_attention_mask:
        #     attn_mask = make_attn_mask(wavs, wav_lens)
        # else:
        #     attn_mask = None

        feats = self.modules.perceived_ssl(wavs)
        if feats.dim() == 4:
            feats = feats[self.hparams.preceived_ssl_emb_layer]

        x = self.modules.enc(feats)
        if getattr(self.modules, "ConformerEncoder", None) is not None:
            from speechbrain.nnet.attention import RelPosEncXL, RelPosMHAXL, RoPEMHA 
            pos_emb = RelPosEncXL(emb_dim=self.hparams.dnn_neurons)(x).to(self.device)
            # import pdb; pdb.set_trace()
            x, _ = self.modules.ConformerEncoder(x, pos_embs=pos_emb)
        # Get RVQ if exists
        if getattr(self.modules, "RVQ", None) is not None:
            # Expect [B, C, T]
            x = x.transpose(1, 2)  # [B, T, C] -> [B, C, T]
            discrete_embeddings, codes, latents, commitment_loss, codebook_loss = self.modules.RVQ(x)
            # Use discrete representations for both perceived and canonical phoneme CTC
            perc_x = discrete_embeddings.transpose(1, 2)  # [B, C, T] -> [B, T, C]
            cano_x = discrete_embeddings.transpose(1, 2)  # [B, C, T] -> [B, T, C]
            
        # else:
        # output layer for ctc log-probabilities
        logits = self.modules.ctc_lin(perc_x)
        p_ctc = self.hparams.log_softmax(logits) # (B, T, C)
        # Canonical 
        cano_logits = self.modules.ctc_cano_lin(cano_x)
        p_cano_ctc = self.hparams.log_softmax(cano_logits) # (
        
        if self.modules.RVQ is not None:
            return p_ctc, p_cano_ctc, wav_lens, commitment_loss, codebook_loss
        return p_ctc, wav_lens

    def compute_objectives(self, predictions, batch, stage):
        "Given the network predictions and targets computed the NLL loss."
        if self.modules.RVQ is not None:
            p_ctc, p_cano_ctc, wav_lens, commitment_loss, codebook_loss = predictions
        else:
            p_ctc, wav_lens = predictions
        ids = batch.id
        targets, target_lens = batch.phn_encoded_target
        
        canonicals, canonical_lens = batch.phn_encoded_canonical
        perceiveds, perceived_lens = batch.phn_encoded_perceived

        loss_ctc = self.hparams.ctc_cost(p_ctc, targets, wav_lens, target_lens)
        loss_cano_ctc = self.hparams.ctc_cost(p_cano_ctc, canonicals, wav_lens, canonical_lens) 
        
        if self.modules.RVQ is not None:
            loss = loss_ctc + loss_cano_ctc + (commitment_loss + codebook_loss)
        else:
            loss = loss_ctc

        # Record losses for posterity
        if stage != sb.Stage.TRAIN:
            # Note: sb.decoders.ctc_greedy_decode will also remove padded tokens
            # that is, it return a list of list with different lengths
            sequence = sb.decoders.ctc_greedy_decode(
                p_ctc, wav_lens, blank_id=self.hparams.blank_index
            )
            sequence_cano = sb.decoders.ctc_greedy_decode(
                p_cano_ctc, wav_lens, blank_id=self.hparams.blank_index
            )
            
            self.ctc_metrics.append(ids, p_ctc, targets, wav_lens, target_lens)
            self.ctc_cano_metrics.append(ids, p_cano_ctc, canonicals, wav_lens, canonical_lens)
            
            self.per_metrics.append(
                ids=ids,
                predict=sequence,
                target=targets,
                predict_len=None,
                target_len=target_lens,
                ind2lab=self.label_encoder.decode_ndim,
            )
            
            self.per_cano_metrics.append(
                ids=ids,
                predict=sequence_cano,
                target=canonicals,
                predict_len=None,
                target_len=canonical_lens,
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
    
class PhnMonoSSLModel_EMA(PhnMonoSSLModel):
    def __init__(self, *args, patience=20, **kwargs):
        super().__init__(*args, **kwargs)
        import sparc
        from sparc import load_model
        coder = load_model("en", device= "cuda:0")  # For using GPU
        # coder_from_config = load_model(config="/home/kevingenghaopeng/work/Speech-Articulatory-Coding/configs/feature_extraction.yaml")
        self.EMA_coder = coder
        
    def compute_forward(self, batch, stage):
        "Given an input batch it computes the phoneme probabilities."
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        # phns_bos, _ = batch.phn_encoded_bos

        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs)

        # some wav2vec models (e.g. large-lv60) needs attention_mask
        # if self.modules.perceived_ssl.feature_extractor.return_attention_mask:
        #     attn_mask = make_attn_mask(wavs, wav_lens)
        # else:
        #     attn_mask = None

        feats = self.modules.perceived_ssl(wavs)
        
        x = self.modules.enc(feats)
        # 
        # EMA_feats = self.EMA_coder.encode(batch.id, split_batch=False)
        # Duration might be different
        # ema_feats = EMA_feats["ema"]
        # import pdb; pdb.set_trace()
        from sparc import SpeechWave
        EMA_wavs = SpeechWave(wavs, (wav_lens*wavs.shape[-1]).int())
        # import pdb; pdb.set_trace()
        ema = self.EMA_coder.inverter(EMA_wavs, {})['ema']
        ema = torch.Tensor(ema).to(self.device)
        h_ema = self.modules.EMA_enc(ema)
        
        # Method 1 ADD and norm
        # x += h_ema
        # x = torch.nn.LayerNorm(x.shape[-1])(x)
        
        # Method 2 concat
        x = torch.concat([x, h_ema], dim=-1)
        # pdb.set_trace()

        if getattr(self.modules, "ConformerEncoder", None) is not None:
            from speechbrain.nnet.attention import RelPosEncXL, RelPosMHAXL, RoPEMHA 
            pos_emb = RelPosEncXL(emb_dim=self.hparams.dnn_neurons)(x).to(self.device)
            # import pdb; pdb.set_trace()
            x, _ = self.modules.ConformerEncoder(x, pos_embs=pos_emb)
        # Get RVQ if exists
        if getattr(self.modules, "RVQ", None) is not None:
            # Expect [B, C, T]
            x = x.transpose(1, 2)  # [B, T, C] -> [B, C, T]
            discrete_embeddings, codes, latents, commitment_loss, codebook_loss = self.modules.RVQ(x)
            x = discrete_embeddings.transpose(1, 2)  # [B, C, T] -> [B, T, C]
        # output layer for ctc log-probabilities
        logits = self.modules.ctc_lin(x)
        p_ctc = self.hparams.log_softmax(logits) # (B, T, C)
        if getattr(self.modules, "RVQ", None) is not None:
            return p_ctc, wav_lens, commitment_loss, codebook_loss
        return p_ctc, wav_lens

class PhnMonoEMAModel(PhnMonoSSLModel):
    def __init__(self, *args, patience=20, **kwargs):
        super().__init__(*args, **kwargs)
        import sparc
        from sparc import load_model
        coder = load_model("en", device= "cuda:0")  # For using GPU
        # coder_from_config = load_model(config="/home/kevingenghaopeng/work/Speech-Articulatory-Coding/configs/feature_extraction.yaml")
        self.EMA_coder = coder
        
    def compute_forward(self, batch, stage):
        "Given an input batch it computes the phoneme probabilities."
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        # phns_bos, _ = batch.phn_encoded_bos

        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs)

        from sparc import SpeechWave
        EMA_wavs = SpeechWave(wavs, (wav_lens*wavs.shape[-1]).int())
        # import pdb; pdb.set_trace()
        ema = self.EMA_coder.inverter(EMA_wavs, {})['ema']
        ema = torch.Tensor(ema).to(self.device)
        x = self.modules.EMA_enc(ema)
        
        # Method 1 ADD and norm
        # x += h_ema
        # x = torch.nn.LayerNorm(x.shape[-1])(x)
        
        # Method 2 concat
        # x = torch.concat([x, h_ema], dim=-1)

        if getattr(self.modules, "ConformerEncoder", None) is not None:
            from speechbrain.nnet.attention import RelPosEncXL, RelPosMHAXL, RoPEMHA 
            pos_emb = RelPosEncXL(emb_dim=self.hparams.dnn_neurons)(x).to(self.device)
            # import pdb; pdb.set_trace()
            x, _ = self.modules.ConformerEncoder(x, pos_embs=pos_emb)
        # Get RVQ if exists
        if getattr(self.modules, "RVQ", None) is not None:
            # Expect [B, C, T]
            x = x.transpose(1, 2)  # [B, T, C] -> [B, C, T]
            discrete_embeddings, codes, latents, commitment_loss, codebook_loss = self.modules.RVQ(x)
            x = discrete_embeddings.transpose(1, 2)  # [B, C, T] -> [B, T, C]
        # output layer for ctc log-probabilities
        logits = self.modules.ctc_lin(x)
        p_ctc = self.hparams.log_softmax(logits) # (B, T, C)
        if getattr(self.modules, "RVQ", None) is not None:
            return p_ctc, wav_lens, commitment_loss, codebook_loss
        return p_ctc, wav_lens
    
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
            self.adam_optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)

            # normalize the loss by gradient_accumulation and scale for mixed precision
            self.scaler.scale(loss / self.hparams.gradient_accumulation).backward()
            
            self.scaler.unscale_(self.adam_optimizer)

            if self.check_gradients(loss):
                if any(p.requires_grad for p in self.adam_optimizer.param_groups[0]['params']):
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
                    self.adam_optimizer.step()

                
                self.adam_optimizer.zero_grad()    

        return loss.detach().cpu()

    def init_optimizers(self):
        self.adam_optimizer = self.hparams.adam_opt_class(
            self.hparams.model.parameters(), 
        )

        if self.checkpointer is not None:
            # if self.hparams.perceived_ssl is not None and not self.hparams.perceived_ssl.freeze:
            self.checkpointer.add_recoverable("adam_opt", self.adam_optimizer)
            # self.checkpointer.add_recoverable("pretrained_opt", self.pretrained_opt_class)
            # import pdb; pdb.set_trace()
            self.checkpointer.add_recoverable("tokenizer", self.label_encoder)  
    def on_stage_start(self, stage, epoch):
        "Gets called when a stage (either training, validation, test) starts."
        self.ctc_metrics = self.hparams.ctc_stats()

        if stage != sb.Stage.TRAIN:
  
            self.per_metrics = self.hparams.per_stats()
            self.mpd_metrics = MpdStats()


class PhnMonoSSLModel_CRCTC(PhnMonoSSLModel):
    """
    PhnMonoSSLModel with Consistency-Regularized CTC (CR-CTC) loss.
    
    CR-CTC applies two different augmentations to the input and enforces
    consistency between their outputs via KL divergence loss.
    
    Reference: https://arxiv.org/abs/2310.11905
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # CR-CTC hyperparameters with defaults
        self.cr_loss_weight = getattr(self.hparams, "cr_loss_weight", 0.1)
        self.cr_loss_masked_scale = getattr(self.hparams, "cr_loss_masked_scale", 1.0)
        # CR loss tracking
        self.cr_loss_sum = 0.0
        self.cr_loss_count = 0
        self.ctc_loss_sum = 0.0
        self.ctc_loss_count = 0
    
    def _apply_time_warp_augmentation(self, wavs, wav_lens, time_warp_factor=1600, p=1.0, save_comparison=False):
        """
        Apply time warping augmentation to waveforms.
        
        Args:
            wavs: Input waveforms [B, T]
            wav_lens: Relative lengths of waveforms [B]
            time_warp_factor: Time warping factor (default: 1600)
            p: Probability of applying time warp (default: 1.0)
            save_comparison: Whether to save comparison plots and audio files (default: False)
            
        Returns:
            wavs_warped: Time-warped waveforms [B, T]
        """
        from icefall.utils import time_warp
        
        # Create supervision_segments to avoid applying time_warp on padding frames
        # Format: [sequence_idx, start_frame, num_frames]
        num_frames = (wav_lens * wavs.shape[-1]).to(self.device).int()
        supervision_segments = torch.stack([
            torch.arange(wavs.shape[0]).to(self.device), 
            torch.zeros(wavs.shape[0], dtype=torch.int32).to(self.device), 
            num_frames
        ], dim=1)
        
        # Apply time_warp
        wavs_warped = time_warp(
            features=wavs.unsqueeze(-1),
            p=p,
            time_warp_factor=time_warp_factor,
            supervision_segments=supervision_segments,
        ).squeeze(-1)
        
        # Optional: save comparison for debugging
        if save_comparison:
            diff = torch.abs(wavs - wavs_warped)
            print(f"Max difference between original and time-warped waveform: {diff.max().item()}")
            
            import matplotlib.pyplot as plt
            import torchaudio
            
            for i, (wav, wav_warped) in enumerate(zip(wavs, wavs_warped)):
                # Plot waveforms
                fig, axs = plt.subplots(2, 1, figsize=(10, 6))
                axs[0].plot(wavs[i].detach().cpu().numpy())
                axs[0].set_title('Original Waveform')
                axs[1].plot(wavs_warped[i].detach().cpu().numpy())
                axs[1].set_title('Time-Warped Waveform')
                plt.tight_layout()
                plt.savefig(f'crctc_time_warp_example_{i}.png')
                plt.close(fig)
                
                # Save audio files
                torchaudio.save(f'original_waveform_{i}.wav', wavs[i].unsqueeze(0).detach().cpu(), 16000)    
                torchaudio.save(f'time_warped_waveform_{i}.wav', wavs_warped[i].unsqueeze(0).detach().cpu(), 16000)
        
        return wavs_warped
        
    def compute_forward(self, batch, stage):
        "Given an input batch it computes the phoneme probabilities."
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig

        # Apply standard augmentation if in training
        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "speed_augmentation"):
                wavs = self.hparams.speed_augmentation(wavs)
            if hasattr(self.hparams, "augmentation"):
                # wavs = self.hparams.augmentation(wavs)
                wavs_1, wav_lens_1 = self.hparams.augmentation.forward(wavs, lengths=wav_lens)
                wavs_2, wav_lens_2 = self.hparams.augmentation.forward(wavs, lengths=wav_lens)
                
                # import matplotlib.pyplot as plt
                # fig, axs = plt.subplots(3, 1, figsize=(10, 8))
                # axs[0].specgram(wavs[0].detach().cpu().numpy(), Fs=16000)
                # axs[0].set_title('Original Features')
                # axs[1].specgram(wavs_1[0].detach().cpu().numpy(), Fs=16000)
                # axs[1].set_title('Augmented Features 1')
                # axs[2].specgram(wavs_2[0].detach().cpu().numpy(), Fs=16000)
                # axs[2].set_title('Augmented Features 2')
                # plt.tight_layout()
                # plt.savefig('crctc_waveform_augmentation_example.png')
                # plt.close(fig)
                # import pdb; pdb.set_trace()
                # assert torch.allclose(wav_lens, wav_lens_1)
                # assert torch.allclose(wav_lens, wav_lens_2)
                # assert wav_lens == wav_lens_1
        
        # Extract SSL features
        feats = self.modules.perceived_ssl(wavs)
        
        use_crctc = getattr(self.hparams, "use_crctc", True) and stage == sb.Stage.TRAIN
        
        if use_crctc:
            feats_1 = self.modules.perceived_ssl(wavs_1)
            feats_2 = self.modules.perceived_ssl(wavs_2)
            # CR-CTC: duplicate features and apply different augmentations
            # feats shape: [B, T, D]
            B, T, D = feats.shape
            
            # Get CR-CTC hyperparameters with defaults following icefall/k2 conventions
            # Regular SpecAugment: 2 freq masks (width 27), 10 time masks (width 100), 15% time masking
            # CR-CTC: 2.5x multiplier on time masking
            
            # freq mask ratio: 3/16
            # time mask ratio: 15%
            
            # num_freq_masks = getattr(self.hparams, "cr_num_freq_masks", 2)
            # max_freq_mask_width = getattr(self.hparams, "cr_max_freq_mask_width", int(3/16 * D))  # 3/16 of feature dim
            
            # num_time_masks = getattr(self.hparams, "cr_num_time_masks", 25)  # 10 * 2.5
            # max_time_mask_width = getattr(self.hparams, "cr_max_time_mask_width", 100)
            # max_time_mask_ratio = getattr(self.hparams, "cr_max_time_mask_ratio", 0.375)  # 15% * 2.5
            
            # Create two augmented versions with different random masks
            # Augmentation 1
            # import pdb; pdb.set_trace()
            # feats_1 = self._apply_specaugment(
            #     feats.clone(),
            #     num_freq_masks=num_freq_masks,
            #     max_freq_mask_width=max_freq_mask_width,
            #     num_time_masks=num_time_masks,
            #     max_time_mask_width=max_time_mask_width,
            #     max_time_mask_ratio=max_time_mask_ratio
            # )
            
            # Augmentation 2 (different random patterns)
            # feats_2 = self._apply_specaugment(
            #     feats.clone(),
            #     num_freq_masks=num_freq_masks,
            #     max_freq_mask_width=max_freq_mask_width,
            #     num_time_masks=num_time_masks,
            #     max_time_mask_width=max_time_mask_width,
            #     max_time_mask_ratio=max_time_mask_ratio
            # )
            
            # compare original and augmented, plot in 3*1 subplots
            # import matplotlib.pyplot as plt
            # fig, axs = plt.subplots(3, 1, figsize=(10, 8))
            # axs[0].imshow(feats[0].detach().cpu().numpy().T, aspect='auto', origin='lower')
            # axs[0].set_title('Original Features')
            # axs[1].imshow(feats_1[0].detach().cpu().numpy().T, aspect='auto', origin='lower')
            # axs[1].set_title('Augmented Features 1')
            # axs[2].imshow(feats_2[0].detach().cpu().numpy().T, aspect='auto', origin='lower')
            # axs[2].set_title('Augmented Features 2')
            # plt.tight_layout()
            # plt.savefig('crctc_feature_augmentation_example.png')
            # plt.close(fig)
            
            # pdb.set_trace()
            # Concatenate both versions: [2*B, T, D]
            feats_combined = torch.cat([feats_1, feats_2], dim=0)
            wav_lens_combined = wav_lens.repeat(2)
            
            # Encode combined features
            x = self.modules.enc(feats_combined)
            
            if getattr(self.modules, "ConformerEncoder", None) is not None:
                from speechbrain.nnet.attention import RelPosEncXL
                pos_emb = RelPosEncXL(emb_dim=self.hparams.dnn_neurons)(x).to(self.device)
                x, _ = self.modules.ConformerEncoder(x, pos_embs=pos_emb)
            
            # CTC output layer
            logits = self.modules.ctc_lin(x)
            p_ctc = self.hparams.log_softmax(logits)  # [2*B, T, C]
            
            return p_ctc, wav_lens_combined, None, True  # True indicates CR-CTC mode
            
        else:
            # Standard forward pass (no CR-CTC)
            x = self.modules.enc(feats)
            
            if getattr(self.modules, "ConformerEncoder", None) is not None:
                from speechbrain.nnet.attention import RelPosEncXL
                pos_emb = RelPosEncXL(emb_dim=self.hparams.dnn_neurons)(x).to(self.device)
                x, _ = self.modules.ConformerEncoder(x, pos_embs=pos_emb)
            
            # CTC output layer
            logits = self.modules.ctc_lin(x)
            p_ctc = self.hparams.log_softmax(logits)  # [B, T, C]
            
            return p_ctc, wav_lens, None, False  # False indicates standard mode
    
    def _apply_specaugment(self, feats, num_freq_masks=2, max_freq_mask_width=27, 
                          num_time_masks=25, max_time_mask_width=100, 
                          max_time_mask_ratio=0.375):
        """
        Apply SpecAugment with both time and frequency masking.
        
        Reference: Park et al., 2019. "SpecAugment: A Simple Data Augmentation Method 
        for Automatic Speech Recognition"
        
        For CR-CTC, we use 2.5x augmentation on time masking:
        - Regular: 10 time masks, 15% masking fraction
        - CR-CTC: 25 time masks (10*2.5), 37.5% masking fraction (15%*2.5)
        
        Args:
            feats: Input features [B, T, D]
            num_freq_masks: Number of frequency masking regions
            max_freq_mask_width: Maximum width for frequency masking
            num_time_masks: Number of time masking regions
            max_time_mask_width: Maximum width for time masking
            max_time_mask_ratio: Maximum ratio of total time to mask
            
        Returns:
            augmented_feats: Masked features [B, T, D]
        """
        B, T, D = feats.shape
        augmented_feats = feats.clone()
        
        # ===== Frequency Masking (along feature dimension D) =====
        for b in range(B):
            for _ in range(num_freq_masks):
                # Randomly select masking region width
                mask_width = torch.randint(1, max_freq_mask_width + 1, (1,)).item()
                # Randomly select starting position
                mask_start = torch.randint(0, max(1, D - mask_width + 1), (1,)).item()
                mask_end = min(mask_start + mask_width, D)
                
                # Apply frequency mask (set to 0)
                augmented_feats[b, :, mask_start:mask_end] = 0.0
        
        # ===== Time Masking (along time dimension T) =====
        # Calculate maximum number of frames to mask based on ratio
        max_frames_to_mask = int(T * max_time_mask_ratio)
        total_frames_masked = 0
        
        for b in range(B):
            num_masks_applied = 0
            attempts = 0
            max_attempts = num_time_masks * 3  # Allow extra attempts in case of overlaps
            
            while num_masks_applied < num_time_masks and attempts < max_attempts:
                attempts += 1
                
                # Randomly select masking region width
                mask_width = torch.randint(1, max_time_mask_width + 1, (1,)).item()
                
                # Check if adding this mask would exceed the maximum ratio
                if total_frames_masked + mask_width > max_frames_to_mask:
                    # Skip this mask if it would exceed the limit
                    continue
                
                # Randomly select starting position
                mask_start = torch.randint(0, max(1, T - mask_width + 1), (1,)).item()
                mask_end = min(mask_start + mask_width, T)
                
                # Apply time mask (set to 0)
                augmented_feats[b, mask_start:mask_end, :] = 0.0
                
                total_frames_masked += mask_end - mask_start
                num_masks_applied += 1
        
        return augmented_feats

    def _create_time_mask(self, batch_size, seq_len, mask_prob=0.1, mask_length=10):
        """
        Create a time mask for CR-CTC augmentation.
        
        Args:
            batch_size: Number of samples in batch
            seq_len: Sequence length
            mask_prob: Probability of starting a mask at each position
            mask_length: Maximum length of each mask span
            
        Returns:
            mask: Boolean tensor of shape [B, T], True indicates masked positions
        """
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        
        for b in range(batch_size):
            # Randomly select positions to start masks
            num_masks = max(1, int(seq_len * mask_prob / mask_length))
            mask_starts = torch.randint(0, max(1, seq_len - mask_length), (num_masks,))
            
            for start in mask_starts:
                length = torch.randint(1, mask_length + 1, (1,)).item()
                end = min(start + length, seq_len)
                mask[b, start:end] = True
                
        return mask

    def compute_objectives(self, predictions, batch, stage):
        "Given the network predictions and targets computed the NLL loss."

        p_ctc, wav_lens, time_mask, is_crctc_mode = predictions
        ids = batch.id
        targets, target_lens = batch.phn_encoded_target
        canonicals, canonical_lens = batch.phn_encoded_canonical
        perceiveds, perceived_lens = batch.phn_encoded_perceived

        if is_crctc_mode:
            # CR-CTC mode: p_ctc is [2*B, T, C], need to split and compute losses
            B = len(ids)
            
            # Duplicate targets for the combined batch
            targets_combined = targets.repeat(2, 1) if targets.dim() > 1 else targets.repeat(2)
            target_lens_combined = target_lens.repeat(2)
            
            # Compute CTC loss on combined batch
            loss_ctc = self.hparams.ctc_cost(
                p_ctc, targets_combined, wav_lens, target_lens_combined
            )
            
            # Compute Consistency Regularization (CR) loss
            # Split predictions back into two halves
            p_ctc_1, p_ctc_2 = p_ctc.chunk(2, dim=0)  # Each [B, T, C]
            
            # CR loss: KL divergence between the two augmented outputs
            # Use detached targets for stable training
            cr_loss = self._compute_cr_loss(
                p_ctc_1, p_ctc_2, 
                wav_lens[:B],  # Original wav_lens
                time_mask
            )
            # import pdb; pdb.set_trace()
            
            # Combine losses
            # Scale CTC loss by 0.5 since we have 2x samples
            loss = 0.5 * loss_ctc + self.cr_loss_weight * cr_loss
            
            # Track losses for logging
            if stage == sb.Stage.TRAIN:
                self.cr_loss_sum += cr_loss.detach().item()
                self.cr_loss_count += 1
                self.ctc_loss_sum += (0.5 * loss_ctc).detach().item()
                self.ctc_loss_count += 1
            
        else:
            # Standard mode
            loss_ctc = self.hparams.ctc_cost(p_ctc, targets, wav_lens, target_lens)
            cr_loss = torch.tensor(0.0, device=self.device)
            loss = loss_ctc
            
            # Track CTC loss for logging
            if stage == sb.Stage.TRAIN:
                self.ctc_loss_sum += loss_ctc.detach().item()
                self.ctc_loss_count += 1

        # Record losses for posterity
        if stage != sb.Stage.TRAIN:
            # Use original batch size predictions for evaluation
            if is_crctc_mode:
                # Take first half for evaluation
                p_ctc_eval = p_ctc[:len(ids)]
                wav_lens_eval = wav_lens[:len(ids)]
            else:
                p_ctc_eval = p_ctc
                wav_lens_eval = wav_lens
                
            sequence = sb.decoders.ctc_greedy_decode(
                p_ctc_eval, wav_lens_eval, blank_id=self.hparams.blank_index
            )
            
            self.ctc_metrics.append(ids, p_ctc_eval, targets, wav_lens_eval, target_lens)
            
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
    
    def _compute_cr_loss(self, p_ctc_1, p_ctc_2, wav_lens, time_mask=None):
        """
        Compute Consistency Regularization loss between two augmented outputs.
        
        Args:
            p_ctc_1: Log probabilities from first augmentation [B, T, C]
            p_ctc_2: Log probabilities from second augmentation [B, T, C]
            wav_lens: Relative lengths of sequences [B]
            time_mask: Optional mask (not used in current implementation)
            
        Returns:
            cr_loss: Scalar consistency regularization loss
        """
        B, T, C = p_ctc_1.shape
        
        # KL divergence: symmetric version
        # KL(p1 || p2) + KL(p2 || p1) where p1, p2 are log probabilities
        # Using log_target=True since both are log probabilities
        
        kl_1_to_2 = torch.nn.functional.kl_div(
            input=p_ctc_1,
            target=p_ctc_2.detach(),
            reduction="none",
            log_target=True,
        )  # [B, T, C]
        
        kl_2_to_1 = torch.nn.functional.kl_div(
            input=p_ctc_2,
            target=p_ctc_1.detach(),
            reduction="none",
            log_target=True,
        )  # [B, T, C]
        
        # Symmetric KL loss
        cr_loss = kl_1_to_2 + kl_2_to_1  # [B, T, C]
        
        # Create length mask to ignore padded positions
        max_len = T
        abs_lens = (wav_lens * max_len).long()  # Convert relative to absolute lengths
        length_mask = torch.arange(max_len, device=self.device).unsqueeze(0) >= abs_lens.unsqueeze(1)
        length_mask = length_mask.unsqueeze(-1)  # [B, T, 1]
        
        # Zero out padded positions
        cr_loss = cr_loss.masked_fill(length_mask, 0.0)
        
        # Average over valid positions
        num_valid = (~length_mask).sum()
        cr_loss = cr_loss.sum() / max(1, num_valid)
        
        return cr_loss
    
    def on_stage_start(self, stage, epoch):
        "Gets called when a stage (either training, validation, test) starts."
        self.ctc_metrics = self.hparams.ctc_stats()
        if hasattr(self.hparams, "augmentation"):
            self.modules.perceived_ssl.model.config.apply_spec_augment = True

        # Reset CR loss tracking at the start of each epoch
        if stage == sb.Stage.TRAIN:
            self.cr_loss_sum = 0.0
            self.cr_loss_count = 0
            self.ctc_loss_sum = 0.0
            self.ctc_loss_count = 0

        if stage != sb.Stage.TRAIN:
            self.per_metrics = self.hparams.per_stats()
            self.mpd_metrics = MpdStats()
    
    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch."""
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
            # Compute average CR loss and CTC loss for the epoch
            self.avg_cr_loss = self.cr_loss_sum / max(1, self.cr_loss_count)
            self.avg_ctc_loss_train = self.ctc_loss_sum / max(1, self.ctc_loss_count)
        else:
            per = self.per_metrics.summarize("error_rate")
            mpd_f1 = self.mpd_metrics.summarize("mpd_f1")

        if stage == sb.Stage.VALID:
            # Log stats
            self.hparams.train_logger.log_stats(
                stats_meta={
                    "epoch": epoch,
                    "lr_adam": self.adam_optimizer.param_groups[0]["lr"],
                    "lr_pretrained": self.pretrained_opt_class.param_groups[0]["lr"],
                },
                train_stats={
                    "loss": self.train_loss,
                    "ctc_loss": getattr(self, 'avg_ctc_loss_train', 0.0),
                    "cr_loss": getattr(self, 'avg_cr_loss', 0.0),
                },
                valid_stats={
                    "loss": stage_loss,
                    "ctc_loss": self.ctc_metrics.summarize("average"),
                    "PER": per,
                    "mpd_f1": mpd_f1
                },
            )
            
            # Save best models
            improved = False
            ckpt_name = f"{epoch:03d}_PER_{per:.4f}_F1_{mpd_f1:.4f}.ckpt"

            self.checkpointer.save_and_keep_only(
                meta={"PER": per, "mpd_f1": mpd_f1, "epoch": epoch},
                name=ckpt_name,
                num_to_keep=3,
                importance_keys=[
                    lambda ckpt: (-ckpt.meta["PER"], ckpt.meta["mpd_f1"]),
                ]
            )

            # Early stopping logic
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

            # Logging to wandb
            wandb.log({
                "epoch": epoch,
                "train_loss": self.train_loss,
                "train_ctc_loss": getattr(self, 'avg_ctc_loss_train', 0.0),
                "train_cr_loss": getattr(self, 'avg_cr_loss', 0.0),
                "valid_loss": stage_loss,
                "valid_ctc_loss": self.ctc_metrics.summarize("average"),
                "PER": per,
                "mpd_f1": mpd_f1,
            }, step=epoch)
            
            # Early stop if patience exceeded
            if self.no_improve_epochs >= self.patience:
                print(f"Early stopping at epoch {epoch}, no improvement for {self.patience} epochs.")
                raise KeyboardInterrupt("Early stopping triggered")

        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats={"loss": stage_loss, "PER": per, "mpd_f1": mpd_f1},
            )
            
            with open(self.hparams.per_file, "w") as w:
                self.per_metrics.write_stats(w)
                
            with open(self.hparams.mpd_file, "w") as m:
                self.mpd_metrics.write_stats(m)