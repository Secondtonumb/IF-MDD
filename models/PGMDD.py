import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, MultiheadAttention, Conv1d, Linear
import torch.nn.functional as F
import speechbrain as sb
from torch.nn import Conv1d
# from trainer.ArticulationLoader import ArticulatoryFeatureExtractor
from mpd_eval_v3 import MpdStats

import sys
sys.path.append("../trainer")  # Adjust the path as necessary
from trainer.AudioSSLoader import AudioSSLLoader

class PGMDD(sb.Brain):
    def __init__(self, *args, patience=20, **kwargs):
        super().__init__(*args, **kwargs)
        self.patience = patience
        self.no_improve_epochs = 0
        self.best_per_list = []  # List of (PER, epoch, ckpt_name)
        self.best_mpd_f1_list = []  # List of (mpd_f1, epoch, ckpt_name)
        self.best_per = float('inf')
        self.best_mpd_f1 = float('-inf')
        self.last_improved_epoch = 0
        self.best_valid_loss = float('inf')
        self.best_valid_loss_list = []  # List of (valid_loss, epoch, ckpt_name)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
        self.perceived_encoder = AudioSSLLoader(
            model_name="wav2vec2_base",
            freeze=False,
            freeze_feature_extractor=True,
            save_path="pretrained_models/"
        )
        self.CanoPhnEmbedding = torch.nn.Embedding(
            num_embeddings=42,  # Example size, adjust as needed
            embedding_dim=768  # Example dimension, adjust as needed
        )
        # Prompt encoder: embedding + single-layer Transformer
        enc_layer = TransformerEncoderLayer(d_model=768, nhead=8)
        self.CanoTransEncoder = TransformerEncoder(enc_layer, num_layers=1)
        # Cross-attention aligner
        self.Aligner = MultiheadAttention(embed_dim=768, num_heads=8)
        # Concat Transformer encoder (2 layers)
        concat_layer = TransformerEncoderLayer(d_model=768, nhead=8)
        self.ConcatTransEncoder = TransformerEncoder(concat_layer, num_layers=2)
        # Depth-wise convolution for multi-view fusion
        self.depthwise_conv = Conv1d(in_channels=768*2, out_channels=768*2, kernel_size=3, padding=1, groups=768*2)
        # Projection back to model dimension
        self.depth_proj = Linear(768*2, 768)
        # Mispronunciation detection head
        self.mispro_head = Linear(768, 1)
        # Phone prediction head for CTC
        self.Phn_pred_head = Linear(768, 42)
        # Articulatory feature extractor

    def compute_forward(self, batch, stage):
        """
        Given an input batch it computes the CTC logits and mispronunciation probabilities.
        """
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        canon_seq, canon_lens = batch.phn_encoded_canonical

        # Acoustic features from SSL encoder
        H_a = self.perceived_encoder(wavs)
        # Articulatory features
        H_art_raw = self.artic_encoder(wavs)
        H_art = self.depth_proj(torch.cat([
            H_a.permute(0,2,1),
            Conv1d(in_channels=H_art_raw.size(-1),
                   out_channels=768,
                   kernel_size=1)(H_art_raw.permute(0,2,1))
        ], dim=1)).permute(0,2,1)

        # Multi-view fusion
        H_mv = torch.cat([H_a.permute(0,2,1), H_art.permute(0,2,1)], dim=1)
        H_mv = self.depthwise_conv(H_mv)
        H_mv = self.depth_proj(H_mv.permute(0,2,1))

        # Prompt encoding
        E_q = self.CanoPhnEmbedding(canon_seq).permute(1,0,2)
        H_p = self.CanoTransEncoder(E_q).permute(1,0,2)
        H_align, _ = self.Aligner(H_p.permute(1,0,2), H_mv.permute(1,0,2), H_mv.permute(1,0,2))
        H_align = H_align.permute(1,0,2)
        concat_input = torch.cat([H_p.permute(1,0,2), H_align.permute(1,0,2)], dim=2)
        H_concat = self.ConcatTransEncoder(concat_input).permute(1,0,2)

        # Heads
        mispro_logits = self.mispro_head(H_concat).squeeze(-1)
        mispro_probs = torch.sigmoid(mispro_logits)
        phn_logits = self.Phn_pred_head(H_mv)

        # CTC probabilities
        p_ctc = self.hparams.log_softmax(phn_logits)
        return p_ctc, mispro_probs, wav_lens

    def compute_objectives(self, predictions, batch, stage):
        """
        Given predictions and batch, compute the loss.
        """
        p_ctc, mispro_probs, wav_lens = predictions
        ids = batch.id
        targets, target_lens = batch.phn_encoded_target

        # CTC loss
        loss = self.hparams.ctc_cost(p_ctc, targets, wav_lens, target_lens)

        # Append metrics in validation/test
        if stage != sb.Stage.TRAIN:
            self.ctc_metrics.append(ids, p_ctc, targets, wav_lens, target_lens)

        return loss
    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        predictions = self.compute_forward(batch, stage=stage)
        loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()
    
    def on_stage_start(self, stage, epoch=None):
        """Actions to perform at the start of each stage."""
        if stage != sb.Stage.TRAIN:
  
            self.per_metrics = self.hparams.per_stats()
            self.mpd_metrics = MpdStats()