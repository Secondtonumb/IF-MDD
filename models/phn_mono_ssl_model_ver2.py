import os
import sys
import torch
import torch.nn
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
from torch.nn.functional import kl_div
from speechbrain.nnet.loss.guidedattn_loss import GuidedAttentionLoss
from speechbrain.lobes.models.dual_path import PyTorchPositionalEncoding
from speechbrain.nnet.attention import RelPosEncXL, RelPosMHAXL
from speechbrain.lobes.models.VanillaNN import VanillaNN
import re
from speechbrain.utils.metric_stats import MetricStats
from speechbrain.nnet.losses import ctc_loss
from torch.nn import functional as F
import pdb
# from losses.BCE_Loss import BCELoss

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
        if self.modules.perceived_ssl is not None:
            self.modules.perceived_ssl.to(self.device)
        if self.modules.canonical_ssl is not None:
            self.modules.canonical_ssl.to(self.device)

        self.best_valid_loss = float('inf')
        self.best_valid_loss_list = []  # List of (valid_loss, epoch, ckpt_name)

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

        # output layer for ctc log-probabilities
        logits = self.modules.ctc_lin(x)
        p_ctc = self.hparams.log_softmax(logits) # (B, T, C)
        return p_ctc, wav_lens

    def compute_objectives(self, predictions, batch, stage):
        "Given the network predictions and targets computed the NLL loss."
        p_ctc, wav_lens = predictions
        ids = batch.id
        targets, target_lens = batch.phn_encoded_target
        # Additional: BCE loss on binary mispronunciation prediction
        if stage != sb.Stage.TRAIN:
            canonicals, canonical_lens = batch.phn_encoded_canonical
            perceiveds, perceived_lens = batch.phn_encoded_perceived

        loss_ctc = self.hparams.ctc_cost(p_ctc, targets, wav_lens, target_lens)
        loss = loss_ctc

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

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        predictions = self.compute_forward(batch, stage=stage)
        loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()

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
            # Save best 3 Models
            improved = False
            # Save best 3 PER models (lower is better)
            if per < self.best_per or len(self.best_per_list) < 3:
                ckpt_name = f"best_per_{epoch:03d}_{per:.4f}.ckpt"
                self.checkpointer.save_and_keep_only(
                    meta={"PER": per, "mpd_f1": mpd_f1, "epoch": epoch},
                    name=ckpt_name,
                    num_to_keep=5,
                    min_keys=["PER"]
                )
                self.best_per_list.append((per, epoch, ckpt_name))
                self.best_per_list = sorted(self.best_per_list, key=lambda x: x[0])[:3]
                self.best_per = self.best_per_list[0][0]
                improved = True
                # Remove extra checkpoints
                if len(self.best_per_list) > 3:
                    to_remove = self.best_per_list[3:]
                    for _, _, name in to_remove:
                        self.checkpointer.delete_checkpoint(name)
                    self.best_per_list = self.best_per_list[:3]
            # Save best 3 mpd_f1 models (higher is better)
            if mpd_f1 > self.best_mpd_f1 or len(self.best_mpd_f1_list) < 3:
                ckpt_name = f"best_mpdf1_{epoch:03d}_{mpd_f1:.4f}.ckpt"
                self.checkpointer.save_and_keep_only(
                    meta={"PER": per, "mpd_f1": mpd_f1, "epoch": epoch},
                    name=ckpt_name,
                    num_to_keep=5,
                    max_keys=["mpd_f1"]
                )
                self.best_mpd_f1_list.append((mpd_f1, epoch, ckpt_name))
                self.best_mpd_f1_list = sorted(self.best_mpd_f1_list, key=lambda x: -x[0])[:3]
                self.best_mpd_f1 = self.best_mpd_f1_list[0][0]
                improved = True
                # Remove extra checkpoints
                if len(self.best_mpd_f1_list) > 3:
                    to_remove = self.best_mpd_f1_list[3:]
                    for _, _, name in to_remove:
                        self.checkpointer.delete_checkpoint(name)
                    self.best_mpd_f1_list = self.best_mpd_f1_list[:3]

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
                self.scaler.step(self.pretrained_opt_class)
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

        # Load latest checkpoint to resume training if interrupted
        ## NOTE: make sure to use the "best" model to continual training
        ## so we set the `min_key` argument
        if self.checkpointer is not None:
            # TODO: support recover best on PER or mpd_f1 or averaged model of best PER and mpd_f1
            self.checkpointer.recover_if_possible(
                # min_key="PER",
                max_key="mpd_f1",
            )

class HMA_attn_ctc_to_mispro_ver2(PhnMonoSSLModel):
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
        # pdb.set_trace()
        super().__init__(*args, **kwargs)
        # Canonical Phoneme Embedding, LSTM + Linear, LSTM out as key, Linear out as value
        self.AcousticEnc = VanillaNN(
            input_shape=(None, None, 768),  # [B, T, ENC_DIM]
            dnn_neurons=384,  # Output dimension matches input dimension
        )
        
        self.PhnEmb = torch.nn.Sequential(
            torch.nn.Embedding(42, 384),  # 42 phonemes, 384 dim
            PyTorchPositionalEncoding(d_model=384, max_len=1000),  # Positional encoding
        )
        self.PhnEmb_Key = torch.nn.Sequential(
            torch.nn.Linear(384, 384),  # Linear layer to transform phoneme embeddings to key space
            torch.nn.LSTM(384, 384//2, batch_first=True, bidirectional=True)  # LSTM to process phoneme embeddings
        )
        
        self.PhnEmb_Value = torch.nn.Linear(384, 384)  # Linear layer to transform phoneme embeddings to value space

        self.MHA = torch.nn.MultiheadAttention(
            embed_dim=384, num_heads=8, batch_first=True, dropout=0.1
        )
        self.attn_out = VanillaNN(
            input_shape=(None, None, 384),  # [B, T, ENC_DIM]
            dnn_neurons=42,  # Output dimension matches input dimension
            activation=torch.nn.LogSoftmax  # Softmax for probabilities
        )
        self.TransformerDecoder = torch.nn.TransformerDecoder(
            torch.nn.TransformerDecoderLayer(
                d_model=384, nhead=8, dim_feedforward=1024, dropout=0.1
            ),
            num_layers=6,
        )
        # Acoustic CTC head for phoneme recognition
        self.ctc_phn_lin = torch.nn.Sequential(
            torch.nn.Linear(384, 42),  # Output dimension matches phoneme classes
            torch.nn.LogSoftmax(dim=-1)  # LogSoftmax for CTC probabilities
        )
        # [Acoustic + Attn] CTC head for phoneme recognition 
        self.ctc_concat_head = torch.nn.Sequential(
            torch.nn.Linear(384+768, 42),  # Output dimension matches phoneme classes
            torch.nn.LogSoftmax(dim=-1)  # LogSoftmax for CTC probabilities
        )
        # Binary mispronunciation detection head
        self.mispro_head = VanillaNN(
            input_shape=(None, None, 384+768),  # [B, T, ENC_DIM]
            dnn_neurons= 1,  # Binary classification
            activation=torch.nn.Sigmoid  # Sigmoid for binary classification
        ) 
        # Sequence labeling decoder 
        self.lab_sequence_head = torch.nn.Sequential(
            torch.nn.Linear(384+768, 42),  # Output dimension matches phoneme classes
            torch.nn.LogSoftmax(dim=-1)  # LogSoftmax for CTC probabilities
        )

        self.modules_to_train = torch.nn.ModuleList([
            self.AcousticEnc,
            self.PhnEmb,
            self.PhnEmb_Key,
            self.PhnEmb_Value,
            self.MHA,
            self.attn_out,
            self.TransformerDecoder,
            self.ctc_phn_lin,
            self.ctc_concat_head,
            self.mispro_head,
            self.lab_sequence_head
        ]).to(self.device)

        self.best_mpd_f1_fuse_list = []  # List of (mpd_f1, epoch, ckpt_name)
        self.best_mpd_f1_fuse = float('-inf')
        self.best_mpd_per_list = []  # List of (PER, epoch, ckpt_name)
        self.best_mpd_per = float('inf')

    def compute_forward(self, batch, stage):
        "Given an input batch it computes the phoneme probabilities."
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        canonicals, canonical_lens = batch.phn_encoded_canonical # [B, T_p]
        perceiveds, perceived_lens = batch.phn_encoded_perceived # [B, T_p]
        feats = self.modules.perceived_ssl(wavs) # [B, T_s, ENC_DIM]
        x = self.AcousticEnc(feats) # [B, T_s, D]
        p_ctc_feat = self.ctc_phn_lin(x)  # [B, T_s, C]

        Phn_h = self.PhnEmb(canonicals) # [B, T_p, D]
        Phn_h_key, _ = self.PhnEmb_Key(Phn_h)
        Phn_h_value = self.PhnEmb_Value(Phn_h)  # [B, T_p, D]
        # ---- Cross-attention with canonical phoneme embeddings ----
        
        # SpeechBrain length tensors are ratios in [0,1]; convert to absolute token lengths
        max_ph = Phn_h.size(1)
        canon_token_lens = (canonical_lens.to(self.device).float() * max_ph).round().clamp(max=max_ph).long()  # [B]
        key_padding_mask = torch.arange(max_ph, device=self.device).unsqueeze(0) >= canon_token_lens.unsqueeze(1)  # [B, S], True = pad

        max_s = feats.size(1)
        speech_token_lens = (wav_lens.to(self.device).float() * max_s).round().clamp(max=max_s).long()
        query_pad_mask = torch.arange(max_s, device=self.device).unsqueeze(0) >= speech_token_lens.unsqueeze(1)  # [B, T]
        
        # Fuse the phoneme embeddings with the acoustic features using multi-head attention
        attn, attn_map = self.MHA(
            query=x,  # [B, T_s, D]
            key=Phn_h_key,  # [B, T_p, D]
            value=Phn_h_value,  # [B, T_p, D]
            key_padding_mask=key_padding_mask,
        )  # [B, T_s, D]

        # concat attn and acoustic features
        attn_concat = torch.cat([feats, attn], dim=-1)  # [B, T_s, D1 + D2]
        #m Phoneme recognition CTC head
        p_ctc_concat = self.ctc_concat_head(attn_concat)  # [B, T_s, C]
        # mispronunciation detection head
        out_mispro = self.mispro_head(attn_concat)  # [B, T_s, 1]
        # sequence labeling decoder
        out_labseq = self.lab_sequence_head(attn_concat)  # [B, T_s + T_p, C]

        return {
            "p_ctc_feat": p_ctc_feat,  # [B, T_s, C]
            "p_ctc_concat": p_ctc_concat,  # [B, T_s + T_p, C]
            "out_labseq": out_labseq,  # [B, T_s + T_p, C]
            "out_mispro": out_mispro,  # [B, T_s + T_p, 1]
            "feats": feats,  # [B, T_s, D]
            "attn_map": attn_map,  # [B, T_s, T_p]
        }
    def compute_objectives(self, predictions, batch, stage):
        "Given the network predictions and targets computed the NLL loss."
        p_ctc_feat = predictions["p_ctc_feat"]
        p_ctc_concat = predictions["p_ctc_concat"]
        out_mispro = predictions["out_mispro"]
        out_labseq = predictions["out_labseq"]
        feats = predictions["feats"]
        attn_map = predictions["attn_map"]

        # GTs
        ids = batch.id
        wavs, wav_lens = batch.sig
        targets, target_lens = batch.phn_encoded_target
        canonicals, canonical_lens = batch.phn_encoded_canonical
        perceiveds, perceived_lens = batch.phn_encoded_perceived
        
        mispronunciation_labels, _ = batch.mispro_label_framewise
        phn_seq_labels, _ = batch.phn_encoded_target_bin

        # CTC losses
        loss_ctc = self.hparams.ctc_cost(p_ctc_feat, targets, wav_lens, target_lens)
        loss_attn_concat = self.hparams.ctc_cost(p_ctc_concat, targets, wav_lens, target_lens)

        # Framewise mispronunciation detection loss
        ds_mislabels = downsample_pool_argmax(
            mispronunciation_labels, 
            T_out=p_ctc_feat.size(1), 
            C=2  # Binary classification: mispronounced or not
        )
        ds_seqlabels = downsample_pool_argmax(
            phn_seq_labels, 
            T_out=p_ctc_feat.size(1), 
            C=42  # 42 phonemes
        )

        # Binary mispronunciation detection loss
        loss_mispro = sb.nnet.losses.bce_loss(
            inputs=out_mispro,
            targets=ds_mislabels,
            length=wav_lens,
        )

        CE_loss = torch.nn.CrossEntropyLoss(ignore_index=self.label_encoder.lab2ind["<blank>"])
        loss_label_seq = CE_loss(
            input=out_labseq.view(-1, out_labseq.size(-1)),  # Flatten to [B*T, C]
            target=ds_seqlabels.view(-1)  # Flatten to [B*T
        )

        loss = loss_ctc + loss_attn_concat + 5* loss_mispro + 0.1*loss_label_seq

        # Record losses for posterity
        if stage != sb.Stage.TRAIN:
            # 只保留主 CTC 路径和 concat 路径的解码与指标
            sequence = sb.decoders.ctc_greedy_decode(
                p_ctc_feat, wav_lens, blank_id=self.hparams.blank_index
            )
            sequence_fuse = sb.decoders.ctc_greedy_decode(
                p_ctc_concat, wav_lens, blank_id=self.hparams.blank_index
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

            self.ctc_metrics.append(ids, p_ctc_feat, targets, wav_lens, target_lens)
            self.ctc_metrics_fuse.append(ids, p_ctc_concat, targets, wav_lens, target_lens)
            self.per_metrics.append(
                ids=ids,
                predict=sequence,
                target=targets,
                predict_len=None,
                target_len=target_lens,
                ind2lab=self.label_encoder.decode_ndim,
            )
            self.per_metrics_fuse.append(
                ids=ids,
                predict=sequence_fuse,
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
            self.mpd_metrics_fuse.append(
                ids=ids,
                predict=sequence_fuse,
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
                    "loss": loss.item(),
                    "loss_ctc_head": loss_ctc.item(),
                    "loss_ctc_fuse": loss_attn_concat.item(),
                    "loss_mispro": loss_mispro.item(),
                    "loss_label_seq": loss_label_seq.item(),
                }, step=self.hparams.epoch_counter.current)
            except Exception:
                pass

        return loss

    def init_optimizers(self):
        modules_to_be_trained = torch.nn.ModuleList([
            self.hparams.model,
            self.modules_to_train
        ])
        self.adam_optimizer = self.hparams.adam_opt_class(
            modules_to_be_trained.parameters(),
        )
        
        self.pretrained_opt_class = self.hparams.pretrained_opt_class(
            self.modules.perceived_ssl.parameters(), 
        )
        if self.checkpointer is not None:
            # if self.hparams.perceived_ssl is not None and not self.hparams.perceived_ssl.freeze:
            self.checkpointer.add_recoverable("adam_opt", self.adam_optimizer)
            self.checkpointer.add_recoverable("pretrained_opt", self.pretrained_opt_class)
    def on_stage_start(self, stage, epoch):

        "Gets called when a stage (either training, validation, test) starts."
        self.ctc_metrics = self.hparams.ctc_stats()
        self.ctc_metrics_attn = self.hparams.ctc_stats()
        self.ctc_metrics_fuse =self.hparams.ctc_stats()


        if stage != sb.Stage.TRAIN:
  
            self.per_metrics = self.hparams.per_stats()
            self.mpd_metrics = MpdStats()
            
            self.per_metrics_attn = self.hparams.per_stats()
            self.mpd_metrics_attn = MpdStats()

            self.per_metrics_fuse = self.hparams.per_stats()
            self.mpd_metrics_fuse = MpdStats()
    
    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        else:
            per = self.per_metrics.summarize("error_rate")
            mpd_f1 = self.mpd_metrics.summarize("mpd_f1")
            per_attn = self.per_metrics_attn.summarize("error_rate")
            mpd_f1_attn = self.mpd_metrics_attn.summarize("mpd_f1")
            per_fuse = self.per_metrics_fuse.summarize("error_rate")
            mpd_f1_fuse = self.mpd_metrics_fuse.summarize("mpd_f1")

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
                    "ctc_loss_fuse": self.ctc_metrics_fuse.summarize("average"),
                    "PER": per,
                    "mpd_f1": mpd_f1,
                    "PER_attn": per_attn,
                    "mpd_f1_attn": mpd_f1_attn,
                    "PER_fuse": per_fuse,
                    "mpd_f1_fuse": mpd_f1_fuse,
                },
            )
            # Save best 3 Models based on PER_* and mpd_f1_* 
            improved = False
            # Save best 3 PER models (lower is better)
            if per < self.best_per or len(self.best_per_list) < 3:
                ckpt_name = f"best_per_{epoch:03d}_{per:.4f}.ckpt"
                self.checkpointer.save_and_keep_only(
                    meta={"PER": per, "mpd_f1": mpd_f1, "epoch": epoch},
                    name=ckpt_name,
                    num_to_keep=5,
                    min_keys=["PER"]
                )
                self.best_per_list.append((per, epoch, ckpt_name))
                self.best_per_list = sorted(self.best_per_list, key=lambda x: x[0])[:3]
                self.best_per = self.best_per_list[0][0]
                improved = True
                # Remove extra checkpoints
                if len(self.best_per_list) > 3:
                    to_remove = self.best_per_list[3:]
                    for _, _, name in to_remove:
                        self.checkpointer.delete_checkpoint(name)
                    self.best_per_list = self.best_per_list[:3]
            # Save best 3 mpd_f1 models (higher is better)
            if mpd_f1 > self.best_mpd_f1 or len(self.best_mpd_f1_list) < 3:
                ckpt_name = f"best_mpdf1_{epoch:03d}_{mpd_f1:.4f}.ckpt"
                self.checkpointer.save_and_keep_only(
                    meta={"PER": per, "mpd_f1": mpd_f1, "epoch": epoch},
                    name=ckpt_name,
                    num_to_keep=5,
                    max_keys=["mpd_f1"]
                )
                self.best_mpd_f1_list.append((mpd_f1, epoch, ckpt_name))
                self.best_mpd_f1_list = sorted(self.best_mpd_f1_list, key=lambda x: -x[0])[:3]
                self.best_mpd_f1 = self.best_mpd_f1_list[0][0]
                improved = True
                # Remove extra checkpoints
                if len(self.best_mpd_f1_list) > 3:
                    to_remove = self.best_mpd_f1_list[3:]
                    for _, _, name in to_remove:
                        self.checkpointer.delete_checkpoint(name)
                    self.best_mpd_f1_list = self.best_mpd_f1_list[:3]   
            # Save best 3 PER_fuse models (lower is better)
            if per_fuse < self.best_per_fuse or len(self.best_per_fuse_list) < 3:
                ckpt_name = f"best_per_fuse_{epoch:03d}_{per_fuse:.4f}.ckpt"
                self.checkpointer.save_and_keep_only(
                    meta={"PER_fuse": per_fuse, "mpd_f1_fuse": mpd_f1_fuse, "epoch": epoch},
                    name=ckpt_name,
                    num_to_keep=5,
                    min_keys=["PER_fuse"]
                )
                self.best_per_fuse_list.append((per_fuse, epoch, ckpt_name))
                self.best_per_fuse_list = sorted(self.best_per_fuse_list, key=lambda x: x[0])[:3]
                self.best_per_fuse = self.best_per_fuse_list[0][0]
                improved = True
                # Remove extra checkpoints
                if len(self.best_per_fuse_list) > 3:
                    to_remove = self.best_per_fuse_list[3:]
                    for _, _, name in to_remove:
                        self.checkpointer.delete_checkpoint(name)
                    self.best_per_fuse_list = self.best_per_fuse_list[:3]
            # Save best 3 mpd_f1_fuse models (higher is better)
            if mpd_f1_fuse > self.best_mpd_f1_fuse or len(self.best_mpd_f1_fuse_list) < 3:
                ckpt_name = f"best_mpdf1_fuse_{epoch:03d}_{mpd_f1_fuse:.4f}.ckpt"
                self.checkpointer.save_and_keep_only(
                    meta={"PER_fuse": per_fuse, "mpd_f1_fuse": mpd_f1_fuse, "epoch": epoch},
                    name=ckpt_name,
                    num_to_keep=5,
                    max_keys=["mpd_f1_fuse"]
                )
                self.best_mpd_f1_fuse_list.append((mpd_f1_fuse, epoch, ckpt_name))
                self.best_mpd_f1_fuse_list = sorted(self.best_mpd_f1_fuse_list, key=lambda x: -x[0])[:3]
                self.best_mpd_f1_fuse = self.best_mpd_f1_fuse_list[0][0]
                improved = True
                # Remove extra checkpoints
                if len(self.best_mpd_f1_fuse_list) > 3:
                    to_remove = self.best_mpd_f1_fuse_list[3:]
                    for _, _, name in to_remove:
                        self.checkpointer.delete_checkpoint(name)
                    self.best_mpd_f1_fuse_list = self.best_mpd_f1_fuse_list[:3]
            

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
                "ctc_loss_fuse": self.ctc_metrics_fuse.summarize("average"),
                "PER": per,
                "PER_fuse": per_fuse,
                "mpd_f1_attn": mpd_f1_attn,
                "mpd_f1": mpd_f1,
                "mpd_f1_fuse": mpd_f1_fuse,
            }, step=epoch)
            # Early stop if patience exceeded
            if self.no_improve_epochs >= self.patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {self.patience} epochs)")
                raise StopIteration

        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats={"loss": stage_loss, "PER": per, "mpd_f1": mpd_f1, 
                            "PER_fuse": per_fuse, "mpd_f1_fuse": mpd_f1_fuse},
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