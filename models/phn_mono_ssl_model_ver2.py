import os

import torch
import torch.nn

import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from mpd_eval_v3 import MpdStats


import wandb

from speechbrain.nnet.loss.guidedattn_loss import GuidedAttentionLoss
from speechbrain.lobes.models.dual_path import PyTorchPositionalEncoding
from speechbrain.nnet.attention import RelPosEncXL, RelPosMHAXL
from speechbrain.lobes.models.VanillaNN import VanillaNN
from speechbrain.nnet.transducer import transducer_joint
from speechbrain.utils.metric_stats import MetricStats
from speechbrain.nnet.losses import ctc_loss
from torch.nn import functional as F
import pdb
from speechbrain.decoders import S2STransformerBeamSearcher, S2STransformerGreedySearcher, S2SBaseSearcher, S2SBeamSearcher, S2SGreedySearcher
from speechbrain.lobes.models.transformer.Transformer import TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer, TransformerInterface
from speechbrain.decoders import S2STransformerBeamSearcher, CTCScorer, ScorerBuilder
from speechbrain.lobes.models.transformer.TransformerASR import TransformerASR

from speechbrain.decoders.utils import (
    _update_mem,
    inflate_tensor,
    mask_by_condition,
)
# from speechbrain.lobes.models.transformer.TransformerASR

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer, TransformerDecoderLayer

from utils.layers.subsampling import (Conv1dSubsampling,
    Conv2dSubsampling,
    Conv2dSubsampling2,
    Conv2dSubsampling6,
    Conv2dSubsampling8,
)

from utils.layers.utils import make_pad_mask

class MyTransformerDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, tgt, memory,
                tgt_mask=None,
                memory_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None,
                return_attention=False,
                **kwargs):
        
        # Self-attention
        tgt2, self_attn_weights = self.self_attn(
            tgt, tgt, tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=True,
            average_attn_weights=False
        )
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross-attention
        tgt2, cross_attn_weights = self.multihead_attn(
            tgt, memory, memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            need_weights=True,
            average_attn_weights=False
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # FFN
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        if return_attention:
            return tgt, self_attn_weights, cross_attn_weights
        else:
            return tgt

class MyTransformerDecoder(nn.Module):
    """Custom TransformerDecoder that can return attention weights from the last layer"""
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        # Store the first layer, and create copies for the rest
        self.layers = nn.ModuleList([decoder_layer])
        for _ in range(num_layers - 1):
            new_layer = MyTransformerDecoderLayer(
                d_model=decoder_layer.self_attn.embed_dim,
                nhead=decoder_layer.self_attn.num_heads,
                dim_feedforward=decoder_layer.linear1.out_features,
                dropout=decoder_layer.dropout.p if hasattr(decoder_layer.dropout, 'p') else 0.1
            )
            self.layers.append(new_layer)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        output = tgt
        self_attn_weights = None
        cross_attn_weights = None

        for i, mod in enumerate(self.layers):
            if i == len(self.layers) - 1:  # Last layer
                output, self_attn_weights, cross_attn_weights = mod(
                    output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                    return_attention=True
                )
            else:
                output = mod(
                    output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                    return_attention=False
                )

        if self.norm is not None:
            output = self.norm(output)

        return output, self_attn_weights, cross_attn_weights

class JointCTCAttentionDecoder:
    """
    Joint CTC-Attention Decoder for Hybrid ASR models
    Combines CTC and attention-based decoder outputs using different strategies
    """
    def __init__(self, 
                 blank_id=0, 
                 ctc_weight=0.3, 
                 att_weight=0.7,
                 beam_size=10,
                 max_len_ratio=1.0,
                 lm_weight=0.0):
        """
        Args:
            blank_id: CTC blank token ID
            ctc_weight: Weight for CTC scores (typically 0.3-0.5)
            att_weight: Weight for attention scores (typically 0.5-0.7)
            beam_size: Beam search size
            max_len_ratio: Maximum output length ratio relative to input
            lm_weight: Language model weight (if using external LM)
        """
        self.blank_id = blank_id
        self.ctc_weight = ctc_weight
        self.att_weight = att_weight
        self.beam_size = beam_size
        self.max_len_ratio = max_len_ratio
        self.lm_weight = lm_weight
        
    def joint_greedy_decode(self, ctc_log_probs, att_log_probs, input_lengths):
        """
        Simple greedy joint decoding
        Args:
            ctc_log_probs: [B, T, V] CTC log probabilities
            att_log_probs: [B, U, V] Attention decoder log probabilities  
            input_lengths: [B] Input sequence lengths
        Returns:
            List[List[int]]: Decoded sequences for each batch
        """
        batch_size = ctc_log_probs.size(0)
        results = []
        
        for b in range(batch_size):
            # Get valid lengths
            T = int(input_lengths[b] * ctc_log_probs.size(1))
            U = att_log_probs.size(1)
            
            # CTC greedy path
            ctc_path = torch.argmax(ctc_log_probs[b, :T], dim=-1)  # [T]
            
            # Remove blanks and consecutive duplicates for CTC
            ctc_tokens = []
            prev_token = None
            for token in ctc_path:
                if token != self.blank_id and token != prev_token:
                    ctc_tokens.append(token.item())
                prev_token = token
            
            # Attention decoder path
            att_tokens = torch.argmax(att_log_probs[b, :U], dim=-1).tolist()  # [U]
            
            # Simple fusion: take shorter sequence and combine scores
            min_len = min(len(ctc_tokens), len(att_tokens))
            if min_len == 0:
                results.append(ctc_tokens if len(ctc_tokens) > 0 else att_tokens)
                continue
                
            joint_sequence = []
            for i in range(min_len):
                # Get scores for current position
                ctc_score = ctc_log_probs[b, i, ctc_tokens[i]] if i < len(ctc_tokens) else -float('inf')
                att_score = att_log_probs[b, i, att_tokens[i]] if i < len(att_tokens) else -float('inf')
                
                # Weighted combination
                if ctc_tokens[i] == att_tokens[i]:
                    # Both agree - use the token
                    joint_sequence.append(ctc_tokens[i])
                else:
                    # Disagreement - use weighted scores to decide
                    combined_ctc = self.ctc_weight * ctc_score
                    combined_att = self.att_weight * att_score
                    
                    if combined_ctc > combined_att:
                        joint_sequence.append(ctc_tokens[i])
                    else:
                        joint_sequence.append(att_tokens[i])
            
            results.append(joint_sequence)
        
        return results
    
    def joint_beam_search(self, ctc_log_probs, att_log_probs, input_lengths, 
                         decoder_model=None, memory=None, memory_mask=None):
        """
        Joint CTC-Attention beam search decoding
        Args:
            ctc_log_probs: [B, T, V] CTC log probabilities
            att_log_probs: [B, U, V] Initial attention decoder probabilities
            input_lengths: [B] Input sequence lengths
            decoder_model: Transformer decoder model for incremental decoding
            memory: [T, B, D] Encoder memory for attention decoder
            memory_mask: [B, T] Memory padding mask
        Returns:
            List[List[int]]: Best decoded sequences for each batch
        """
        batch_size = ctc_log_probs.size(0)
        vocab_size = ctc_log_probs.size(-1)
        results = []
        
        for b in range(batch_size):
            T = int(input_lengths[b] * ctc_log_probs.size(1))
            
            # Initialize beam
            beam = [([], 0.0, 0)]  # (sequence, score, att_pos)
            
            # CTC prefix score computation
            ctc_probs = torch.exp(ctc_log_probs[b, :T])  # [T, V]
            
            max_len = int(T * self.max_len_ratio)
            
            for step in range(max_len):
                candidates = []
                
                for seq, score, att_pos in beam:
                    if len(seq) >= max_len:
                        candidates.append((seq, score, att_pos))
                        continue
                    # import pdb; pdb.set_trace()
                    # Get top-k tokens from vocabulary
                    for token in range(vocab_size):
                        if token == self.blank_id:
                            continue
                            
                        new_seq = seq + [token]
                        
                        # Compute CTC prefix score
                        ctc_score = self._ctc_prefix_score(new_seq, ctc_probs)
                        
                        # Compute attention score
                        if decoder_model is not None and memory is not None:
                            att_score = self._attention_score(
                                new_seq, decoder_model, memory[b:b+1], 
                                memory_mask[b:b+1] if memory_mask is not None else None
                            )
                        else:
                            # Use precomputed attention scores
                            if att_pos < att_log_probs.size(1):
                                att_score = att_log_probs[b, att_pos, token].item()
                            else:
                                att_score = -float('inf')
                        
                        # Joint score
                        joint_score = self.ctc_weight * ctc_score + self.att_weight * att_score
                        new_score = score + joint_score
                        
                        candidates.append((new_seq, new_score, att_pos + 1))
                
                # Select top beam_size candidates
                candidates.sort(key=lambda x: x[1], reverse=True)
                beam = candidates[:self.beam_size]
                
                # Early stopping if all beams end with EOS or reach max length
                if all(len(seq) >= max_len for seq, _, _ in beam):
                    break
            
            # Return best sequence
            best_seq = max(beam, key=lambda x: x[1])[0]
            results.append(best_seq)
        
        return results
    
    def _ctc_prefix_score(self, sequence, ctc_probs):
        """
        Compute CTC prefix score for a sequence
        Args:
            sequence: List[int] - token sequence
            ctc_probs: [T, V] - CTC probabilities
        Returns:
            float: CTC prefix score
        """
        if len(sequence) == 0:
            return 0.0
            
        T, V = ctc_probs.shape
        
        # Simple approximation: product of max probabilities for each token
        score = 0.0
        for token in sequence:
            
            if token < V:
                max_prob = torch.max(ctc_probs[:, token])
                score += torch.log(max_prob + 1e-10).item()
        
        return score / len(sequence)  # Normalize by length
    
    def _attention_score(self, sequence, decoder_model, memory, memory_mask=None):
        """
        Compute attention decoder score for a sequence
        Args:
            sequence: List[int] - token sequence
            decoder_model: Transformer decoder
            memory: [T, 1, D] - encoder memory
            memory_mask: [1, T] - memory mask
        Returns:
            float: Attention score
        """
        if len(sequence) == 0:
            return 0.0
        
        device = memory.device
        seq_tensor = torch.tensor([sequence], device=device)  # [1, U]
        
        try:
            # Get decoder output
            tgt = seq_tensor.transpose(0, 1)  # [U, 1]
            output, _, _ = decoder_model(
                tgt=tgt,
                memory=memory,
                memory_key_padding_mask=memory_mask
            )
            import pdb; pdb.set_trace()
            # Compute log probability for the sequence
            log_probs = F.log_softmax(output, dim=-1)  # [U, 1, V]
            
            score = 0.0
            for i, token in enumerate(sequence):
                if i < log_probs.size(0):
                    score += log_probs[i, 0, token].item()
            
            return score / len(sequence)  # Normalize by length
            
        except Exception as e:
            print(f"Error in attention scoring: {e}")
            return -float('inf')

# class MyTransformerDecoderSearcher(S2STransformerGreedySearcher):
#     def forward_step(self, inp_tokens, memory, enc_states, enc_lens):
#         pdb.set_trace()
#         memory = _update_mem(inp_tokens, memory)
#         pred, attn = self.model.forward(memory.unsqueeze(1), enc_states, enc_lens)
#         logits = self.fc(pred)
#         return logits[:, -1, :], memory, attn

# class MyTransDecAR_Searcher(S2SGreedySearcher):
#     def __init__(self, decoder_model, beam_size=10, max_len_ratio=1.0, *args, **kwargs):
#         self.model = decoder_model
    
#     def forward_step(self, inp_tokens, memory, enc_states, enc_lens):
        
#         memory = _update_mem(input_token, memory)
#         pred, attn = self.model.forward(inp_tokens, memory)

    
    
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
        if self.modules.enc is not None:
            self.modules.enc.to(self.device)
        if self.modules.ctc_lin is not None:
            self.modules.ctc_lin.to(self.device)

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
            import pdb; pdb.set_trace()
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
        # self.attn_out = VanillaNN(
        #     input_shape=(None, None, 384),  # [B, T, ENC_DIM]
        #     dnn_neurons=42,  # Output dimension matches input dimension
        #     activation=torch.nn.LogSoftmax  # Softmax for probabilities
        # )

        self.attn_out = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=384, nhead=8, dim_feedforward=1024, dropout=0.1
            ),
            num_layers=6,
        )
        self.TransformerDecoder = torch.nn.TransformerDecoder(
            torch.nn.TransformerDecoderLayer(
                d_model=384+768, nhead=8, dim_feedforward=1024, dropout=0.1
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
        self.best_per_fuse = float('inf')
        self.best_per_fuse_list = []  # List of (PER, epoch, ckpt_name)

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
        
        # apply attn mask as we know x is padded with wav_lens (rate lengths / max_s)
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
        attn_concat = self.TransformerDecoder(attn_concat, attn_concat, tgt_key_padding_mask=query_pad_mask)  # [B, T_s, D]

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
            T_out=p_ctc_concat.size(1), 
            C=2  # Binary classification: mispronounced or not
        )
        ds_seqlabels = downsample_pool_argmax(
            phn_seq_labels, 
            T_out=p_ctc_concat.size(1), 
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

        # Apply guided attention loss
        
        # ---- Guided Attention Loss ----
        # attn_map: [B,H,T_q,T_k] or [B,T_q,T_k]
        if attn_map.dim() == 4:
            A = attn_map.mean(dim=1)          # average heads -> [B,T_p,T_s]
        elif attn_map.dim() == 3:
            A = attn_map
        else:
            raise ValueError(f"Unexpected attn_map shape {attn_map.shape}")

        # GuidedAttentionLoss expects (B, targets, inputs).
        # 我们把“targets”当作 canonical tokens (K)， “inputs”当作 acoustic frames (Q)。
        _, T_p_max, T_s_max = A.shape
        # 语音帧长度
        in_lens_abs = (wav_lens * T_s_max).round().long().clamp(1, T_s_max)
        # canonical token 长度
        tgt_expd_lens_abs = (canonical_lens * T_p_max).round().long().clamp(1, T_p_max)        
        loss_ga = self.guided_attn_loss(
            A, input_lengths=in_lens_abs, target_lengths=tgt_expd_lens_abs
        )

        # lam_main = getattr(self.hparams, "loss_lambda", 0.5)
        # lam_ga   = getattr(self.hparams, "ga_lambda", 0.1)
        loss = loss_attn_concat + loss_ctc + 10*loss_ga

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
            if epoch %5 == 0 or epoch==1:
                # We visualize the first ele batch only
                # Note: If  attn_map is [B, T_s, T_p], we visualize the first element
                # Elese if attn_map is [B, n_heads, T_s, T_p], we visualize the all head as subplots
                if attn_map.ndim == 3:
                    for attn_id, attn in enumerate(attn_map[0: 1]):
                        # get valid target length and wav length from batch[0]
                        wav_len_b = int(wav_lens[attn_id] * attn.size(1))
                        target_len_b = int(target_lens[attn_id] * targets.size(1))
                        attn = attn[:wav_len_b, :target_len_b]  # Crop
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
                elif attn_map.ndim == 4:
                    for attn_id, attn in enumerate(attn_map[0: 1]):
                        # get valid target length and wav length from batch[0]
                        wav_len_b = int(wav_lens[attn_id] * attn.size(1))
                        target_len_b = int(target_lens[attn_id] * targets.size(1))
                        # attn = attn[:wav_len_b, :target_len_b]  # Crop
                        # draw the bpunding box on the attention map with wav_len_b and target_len_b
                        
                        
                        n_heads = attn.size(0)
                        fig, axs = plt.subplots(n_heads//2, 2, figsize=(5, 10))
                        fig.suptitle(f"Attention Map for ID {ids[attn_id]} - Epoch {epoch}")
                        if n_heads == 1:
                            axs = [axs]
                        for i in range(n_heads//2):
                            for j in range(2):
                                axs[i, j].imshow(attn[i*2+j].cpu().detach().numpy(), aspect='auto', origin='lower')
                                axs[i, j].set_xlabel("Canonical Phoneme Index")
                                axs[i, j].set_ylabel("Acoustic Feature Index")
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
                    "loss_bce_mispro": loss_mispro.item(),
                    "loss_mispro": loss_mispro.item(),
                    "loss_label_seq": loss_label_seq.item(),
                }, step=self.hparams.epoch_counter.current)
                if loss_ga is not None:
                    wandb.log({"loss_ga": loss_ga.item()}, step=self.hparams.epoch_counter.current)
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
            
            per_fuse = self.per_metrics_fuse.summarize("error_rate")
            mpd_f1_fuse = self.mpd_metrics_fuse.summarize("mpd_f1")

        current_epoch = self.hparams.epoch_counter.current
        if stage == sb.Stage.VALID and current_epoch % 5 == 0:
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
            per_fuse_file = self.hparams.per_file.replace(".txt", "_fuse.txt")
            mpd_fuse_file = self.hparams.mpd_file.replace(".txt", "_fuse.txt")
            self.hparams.per_fuse_file = per_fuse_file  
            self.hparams.mpd_fuse_file = mpd_fuse_file
            with open(self.hparams.per_fuse_file, "w") as w:
                w.write("CTC loss stats (fuse):\n")
                self.ctc_metrics_fuse.write_stats(w)
                w.write("\nPER stats (fuse):\n")
                self.per_metrics_fuse.write_stats(w)
                print(
                    "CTC and PER stats (fuse) written to file",
                    self.hparams.per_fuse_file,
                )
            with open(self.hparams.mpd_fuse_file, "w") as m:
                m.write("MPD results and stats (fuse):\n")
                self.mpd_metrics_fuse.write_stats(m)
                print(
                    "MPD results and stats (fuse) written to file",
                    self.hparams.mpd_fuse_file,
                )

class HMA_attn_ctc_to_mispro_ver2_1(HMA_attn_ctc_to_mispro_ver2):
    """ Change MHA to TransformerDecoder"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.FUSE_Net = MyTransformerDecoder(
            decoder_layer=MyTransformerDecoderLayer(
                d_model=384, nhead=8, dim_feedforward=1024, dropout=0.1, activation="relu"
            ),
            num_layers=4,
        ).to(self.device)
        self.modules_to_train = self.modules_to_train + [self.FUSE_Net]

        self.guided_attn_loss = GuidedAttentionLoss(sigma=0.2)

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
        
        # apply attn mask as we know x is padded with wav_lens (rate lengths / max_s)
        # Fuse the phoneme embeddings with the acoustic features using multi-head attention
        
        attn, self_attn, cross_attn_map = self.FUSE_Net(
            tgt=x.transpose(0, 1), 
            memory=Phn_h_key.transpose(0, 1),  
            tgt_key_padding_mask=query_pad_mask,
            memory_key_padding_mask=key_padding_mask
        )  # [B, T_s, D]
        
        # memory_key_padding_mask=key_padding_mask,
        attn = attn.transpose(0, 1)  # [B, T_s, D]
        # tgt_key_padding_mask=query_pad_mask,
        # concat attn and acoustic features
        attn_concat = torch.cat([feats, attn], dim=-1)  # [B, T_s, D1 + D2]
        #m Phoneme recognition CTC head
        p_ctc_concat = self.ctc_concat_head(attn_concat)  # [B, T_s, C]
        # mispronunciation detection head
        # attn_concat = self.TransformerDecoder(attn_concat, attn_concat, tgt_key_padding_mask=query_pad_mask)  # [B, T_s, D]

        out_mispro = self.mispro_head(attn_concat)  # [B, T_s, 1]
        # sequence labeling decoder
        out_labseq = self.lab_sequence_head(attn_concat)  # [B, T_s + T_p, C]

        return {
            "p_ctc_feat": p_ctc_feat,  # [B, T_s, C]
            "p_ctc_concat": p_ctc_concat,  # [B, T_s + T_p, C]
            "out_labseq": out_labseq,  # [B, T_s + T_p, C]
            "out_mispro": out_mispro,  # [B, T_s + T_p, 1]
            "feats": feats,  # [B, T_s, D]
            "attn_map": cross_attn_map,  # [B, T_s, T_p]
        }
        
class HMA_attn_ctc_to_mispro_ver2_2(HMA_attn_ctc_to_mispro_ver2):
    """ Change MHA to TransformerDecoder"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.AcousticSubEmb = Conv2dSubsampling2(
            idim=768,  # Input dimension from SSL model
            odim=384,  # Output dimension for acoustic features
            dropout_rate=0.1,  # Dropout rate for regularization
            pos_enc=PyTorchPositionalEncoding(d_model=384, dropout=0.1),
        ).to(self.device)
        
        # LayerNorm after subsampling
        self.acoustic_layer_norm = torch.nn.LayerNorm(384).to(self.device)
        
        self.PhnTransformerEnc = torch.nn.TransformerEncoder(
            encoder_layer=TransformerEncoderLayer(
                d_model=384, nhead=8, dim_feedforward=1024, dropout=0.1, activation="relu"
            ),
            num_layers=4,
        ).to(self.device)
        
        # LayerNorm after phoneme encoding
        self.phoneme_layer_norm = torch.nn.LayerNorm(384).to(self.device)
        
        self.FUSE_Net = MyTransformerDecoder(
            decoder_layer=MyTransformerDecoderLayer(
                d_model=384, nhead=8, dim_feedforward=1024, dropout=0.1, activation="relu"
            ),
            num_layers=4,
        ).to(self.device)
        
        # LayerNorm after fusion
        self.fusion_layer_norm = torch.nn.LayerNorm(384).to(self.device)
        
        self.ctc_concat_head = torch.nn.Sequential(
            torch.nn.Linear(384+384, 42),  # Output dimension matches phoneme classes
            torch.nn.LogSoftmax(dim=-1)  # LogSoftmax for CTC probabilities
        )
        self.mispro_head = VanillaNN(
            input_shape=(None, None, 384+384),  # [B, T, ENC_DIM]
            dnn_neurons=1,  # Binary classification
            activation=torch.nn.Sigmoid  # Sigmoid for binary classification
        )
        self.lab_sequence_head = torch.nn.Sequential(
            torch.nn.Linear(384+384, 42),  # Output dimension matches phoneme classes
            torch.nn.LogSoftmax(dim=-1)  # LogSoftmax for CTC probabilities
        )

        self.new_modules_to_train = torch.nn.ModuleList([
            self.AcousticSubEmb,
            self.acoustic_layer_norm,
            self.PhnTransformerEnc,
            self.phoneme_layer_norm,
            self.FUSE_Net,
            self.fusion_layer_norm,
            self.ctc_concat_head,
            self.mispro_head,
            self.lab_sequence_head
        ]).to(self.device)
        
        self.modules_to_train = self.modules_to_train + self.new_modules_to_train

        self.guided_attention_loss = GuidedAttentionLoss(sigma=0.2)


    def compute_forward(self, batch, stage):
        "Given an input batch it computes the phoneme probabilities."
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        canonicals, canonical_lens = batch.phn_encoded_canonical # [B, T_p]
        perceiveds, perceived_lens = batch.phn_encoded_perceived # [B, T_p]
        # pdb.set_trace()
        
        cano_mask = make_pad_mask(canonical_lens*canonicals.size(1)).to(self.device)  # [B, T_p]
        # fp16
        feats = self.modules.perceived_ssl(wavs) # [B, T_s, ENC_DIM]
        feats_mask = make_pad_mask(wav_lens* feats.size(1)).to(self.device) # [B, T_s]
        
        # Simple NN-> CTC for segmentation and Phoneme Prediction
        x = self.AcousticEnc(feats) # [B, T_s, D]
        p_ctc_feat = self.ctc_phn_lin(x)  # [B, T_s, C]

        # Subsampling the acoustic features for cross-attention
        x_sub, _ = self.AcousticSubEmb(feats, feats_mask.unsqueeze(1))  # [B, T_s / 2-, D]?
        x_sub = self.acoustic_layer_norm(x_sub)  # Apply LayerNorm after subsampling
        x_sub_mask = make_pad_mask(wav_lens * x_sub.size(1)).to(self.device)
        
        # Phneme Embedding
        Phn_h = self.PhnEmb(canonicals) # [B, T_p, D]
        # Phn_h_key, _ = self.PhnEmb_Key(Phn_h)
        # Phn_h_value = self.PhnEmb_Value(Phn_h)  # [B, T_p, D]
        # ---- Cross-attention with canonical phoneme embeddings ----
        Phn_h = self.PhnTransformerEnc(Phn_h)  # [B, T_p, D]
        Phn_h = self.phoneme_layer_norm(Phn_h)  # Apply LayerNorm after phoneme encoding
        
        # x_sub and Phn_h for TransformerDecoder
        
        attn, self_attn, cross_attn = self.FUSE_Net(
            tgt=x_sub.transpose(0, 1),
            memory=Phn_h.transpose(0, 1),
            tgt_key_padding_mask=x_sub_mask,
            memory_key_padding_mask=cano_mask
        )

        attn = attn.transpose(0, 1)  # [B, T_s / 2 -, D]
        attn = self.fusion_layer_norm(attn)  # Apply LayerNorm after fusion
        
        attn_concat = torch.cat([x_sub, attn], dim=-1)  # [B, T_s / 2 , D1 + D2]
        
        p_ctc_concat = self.ctc_concat_head(attn_concat)  # [B, T_s / 2, C]
        # mispronunciation detection head
        # attn_concat = self.TransformerDecoder(attn_concat, attn_concat, tgt_key_padding_mask=query_pad_mask)  # [B, T_s / 2, D]

        out_mispro = self.mispro_head(attn_concat)  # [B, T_s / 2, 1]
        # sequence labeling decoder
        out_labseq = self.lab_sequence_head(attn_concat)  # [B, T_s / 2 + T_p, C]

        return {
            "p_ctc_feat": p_ctc_feat,  # [B, T_s, C]
            "p_ctc_concat": p_ctc_concat,  # [B, T_s + T_p, C]
            "out_labseq": out_labseq,  # [B, T_s + T_p, C]
            "out_mispro": out_mispro,  # [B, T_s + T_p, 1]
            "feats": feats,  # [B, T_s, D]
            "attn_map": cross_attn,  # [B, T_s / 2 -, T_p]
        }
        
class HMA_attn_ctc_to_mispro_ver2_1_perceived(HMA_attn_ctc_to_mispro_ver2_1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.FUSE_Net = MyTransformerDecoder(
            decoder_layer=MyTransformerDecoderLayer(
                d_model=384, nhead=8, dim_feedforward=1024, dropout=0.1, activation="relu"
            ),
            num_layers=4,
        ).to(self.device)
        self.modules_to_train = self.modules_to_train + [self.FUSE_Net]

        self.guided_attn_loss = GuidedAttentionLoss(sigma=0.2)
    """
    Change Canonical Embedding to Perceived Embedding
    """
    def compute_forward(self, batch, stage):
        "Given an input batch it computes the phoneme probabilities."
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        canonicals, canonical_lens = batch.phn_encoded_canonical # [B, T_p]
        perceiveds, perceived_lens = batch.phn_encoded_perceived # [B, T_p]
        feats = self.modules.perceived_ssl(wavs) # [B, T_s, ENC_DIM]
        x = self.AcousticEnc(feats) # [B, T_s, D]
        p_ctc_feat = self.ctc_phn_lin(x)  # [B, T_s, C]

        Phn_h = self.PhnEmb(perceiveds) # [B, T_p, D]
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
        
        # apply attn mask as we know x is padded with wav_lens (rate lengths / max_s)
        # Fuse the phoneme embeddings with the acoustic features using multi-head attention
        
        attn, self_attn, cross_attn_map = self.FUSE_Net(
            tgt=x.transpose(0, 1), 
            memory=Phn_h_key.transpose(0, 1),  
            tgt_key_padding_mask=query_pad_mask,
            memory_key_padding_mask=key_padding_mask
        )  # [B, T_s, D]
        
        # memory_key_padding_mask=key_padding_mask,
        attn = attn.transpose(0, 1)  # [B, T_s, D]
        
        # tgt_key_padding_mask=query_pad_mask,
        # concat attn and acoustic features
        attn_concat = torch.cat([feats, attn], dim=-1)  # [B, T_s, D1 + D2]
        #m Phoneme recognition CTC head
        p_ctc_concat = self.ctc_concat_head(attn_concat)  # [B, T_s, C]
        # mispronunciation detection head
        # attn_concat = self.TransformerDecoder(attn_concat, attn_concat, tgt_key_padding_mask=query_pad_mask)  # [B, T_s, D]

        out_mispro = self.mispro_head(attn_concat)  # [B, T_s, 1]
        # sequence labeling decoder
        out_labseq = self.lab_sequence_head(attn_concat)  # [B, T_s + T_p, C]

        return {
            "p_ctc_feat": p_ctc_feat,  # [B, T_s, C]
            "p_ctc_concat": p_ctc_concat,  # [B, T_s + T_p, C]
            "out_labseq": out_labseq,  # [B, T_s + T_p, C]
            "out_mispro": out_mispro,  # [B, T_s + T_p, 1]
            "feats": feats,  # [B, T_s, D]
            "attn_map": cross_attn_map,  # [B, T_s, T_p]
        }

class Hybrid_CTC_Attention(HMA_attn_ctc_to_mispro_ver2_1):
    """Hybrid CTC and Attention-based model for phoneme recognition with Joint Decoding."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.label_encoder.add_label("<eos>")
        self.guided_attn_loss = GuidedAttentionLoss(sigma=0.2)
        
        self.AcousticEnc = torch.nn.Sequential(
            torch.nn.Linear(768, 384),  # Input dimension from SSL model
            torch.nn.LeakyReLU(),  # Activation function
            PyTorchPositionalEncoding(d_model=384, dropout=0.1),
        ).to(self.device)
        
        # ).to(self.device)
        self.ctc_phn_lin = torch.nn.Linear(384, 44)  # CTC head for phoneme recognition
        self.AcousticTransEnc = TransformerEncoder(
            num_layers=4,
            nhead=4,
            d_model=384,
            d_ffn=1024,
            dropout=0.1,
        ).to(self.device)
        
        self.PhnEmb_Expd = torch.nn.Sequential(
            torch.nn.Embedding(42+2, 384),  # 42 phonemes, 384 dim
            PyTorchPositionalEncoding(d_model=384, max_len=1000),  # Positional encoding
        ).to(self.device)
        
        # self.TransDec = MyTransformerDecoder(
        #     decoder_layer=MyTransformerDecoderLayer(
        #         d_model=384, nhead=8, dim_feedforward=1024, dropout=0.1, activation="relu"
        #     ),
        #     num_layers=4,
        # ).to(self.device)
        self.TransDec = TransformerDecoder(
            num_layers=4,
            nhead=4,
            d_ffn=1024,
            d_model=384,
            dropout=0.1,
        ).to(self.device)
        
        self.PostNet = torch.nn.Sequential(
            torch.nn.Linear(384, 384),  # Input dimension from Transformer Decoder
            torch.nn.LeakyReLU(),  # Activation function
            torch.nn.Linear(384, 42+2),  # Output dimension matches phoneme classes
        ).to(self.device)
        
        
        # VALID Searcher AR
        self.valid_searcher = S2STransformerGreedySearcher(
            modules=[self.TransDec, self.PostNet],
            bos_index=42,  # Start of sequence token
            eos_index=43,  # End of sequence token
            min_decode_ratio=0.0,  # Minimum decoding ratio
            max_decode_ratio=1.0,  # Maximum decoding ratio
        )
        self.test_searcher = S2STransformerGreedySearcher(
            modules=[self.TransDec, self.PostNet],
            bos_index=42, 
            eos_index=43,
            min_decode_ratio=0.0,  # Minimum decoding ratio
            max_decode_ratio=1.0,  # Maximum decoding ratio
        )
        # self.AcousticTransEnc = nn.TransformerEncoder(
        #     encoder_layer=nn.TransformerEncoderLayer(
        #         d_model=384, nhead=8, dim_feedforward=1024, dropout=0.1, activation="relu"
        #     ),
        #     num_layers=4,
        # self.TransDec = nn.TransformerDecoder(
        #     num_layers=4,
        #     nhead=4,
        #     d_ffn=1024,
        #     d_model=384,
        #     dropout=0.1,
        # ).to(self.device)
        # torch.nn.LogSoftmax(dim=-1)  # LogSoftmax for CTC probabilities

        # self.ARdecoder = S2STransformerBeamSearcher(
        #     modules=[self.AcousticTransEnc, self.PostNet],
        #     bos_index=self.label_encoder.get_bos_index(),  # Start of sequence token
        #     eos_index=self.label_encoder.get_eos_index(),  # End of sequence token
        #     min_decode_ratio=0.5,  # Minimum decoding ratio
        #     max_decode_ratio=1.1,  # Maximum decoding ratio
        #     beam_size=7
        # )
        # self.ARDecoder = S2STransformerGreedySearcher(
        #     bos_index=42,  
        #     eos_index=43, 
        #     min_decode_ratio=0.5,  
        #     max_decode_ratio=1.1,  
        # )
            # decoder=self.TransDec,
            # max_len_ratio=getattr(self.hparams, 'max_len_ratio', 1.0),
            # sos_id=getattr(self.label_encoder, "bos_label", 42),  # Start of sequence token
            # eos_id=getattr(self.label_encoder, "eos_label", 43),  # End of sequence token
        # Initialize Joint CTC-Attention Decoder
        # self.joint_decoder = JointCTCAttentionDecoder(
        #     blank_id=getattr(self.hparams, 'blank_index', 0),
        #     ctc_weight=getattr(self.hparams, 'ctc_weight', 0.3),
        #     att_weight=getattr(self.hparams, 'att_weight', 0.7),
        #     beam_size=getattr(self.hparams, 'beam_size', 10),
        #     max_len_ratio=getattr(self.hparams, 'max_len_ratio', 1.0),
        # )
        
        # # Custom ARDecoder based on S2SBaseSearcher
        # class ARDecoder(sb.decoders.seq2seq.S2SBaseSearcher):
        #     def __init__(self, modules, phoneme_embedding, bos_index=42, eos_index=43, 
        #                 min_decode_ratio=0.0, max_decode_ratio=1.0, device=None):
        #         super().__init__(
        #             bos_index=bos_index,
        #             eos_index=eos_index,
        #             min_decode_ratio=min_decode_ratio,
        #             max_decode_ratio=max_decode_ratio,
        #         )
        #         self.decoder = modules[0]  # TransDec
        #         self.output_layer = modules[1]  # PostNet
        #         self.phoneme_embedding = phoneme_embedding  # PhnEmb_Expd
        #         self.device = device
                
        #     def reset_mem(self, batch_size, device):
        #         """Reset memory for new sequence."""
        #         return None
                
        #     def permute_mem(self, memory, index):
        #         """Permute memory according to index."""
        #         return memory
                
        #     def forward_step(self, inp_tokens, memory, enc_states, enc_lens):
        #         """Forward step for one decoding step.
                
        #         Args:
        #             inp_tokens: [batch_size] - current input tokens
        #             memory: memory from previous steps
        #             enc_states: [seq_len, batch_size, feature_dim] - encoder states
        #             enc_lens: [batch_size] - encoder sequence lengths
                    
        #         Returns:
        #             log_probs: [batch_size, vocab_size] - log probabilities
        #             memory: updated memory
        #         """
        #         batch_size = inp_tokens.shape[0]
                
        #         # Handle first step differently - inp_tokens might be just BOS
        #         if inp_tokens.dim() == 1:
        #             # First step: inp_tokens is [batch_size] with BOS tokens
        #             current_len = 1
        #             sequence = inp_tokens.unsqueeze(1)  # [batch_size, 1]
        #         else:
        #             # Subsequent steps: inp_tokens is [batch_size, seq_len]
        #             current_len = inp_tokens.shape[1]
        #             sequence = inp_tokens
                
        #         # Embed the sequence
        #         embedded = self.phoneme_embedding(sequence)  # [batch_size, seq_len, d_model]
                
        #         # Create causal mask for current sequence
        #         tgt_mask = self._generate_square_subsequent_mask(current_len).to(embedded.device)
                
        #         # Create padding mask for target (all valid since we're generating)
        #         tgt_key_padding_mask = torch.zeros(batch_size, current_len, dtype=torch.bool, device=embedded.device)
                
        #         # Create encoder padding mask
        #         max_enc_len = enc_states.shape[0]
        #         enc_key_padding_mask = torch.arange(max_enc_len, device=embedded.device).unsqueeze(0) >= (enc_lens * max_enc_len).unsqueeze(1)
                
        #         # Run decoder
        #         decoder_out, _, attn_weights = self.decoder(
        #             tgt=embedded,  # [batch_size, seq_len, d_model]
        #             memory=enc_states.transpose(0, 1),  # [batch_size, enc_len, d_model]
        #             tgt_key_padding_mask=tgt_key_padding_mask,
        #             memory_key_padding_mask=enc_key_padding_mask,
        #             tgt_mask=tgt_mask,
        #         )
                
        #         # Get output for the last position
        #         last_output = decoder_out[:, -1, :]  # [batch_size, d_model]
                
        #         # Apply output layer
        #         log_probs = self.output_layer(last_output)  # [batch_size, vocab_size]
        #         # log_probs = torch.log_softmax(logits, dim=-1)
                
        #         return log_probs, memory, attn_weights
                
        #     def _generate_square_subsequent_mask(self, sz):
        #         """Generate causal mask."""
        #         mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        #         return mask
        #     def forward(self, enc_states, enc_lens):
        #         """Main forward method for autoregressive decoding.
                
        #         Args:
        #             enc_states: [seq_len, batch_size, feature_dim] - encoder states
        #             enc_lens: [batch_size] - encoder sequence lengths
        #         Returns:
        #             hyps: List of decoded hypotheses for each batch item
        #             scores: List of scores for each hypothesis
        #             lengths: List of lengths of each hypothesis
        #             attn: Attention weights for each hypothesis
        #         """
        #         pdb.set_trace()
        #         batch_size = enc_states.size(1)
        #         max_len = int(enc_lens.max().item() * self.max_decode_ratio)
        #         hyps = [[] for _ in range(batch_size)]
        #         scores = [0.0] * batch_size
        #         lengths = [0] * batch_size  
        #         attn = [None] * batch_size  # Store attention weights
        #         memory = self.reset_mem(batch_size, self.device)
        #         inp_tokens = torch.full((batch_size,), self.bos_index, dtype=torch.long, device=self.device)  # Start with BOS
        #         for step in range(max_len):
        #             # Forward step
        #             log_probs, memory, attn_weights = self.forward_step(inp_tokens, memory, enc_states, enc_lens)

        #             # Get top predictions
        #             top_probs, top_indices = torch.topk(log_probs, k=1, dim=-1)
        #             top_indices = top_indices.squeeze(-1)  # [batch_size]
        #             top_probs = top_probs.squeeze(-1)  # [batch_size]
        #             # Update hypotheses
        #             for b in range(batch_size):
        #                 if top_indices[b].item() == self.eos_index:
        #                     # If EOS token, finalize this hypothesis
        #                     if lengths[b] > 0:
        #                         hyps[b].append(top_indices[b].item())
        #                         scores[b] += top_probs[b].item()
        #                         lengths[b] += 1
        #                     continue
        #                 # Otherwise, continue building the hypothesis
        #                 hyps[b].append(top_indices[b].item())
        #                 scores[b] += top_probs[b].item()
        #                 lengths[b] += 1
        #             # Prepare for next step
        #             inp_tokens = top_indices  # Use the top indices as input for the next step
        #             # Store attention weights for the last ste
        #             if attn is not None:
        #                 attn = [attn_weights[b].detach() for b in range(batch_size)]
        #         # Convert hypotheses to final format
        #         for b in range(batch_size):
        #             if len(hyps[b]) == 0 or hyps[b][-1] != self.eos_index:
        #                 hyps[b].append(self.eos_index)
        #             # Ensure all hypotheses are of the same length
        #             if len(hyps[b]) < max_len:
        #                 hyps[b] += [self.eos_index] * (max_len - len(hyps[b]))
        #         # Convert to tensor format
        #         hyps_tensor = torch.tensor(hyps, dtype=torch.long, device=self.device)
        #         # Convert scores and lengths to tensors
        #         scores_tensor = torch.tensor(scores, dtype=torch.float, device=self.device)
        #         lengths_tensor = torch.tensor(lengths, dtype=torch.long, device=self.device)
        #         pdb.set_trace()
        #         return hyps_tensor, scores_tensor, lengths_tensor, attn  # Return hypotheses, scores, lengths, and attention weights
            
        # # Initialize the custom ARDecoder
        # self.ARDecoder = ARDecoder(
        #     modules=[self.TransDec, self.PostNet],
        #     phoneme_embedding=self.PhnEmb_Expd,
        #     bos_index=42,
        #     eos_index=43, 
        #     min_decode_ratio=0.5,
        #     max_decode_ratio=1.1,
        #     device=self.device
        # )
        
        # # Also keep the SpeechBrain greedy searcher as backup
        # self.attn_decoder = sb.decoders.seq2seq.S2SGreedySearcher(
        #     bos_index=42,  
        #     eos_index=43,  
        #     min_decode_ratio=0.5, 
        #     max_decode_ratio=1.1, 
        # )
        # add to modules_to_train
        self.modules_to_train = torch.nn.ModuleList([
            self.ctc_phn_lin,
            self.PhnEmb_Expd,
            self.AcousticEnc,
            self.AcousticTransEnc,
            self.TransDec,
            self.PostNet
        ]).to(self.device)
            #self.ARDecoder

        
    def compute_forward(self, batch, stage):
        "Given an input batch it computes the phoneme probabilities."
        current_epoch = self.hparams.epoch_counter.current
        
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        canonicals, canonical_lens = batch.phn_encoded_canonical  # [B, T_p]
        perceiveds, perceived_lens = batch.phn_encoded_perceived  # [B, T_p]
        
        targets, target_lens = batch.phn_encoded_target  # [B, T_p]
        targets_expd, target_lens_expd = batch.phn_encoded_target_expd  # [B, T_p+1]
        
        feats = self.modules.perceived_ssl(wavs)  # [B, T_s, ENC_DIM]
        
        x = self.AcousticEnc(feats)  # [B, T_s, D]
        # Transformer Encoder
        h_s, enc_attn_list = self.AcousticTransEnc(x) # [B, T_s, D], [n_layers, B, T_s, T_s] 
        # h_s = h_s.transpose(0, 1) 
        # simple CTC for decoding
        p_ctc_logit = self.ctc_phn_lin(h_s)  # [B, T_s, C]
        p_ctc_feat = torch.log_softmax(p_ctc_logit, dim=-1)  # [B, T_s, C]

        h_s_mask = make_pad_mask(wav_lens * h_s.size(1), maxlen=h_s.size(1)).to(self.device)  # [B, T_s] 
        
        hyps = None
        Phn_h = self.PhnEmb_Expd(targets_expd)  # [B, T_p, D]

        targets_expd_mask = make_pad_mask(target_lens_expd * targets_expd.size(1), maxlen=targets_expd.size(1)).to(self.device)  # [B, T_p]
        if stage == sb.Stage.TRAIN:
            # Training: use teacher forcing with target sequences
            # Add EOS token to targets for proper sequence-to-sequence training
            # Phoneme Embedding for targets (teacher forcing)

            # pdb.set_trace()
            d_out, _, attn_map_list = self.TransDec(
                tgt=Phn_h,  
                memory=h_s,  
                tgt_key_padding_mask=targets_expd_mask,
                memory_key_padding_mask=h_s_mask,
            ) 
            
            d_out = d_out
            # d_out: [T_p, B, D]
            # attn_map: [B, n_layers, T_p, T_s]
            attn_map = attn_map_list[-1]  # Get the last layer attention map
            # get last layer attn_map
            p_dec_logit = self.PostNet(d_out)  # [B, T_p, C+2]
            p_dec_out = torch.log_softmax(p_dec_logit, dim=-1)  # [B, T_p, C+2]
        elif stage == sb.Stage.VALID:
            # if current_epoch % 10 == 0:
            pdb.set_trace()
            hyps, top_lengths, top_scores, top_log_probs = self.valid_searcher(h_s, wav_lens)
            pdb.set_trace()
            # pdb.set_trace()
            p_dec_out = top_log_probs[0]
            attn_map = None
            # else:
            #     with torch.no_grad():
            #         d_out, _, attn_map_list = self.TransDec(
            #             tgt=Phn_h,  # [B, T_p, D]
            #             memory=h_s,  # [B, T_s, D]
            #             tgt_key_padding_mask=targets_expd_mask,
            #             memory_key_padding_mask=h_s_mask,
            #         )
            #         d_out = d_out
            #         attn_map = attn_map_list[-1]  # Get the last layer attention map
            #         p_dec_logit = self.PostNet(d_out)  # [B, T_p, C+2]
            #         p_dec_out = torch.log_softmax(p_dec_logit, dim=-1)
            
        elif stage == sb.Stage.TEST: 
            hyps, top_lengths, top_scores, top_log_probs = self.test_searcher(enc_state=h_s, enc_lens=wav_lens)
            
            p_dec_out = top_log_probs[0]
            attn_map = None

        return {
            "p_ctc_feat": p_ctc_feat,  # [B, T_s, C]
            "p_dec_out": p_dec_out,  # [B, T_s + T_p, C]
            # "out_labseq": out_labseq,  # [B, T_s + T_p, C]
            # "out_mispro": out_mispro,  # [B, T_s + T_p, 1]
            # "feats": feats,  # [B, T_s, D]
            "hyps": hyps,  # [B, T_s + T_p]
            "attn_map": attn_map,  # [B, T_s + T_p, D]
            
        }
        
    def compute_objectives(self, predictions, batch, stage):
        "Computes the loss for the model."
        p_ctc_feat = predictions["p_ctc_feat"]
        p_dec_out = predictions["p_dec_out"]
        # out_labseq = predictions["out_labseq"]
        # out_mispro = predictions["out_mispro"]
        # feats = predictions["feats"]
        hyps = predictions["hyps"]  # [B, T_s + T_p]
        attn_map = predictions["attn_map"]

        wavs, wav_lens = batch.sig
        targets, target_lens = batch.phn_encoded_target
        targets_expd, target_lens_expd = batch.phn_encoded_target_expd
        canonicals, canonical_lens = batch.phn_encoded_canonical
        canonicals_expd, canonical_lens_expd = batch.phn_encoded_canonical_expd
        perceiveds, perceived_lens = batch.phn_encoded_perceived
        perceiveds_expd, perceived_lens_expd = batch.phn_encoded_perceived_expd
        ids = batch.id  # [B]
        
        target_mask = make_pad_mask(target_lens * targets.size(1)).to(self.device)  # [B, T_p]
        # h_s_mask = make_pad_mask(wav_lens * feats.size(1)).to(self.device)  # [B, T_s]
    
        # Compute CTC loss
        loss_ctc = self.hparams.ctc_cost(p_ctc_feat, targets, wav_lens, target_lens)
        
        # import pdb; pdb.set_trace()
        # Compute attention out loss only during training
        if stage == sb.Stage.TRAIN:
            # During training, we have proper target alignment with EOS
            # Re-create targets with EOS for loss calculation
            
            CELoss = torch.nn.CrossEntropyLoss(
                ignore_index=self.hparams.blank_index,
                reduction="mean"
            )
            # Ensure dimensions match for loss calculation
            if p_dec_out.size(1) != targets_expd.size(1):
                # Truncate or pad to match target length
                target_len = targets_expd.size(1)
                if p_dec_out.size(1) > target_len:
                    p_dec_out = p_dec_out[:, :target_len, :]
                else:
                    # Pad with blank predictions
                    pad_len = target_len - p_dec_out.size(1)
                    pad_logits = torch.full((p_dec_out.size(0), pad_len, p_dec_out.size(2)), 
                                          -float('inf'), device=p_dec_out.device)
                    pad_logits[:, :, self.hparams.blank_index] = 0  # Set blank to 0 (log prob 1)
                    p_dec_out = torch.cat([p_dec_out, pad_logits], dim=1)
            
            # TODO: Better to Train CELoss without BOS / EOS
            loss_dec_out = CELoss(
                p_dec_out.view(-1, p_dec_out.size(-1)),
                targets_expd.view(-1),
            )
        else:
            # During inference, don't compute attention loss
            loss_dec_out = torch.tensor(0.0, device=self.device)
        
        # ---- Guided Attention Loss ----
        if stage == sb.Stage.TRAIN:
            # Only compute guided attention loss during training
            # attn_map: [B,H,T_q,T_k] or [B,T_q,T_k]
            if attn_map.dim() == 4:
                A = attn_map.mean(dim=1)          # average heads -> [B,T_q,T_k]
            elif attn_map.dim() == 3:
                A = attn_map
            else:
                raise ValueError(f"Unexpected attn_map shape {attn_map.shape}")
            
            # GuidedAttentionLoss expects (B, targets, inputs).
            # 我们把"targets"当作 canonical tokens (K)， "inputs"当作 acoustic frames (Q)。
            Bsz, T_p_max, T_s_max = A.shape  
            # 语音帧长度
            in_lens_abs = (wav_lens * T_s_max).round().long().clamp(1, T_s_max)
            # phone token 长度
            target_lens_expd_abs = (target_lens_expd * T_p_max).round().long().clamp(1, T_p_max)
            # pdb.set_trace()
            loss_ga = self.guided_attn_loss(
                A, input_lengths=in_lens_abs, target_lengths=target_lens_expd_abs
            )
            # pdb.set_trace()
        else:
            # During inference, don't compute guided attention loss
            loss_ga = torch.tensor(0.0, device=self.device)
        # guide attention loss
        loss = loss_ctc + loss_dec_out + loss_ga

        # Record losses for posterity
        if stage == sb.Stage.VALID:
            # Traditional CTC greedy decoding
            sequence = sb.decoders.ctc_greedy_decode(
                p_ctc_feat, wav_lens, blank_id=self.hparams.blank_index
            )
            print(f"CTC Greedy Decoding: {sequence}")
            # Attention decoder greedy decoding
            sequence_decoder_out = torch.argmax(p_dec_out, dim=-1)  # [B, T_p]
            # replace <eos> token with blank token
            # Tobe be Done
            print(f"Attention Decoder Greedy Decoding: {sequence_decoder_out}")
            sequence_decoder_out = sequence_decoder_out.masked_fill(
                sequence_decoder_out == 42, 
                self.hparams.blank_index
            )
            
            sequence_decoder_out = sequence_decoder_out.masked_fill(
                sequence_decoder_out == 43, 
                self.hparams.blank_index
            )
            # remove eos from joint_sequences_greedy
            
            # Optional: Joint beam search decoding (more computationally expensive)
            # try:
            #     joint_sequences_beam = self.joint_decoder.joint_beam_search(
            #         ctc_log_probs=p_ctc_feat,
            #         att_log_probs=p_dec_out,
            #         input_lengths=wav_lens,
            #         decoder_model=self.TransDec,
            #         memory=feats.transpose(0, 1), 
            #         memory_mask=h_s_mask
            #     )
            # except Exception as e:
            #     print(f"Beam search failed: {e}, using greedy joint decoding")
            
            import matplotlib.pyplot as plt
            attn_dir = os.path.join(self.hparams.output_folder, "attention_maps")
            if not os.path.exists(attn_dir):
                os.makedirs(attn_dir, exist_ok=True)
            epoch = self.hparams.epoch_counter.current
            if epoch %10 == 0 or epoch==1:
                # We visualize the first ele batch only
                # Note: If  attn_map is [B, T_s, T_p], we visualize the first element
                # Elese if attn_map is [B, n_heads, T_s, T_p], we visualize the all head as subplots
                if attn_map.ndim == 3:
                    for attn_id, attn in enumerate(attn_map[0: 1]):
                        # get valid target length and wav length from batch[0]
                        wav_len_b = int(wav_lens[attn_id] * attn.size(1))
                        target_len_b = int(target_lens[attn_id] * targets.size(1))
                        attn = attn[:wav_len_b, :target_len_b]  # Crop
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
                elif attn_map.ndim == 4:
                    for attn_id, attn in enumerate(attn_map[0: 1]):
                        # get valid target length and wav length from batch[0]
                        wav_len_b = int(wav_lens[attn_id] * attn.size(1))
                        target_len_b = int(target_lens[attn_id] * targets.size(1))
                        # attn = attn[:wav_len_b, :target_len_b]  # Crop
                        # draw the bpunding box on the attention map with wav_len_b and target_len_b
                        
                        
                        n_heads = attn.size(0)
                        fig, axs = plt.subplots(n_heads//2, 2, figsize=(5, 10))
                        fig.suptitle(f"Attention Map for ID {ids[attn_id]} - Epoch {epoch}")
                        if n_heads == 1:
                            axs = [axs]
                        for i in range(n_heads//2):
                            for j in range(2):
                                axs[i, j].imshow(attn[i*2+j].cpu().detach().numpy(), aspect='auto', origin='lower')
                                axs[i, j].set_xlabel("Canonical Phoneme Index")
                                axs[i, j].set_ylabel("Acoustic Feature Index")
                        plt.tight_layout()
                        attn_file = os.path.join(attn_dir, f"{ids[attn_id].split('/')[-1]}_epoch{epoch}.png")
                        plt.savefig(attn_file)
                        plt.close()
                        print(f"Saved attention map to {attn_file}")

            self.ctc_metrics.append(ids, p_ctc_feat, targets, wav_lens, target_lens)
            # self.ctc_metrics_fuse.append(ids, sequence_decoder_out, targets, wav_lens, target_lens)
            # CTC-only results
            self.per_metrics.append(
                ids=ids,
                predict=sequence,
                target=targets,
                predict_len=None,
                target_len=target_lens,
                ind2lab=self.label_encoder.decode_ndim,
            )
            
            # Attention-only results
            self.per_metrics_fuse.append(
                ids=ids,
                predict=sequence_decoder_out,
                target=targets,
                predict_len=None,
                target_len=target_lens,
                ind2lab=self.label_encoder.decode_ndim,
            )
            
            
            # MPD metrics
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
                predict=sequence_decoder_out,
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
                }, step=self.hparams.epoch_counter.current)
                if loss_ga is not None:
                    wandb.log({"loss_ga": loss_ga.item()}, step=self.hparams.epoch_counter.current)
                if loss_dec_out is not None:
                    wandb.log({"loss_dec_out": loss_dec_out.item()}, step=self.hparams.epoch_counter.current)
                if loss_ctc is not None:
                    wandb.log({"loss_ctc_head": loss_ctc.item()}, step=self.hparams.epoch_counter.current)
            except Exception:
                pass

        return loss
    
    def on_stage_start(self, stage, epoch):
        """Gets called when a stage starts. Initialize metrics including joint decoding metrics."""
        super().on_stage_start(stage, epoch)
        
        # Initialize joint decoding metrics
        if stage != sb.Stage.TRAIN:
            self.per_metrics_joint = self.hparams.per_stats()
            self.per_metrics_joint_beam = self.hparams.per_stats()
            self.mpd_metrics_joint = MpdStats()
            self.mpd_metrics_joint_beam = MpdStats()
    
    def on_stage_end(self, stage, stage_loss, epoch):
        current_stage = self.hparams.epoch_counter.current
        """Gets called at the end of a epoch."""
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        else:
            if current_stage % 5 == 0:
                per = self.per_metrics.summarize("error_rate")
                mpd_f1 = self.mpd_metrics.summarize("mpd_f1")
                
                # Get joint decoding results if available
                # per_joint_greedy = self.per_metrics_joint.summarize("error_rate") if hasattr(self, 'per_metrics_joint') else per
                # per_joint_beam = self.per_metrics_joint_beam.summarize("error_rate") if hasattr(self, 'per_metrics_joint_beam') else per
                # mpd_f1_joint_greedy = self.mpd_metrics_joint.summarize("mpd_f1") if hasattr(self, 'mpd_metrics_joint') else mpd_f1
                # mpd_f1_joint_beam = self.mpd_metrics_joint_beam.summarize("mpd_f1") if hasattr(self, 'mpd_metrics_joint_beam') else mpd_f1

        if stage == sb.Stage.VALID and current_stage % 5 == 0:
            # Log stats
            valid_stats = {
                "loss": stage_loss,
                "ctc_loss": self.ctc_metrics.summarize("average"),
                "PER": per,
                "mpd_f1": mpd_f1,
                # "PER_joint_greedy": per_joint_greedy,
                # "PER_joint_beam": per_joint_beam,
                # "mpd_f1_joint_greedy": mpd_f1_joint_greedy,
                # "mpd_f1_joint_beam": mpd_f1_joint_beam,
            }
            
            self.hparams.train_logger.log_stats(
                stats_meta={
                    "epoch": epoch,
                    "lr_adam": self.adam_optimizer.param_groups[0]["lr"],
                    "lr_pretrained": self.pretrained_opt_class.param_groups[0]["lr"],
                },
                train_stats={"loss": self.train_loss},
                valid_stats=valid_stats,
            )
            # Save best 3 Models - prioritize joint beam search results
            improved = False
            
            # Use best joint decoding result for model selection
            best_per = min(per)
            best_mpd_f1 = max(mpd_f1)
            
            # Save best 3 PER models (lower is better)
            if best_per < self.best_per or len(self.best_per_list) < 3:
                ckpt_name = f"best_per_joint_{epoch:03d}_{best_per:.4f}.ckpt"
                self.checkpointer.save_and_keep_only(
                    meta={
                        "PER": per, 
                        "best_PER": best_per,
                        "mpd_f1": mpd_f1, 
                        "epoch": epoch
                    },
                    name=ckpt_name,
                    num_to_keep=5,
                    min_keys=["best_PER"]
                )
                self.best_per_list.append((best_per, epoch, ckpt_name))
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
            if best_mpd_f1 > self.best_mpd_f1 or len(self.best_mpd_f1_list) < 3:
                ckpt_name = f"best_mpdf1_joint_{epoch:03d}_{best_mpd_f1:.4f}.ckpt"
                self.checkpointer.save_and_keep_only(
                    meta={
                        "PER": per, 
                        "mpd_f1": mpd_f1,
                        "best_mpd_f1": best_mpd_f1,
                        "epoch": epoch
                    },
                    name=ckpt_name,
                    num_to_keep=5,
                    max_keys=["best_mpd_f1"]
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
            # if not files for joint decoding, create files
            if hasattr(self, 'per_metrics_joint'):
                self.per_metrics_joint = self.hparams.per_file.replace(".txt", "_joint.txt")
                with open(self.per_metrics_joint, "w") as w:
                    w.write("Joint CTC-Attention PER stats:\n")
                    self.per_metrics_joint.write_stats(w)
                    print(
                        "Joint CTC-Attention PER stats written to file",
                        self.per_metrics_joint,
                    )
            if hasattr(self, 'mpd_metrics_joint'):
                self.mpd_metrics_joint = self.hparams.mpd_file.replace(".txt", "_joint.txt")
                with open(self.mpd_metrics_joint, "w") as m:
                    m.write("Joint CTC-Attention MPD results and stats:\n")
                    self.mpd_metrics_joint.write_stats(m)
                    print(
                        "Joint CTC-Attention MPD results and stats written to file",
                        self.mpd_metrics_joint,
                    )
            # joint beam search results
            if hasattr(self, 'per_metrics_joint_beam'):
                self.per_metrics_joint_beam = self.hparams.per_file.replace(".txt", "_joint_beam.txt")
                with open(self.per_metrics_joint_beam, "w") as w:
                    w.write("Joint CTC-Attention PER stats (beam search):\n")
                    self.per_metrics_joint_beam.write_stats(w)
                    print(
                        "Joint CTC-Attention PER stats (beam search) written to file",
                        self.per_metrics_joint_beam,
                    )
            if hasattr(self, 'mpd_metrics_joint_beam'):
                self.mpd_metrics_joint_beam = self.hparams.mpd_file.replace(".txt", "_joint_beam.txt")
                with open(self.mpd_metrics_joint_beam, "w") as m:
                    m.write("Joint CTC-Attention MPD results and stats (beam search):\n")
                    self.mpd_metrics_joint_beam.write_stats(m)
                    print(
                        "Joint CTC-Attention MPD results and stats (beam search) written to file",
                        self.mpd_metrics_joint_beam,
                    )
                    
    def init_optimizers(self):
        self.adam_optimizer = self.hparams.adam_opt_class(
            self.modules_to_train.parameters(),
        )
        
        self.pretrained_opt_class = self.hparams.pretrained_opt_class(
            self.modules.perceived_ssl.parameters(), 
        )
        if self.checkpointer is not None:
            # if self.hparams.perceived_ssl is not None and not self.hparams.perceived_ssl.freeze:
            self.checkpointer.add_recoverable("adam_opt", self.adam_optimizer)
            self.checkpointer.add_recoverable("pretrained_opt", self.pretrained_opt_class)
class Hybrid_CTC_Attention_ver2(Hybrid_CTC_Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.label_encoder.add_label("<eos>")
        self.guided_attn_loss = GuidedAttentionLoss(sigma=0.2)
        
        self.TransASR = TransformerASR(
            tgt_vocab=44,
            input_size=384,
            d_model=384,
            nhead=8,
            num_encoder_layers=6,
            d_ffn=1024,
            dropout=0.1,
            max_length=1000,
            output_hidden_states=True,
            activation=torch.nn.GELU,
        )
        for x in self.TransASR.modules():
            x.to(self.device)
            # else:c
            #
            # encoder_module="transformer",
        self.ctc_lin = torch.nn.Sequential(
            torch.nn.Linear(384, 44),  # CTC output layer
        ).to(self.device)

        self.d_out = sb.nnet.linear.Linear(
            input_shape=[None, None, 384],
            n_neurons=44, 
        ).to(self.device)

        self.ctc_scorer = CTCScorer(
            ctc_fc=self.ctc_lin,
            blank_index=0,
            eos_index=43,
        )
        self.scorer = ScorerBuilder(
            full_scorers=[self.ctc_scorer],
            weights={'ctc': 0.5}
        )
        self.ARdecoder = sb.decoders.seq2seq.S2STransformerGreedySearcher(
                modules=[self.TransASR, self.d_out],
                bos_index=42,
                eos_index=43,
                min_decode_ratio=0.1,
                max_decode_ratio=1.0,
        )
                # scorer=self.scorer,
                # beam_size=5,
        

        self.modules_to_train = torch.nn.ModuleList([x for x in self.TransASR.modules()] + [self.ctc_lin, self.d_out])
        

    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        canonicals, canonical_lens = batch.phn_encoded_canonical  # [B, T_p]
        perceiveds, perceived_lens = batch.phn_encoded_perceived  # [B, T_p]
        
        targets, target_lens = batch.phn_encoded_target  # [B, T_p]
        targets_expd, target_lens_expd = batch.phn_encoded_target_expd  # [B, T_p+1]
        feats = self.modules.perceived_ssl(wavs)  # [B, T_s, ENC_DIM]
    
        feats = self.AcousticEnc(feats)  # [B, T_s, D]
        enc_out, hidden, dec_out = self.TransASR(
            src=feats,
            tgt=targets_expd,
            wav_len=wav_lens,
            pad_idx=0,  # Assuming 0 is the padding index
        )
        
        logits = self.ctc_lin(enc_out)  # [B, T_s, C]
        p_ctc_feat = torch.nn.functional.log_softmax(logits, dim=-1)  # Log

        pred = self.d_out(dec_out)  # [B, T_p+1, C
        p_dec_out = torch.nn.functional.log_softmax(pred, dim=-1)  # Log probabilities
        
        hyps = None
        attn_map = None
        current_epoch = self.hparams.epoch_counter.current
        
        if stage != sb.Stage.TRAIN:    
            # Inference: use autoregressive decoding without target knowledge
            # hyps [B, T_p_max]
            hyps, top_lengths, top_scores, top_log_probs = self.ARdecoder(enc_out.detach(), wav_lens)
            # By modifiling greedy search we can get the attention map
            attn_map = None

        return {
            "p_ctc_feat": p_ctc_feat,  # [B, T_s, C]
            "p_dec_out": p_dec_out,  # [B, T_p+1, C]
            "feats": feats,  # [B, T_s, D]
            "attn_map": attn_map,  # [B, T_p+1, T_s] or similar
            "hyps": hyps,  # [B, T_p+1] or None if not applicable
        }
        # TODO
    def compute_objectives(self, predictions, batch, stage):
        "Computes the loss for the model."
        p_ctc_feat = predictions["p_ctc_feat"]
        p_dec_out = predictions["p_dec_out"]
        feats = predictions["feats"]
        attn_map = predictions["attn_map"]
        hyps = predictions.get("hyps", None)  # [B, T_p+1] or None if not applicable

        wavs, wav_lens = batch.sig
        targets, target_lens = batch.phn_encoded_target
        targets_expd, target_lens_expd = batch.phn_encoded_target_expd
        canonicals, canonical_lens = batch.phn_encoded_canonical
        canonicals_expd, canonical_lens_expd = batch.phn_encoded_canonical_expd
        perceiveds, perceived_lens = batch.phn_encoded_perceived
        perceiveds_expd, perceived_lens_expd = batch.phn_encoded_perceived_expd
        ids = batch.id
        
        loss_ctc = self.hparams.ctc_cost(p_ctc_feat, targets, wav_lens, target_lens)
        
        if stage == sb.Stage.TRAIN:
            # CELoss = torch.nn.CrossEntropyLoss(
            #     ignore_index=self.hparams.blank_index,
            #     reduction="mean"
            # )
            # # Ensure dimensions match for loss calculation
            # if p_dec_out.size(1) != targets_expd.size(1):
            #     # Truncate or pad to match target length
            #     target_len = targets_expd.size(1)
            #     if p_dec_out.size(1) > target_len:
            #         p_dec_out = p_dec_out[:, :target_len, :]
            #     else:
            #         # Pad with blank predictions
            #         pad_len = target_len - p_dec_out.size(1)
            #         pad_logits = torch.full((p_dec_out.size(0), pad_len, p_dec_out.size(2)), 
            #                               -float('inf'), device=p_dec_out.device)
            #         pad_logits[:, :, self.hparams.blank_index] = 0  # Set blank to 0 (log prob 1)
            #         p_dec_out = torch.cat([p_dec_out, pad_logits], dim=1)
            
            # # TODO: Better to Train CELoss without BOS / EOS
                
            loss_dec_out = sb.nnet.losses.kldiv_loss(
                p_dec_out,
                targets_expd,
                length=target_lens_expd,
                label_smoothing=0.1,
                reduction="batchmean",
            )
            
        else: 
            # During inference, don't compute attention loss
            loss_dec_out = torch.tensor(0.0, device=self.device)
        # ---- Guided Attention Loss ----
        
        loss = loss_ctc + loss_dec_out
        if stage != sb.Stage.TRAIN:
            current_epoch = self.hparams.epoch_counter.current
            if current_epoch % 1 == 0:
                # Record losses for posterit
                # Traditional CTC greedy decoding
                sequence = sb.decoders.ctc_greedy_decode(
                    p_ctc_feat, wav_lens, blank_id=self.hparams.blank_index
                )
                # Attention decoder greedy decoding
                # sequence_decoder_out = torch.argmax(p_dec_out, dim=-1)  # [B, T_p]
                sequence_decoder_out = hyps  # [B, T_p+1]
                # replace <eos> token with blank token
                print(f"GT: \n {self.label_encoder.decode_ndim(targets[-1])}")
                print(f"CTC Greedy Decoding: \n {self.label_encoder.decode_ndim(sequence[-1])}")
                print(f"Attention Decoder Greedy Decoding: \n {self.label_encoder.decode_ndim(sequence_decoder_out[-1])}")

                # trim invalid tokens
                # pdb.set_trace()
                # replace <eos> token with blank token
        
                joint_sequences_greedy = sequence_decoder_out
                # Joint CTC-Attention decoding
                # joint_sequences_greedy = self.joint_decoder.joint_greedy_decode(
                #     ctc_log_probs=p_ctc_feat,
                #     att_log_probs=p_dec_out,
                #     input_lengths=wav_lens,
                # )
                # joint_sequences_greedy = torch.Tensor(joint_sequences_greedy).to(self.device)  # [B, T_p]
                # joint_sequences_greedy = joint_sequences_greedy.masked_fill(
                #     joint_sequences_greedy == 43, 
                #     self.hparams.blank_index
                # )
                # remove eos from joint_sequences_greedy
                
                # Optional: Joint beam search decoding (more computationally expensive)
                # try:
                #     joint_sequences_beam = self.joint_decoder.joint_beam_search(
                #         ctc_log_probs=p_ctc_feat,
                #         att_log_probs=p_dec_out,
                #         input_lengths=wav_lens,
                #         decoder_model=self.TransDec,
                #         memory=feats.transpose(0, 1), 
                #         memory_mask=h_s_mask
                #     )
                # except Exception as e:
                #     print(f"Beam search failed: {e}, using greedy joint decoding")
                joint_sequences_beam = joint_sequences_greedy
                
                # import matplotlib.pyplot as plt
                # attn_dir = os.path.join(self.hparams.output_folder, "attention_maps")
                # if not os.path.exists(attn_dir):
                #     os.makedirs(attn_dir, exist_ok=True)
                # epoch = self.hparams.epoch_counter.current
                # if epoch %5 == 0 or epoch==1:
                #     # We visualize the first ele batch only
                #     # Note: If  attn_map is [B, T_s, T_p], we visualize the first element
                #     # Elese if attn_map is [B, n_heads, T_s, T_p], we visualize the all head as subplots
                #     if attn_map.ndim == 3:
                #         for attn_id, attn in enumerate(attn_map[0: 1]):
                #             # get valid target length and wav length from batch[0]
                #             wav_len_b = int(wav_lens[attn_id] * attn.size(1))
                #             target_len_b = int(target_lens[attn_id] * targets.size(1))
                #             attn = attn[:wav_len_b, :target_len_b]  # Crop
                #             plt.figure(figsize=(5, 5))
                #             plt.imshow(attn.cpu().detach().numpy(), aspect='auto', origin='lower')
                #             plt.title(f"Attention Map for ID {ids[attn_id]}")
                #             plt.xlabel("Canonical Phoneme Index")
                #             plt.ylabel("Acoustic Feature Index")
                #             plt.tight_layout()
                #             attn_file = os.path.join(attn_dir, f"{ids[attn_id].split('/')[-1]}_epoch{epoch}.png")
                #             plt.savefig(attn_file)
                #             plt.close()
                #             print(f"Saved attention map to {attn_file}")
                #     elif attn_map.ndim == 4:
                #         for attn_id, attn in enumerate(attn_map[0: 1]):
                #             # get valid target length and wav length from batch[0]
                #             wav_len_b = int(wav_lens[attn_id] * attn.size(1))
                #             target_len_b = int(target_lens[attn_id] * targets.size(1))
                #             # attn = attn[:wav_len_b, :target_len_b]  # Crop
                #             # draw the bpunding box on the attention map with wav_len_b and target_len_b
                            
                            
                #             n_heads = attn.size(0)
                #             fig, axs = plt.subplots(n_heads//2, 2, figsize=(5, 10))
                #             fig.suptitle(f"Attention Map for ID {ids[attn_id]} - Epoch {epoch}")
                #             if n_heads == 1:
                #                 axs = [axs]
                #             for i in range(n_heads//2):
                #                 for j in range(2):
                #                     axs[i, j].imshow(attn[i*2+j].cpu().detach().numpy(), aspect='auto', origin='lower')
                #                     axs[i, j].set_xlabel("Canonical Phoneme Index")
                #                     axs[i, j].set_ylabel("Acoustic Feature Index")
                #             plt.tight_layout()
                #             attn_file = os.path.join(attn_dir, f"{ids[attn_id].split('/')[-1]}_epoch{epoch}.png")
                #             plt.savefig(attn_file)
                #             plt.close()
                #             print(f"Saved attention map to {attn_file}")

                self.ctc_metrics.append(ids, p_ctc_feat, targets, wav_lens, target_lens)
                # self.ctc_metrics_fuse.append(ids, sequence_decoder_out, targets, wav_lens, target_lens)
                # CTC-only results
                self.per_metrics.append(
                    ids=ids,
                    predict=sequence,
                    target=targets,
                    predict_len=None,
                    target_len=target_lens,
                    ind2lab=self.label_encoder.decode_ndim,
                )
                
                # Attention-only results
                self.per_metrics_fuse.append(
                    ids=ids,
                    predict=sequence_decoder_out,
                    target=targets,
                    predict_len=None,
                    target_len=target_lens,
                    ind2lab=self.label_encoder.decode_ndim,
                )
                
                # Joint CTC-Attention results (greedy)
                if hasattr(self, 'per_metrics_joint'):
                    self.per_metrics_joint.append(
                        ids=ids,
                        predict=joint_sequences_greedy,
                        target=targets,
                        predict_len=None,
                        target_len=target_lens,
                        ind2lab=self.label_encoder.decode_ndim,
                    )
                
                # Joint CTC-Attention results (beam search)
                if hasattr(self, 'per_metrics_joint_beam'):
                    self.per_metrics_joint_beam.append(
                        ids=ids,
                        predict=joint_sequences_beam,
                        target=targets,
                        predict_len=None,
                        target_len=target_lens,
                        ind2lab=self.label_encoder.decode_ndim,
                    )
                
                # MPD metrics
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
                    predict=sequence_decoder_out,
                    canonical=canonicals,
                    perceived=perceiveds,
                    predict_len=None,
                    canonical_len=canonical_lens,
                    perceived_len=perceived_lens,
                    ind2lab=self.label_encoder.decode_ndim,
                )
                
                # Joint MPD metrics
                if hasattr(self, 'mpd_metrics_joint'):
                    self.mpd_metrics_joint.append(
                        ids=ids,
                        predict=joint_sequences_greedy,
                        canonical=canonicals,
                        perceived=perceiveds,
                        predict_len=None,
                        canonical_len=canonical_lens,
                        perceived_len=perceived_lens,
                        ind2lab=self.label_encoder.decode_ndim,
                    )
                
                if hasattr(self, 'mpd_metrics_joint_beam'):
                    self.mpd_metrics_joint_beam.append(
                        ids=ids,
                        predict=joint_sequences_beam,
                        canonical=canonicals,
                        perceived=perceiveds,
                        predict_len=None,
                        canonical_len=canonical_lens,
                        perceived_len=perceived_lens,
                        ind2lab=self.label_encoder.decode_ndim,
                    )

        # Log to wandb if available (VALID stage only)
        current_epoch = self.hparams.epoch_counter.current
        if stage == sb.Stage.VALID and current_epoch % 5 == 0:
            try:
                import wandb
                wandb.log({
                    "loss": loss.item(),
                }, step=self.hparams.epoch_counter.current)
                # if loss_ga is not None:
                #     wandb.log({"loss_ga": loss_ga.item()}, step=self.hparams.epoch_counter.current)
                if loss_dec_out is not None:
                    wandb.log({"loss_dec_out": loss_dec_out.item()}, step=self.hparams.epoch_counter.current)
                if loss_ctc is not None:
                    wandb.log({"loss_ctc_head": loss_ctc.item()}, step=self.hparams.epoch_counter.current)
            except Exception:
                pass

        return loss


    def on_stage_end(self, stage, stage_loss, epoch):
        current_stage = self.hparams.epoch_counter.current
        """Gets called at the end of a epoch."""
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        else:
            if current_stage % 1 == 0:
                per = self.per_metrics.summarize("error_rate")
                mpd_f1 = self.mpd_metrics.summarize("mpd_f1")
                
                # Get joint decoding results if available
                # per_joint_greedy = self.per_metrics_joint.summarize("error_rate") if hasattr(self, 'per_metrics_joint') else per
                # per_joint_beam = self.per_metrics_joint_beam.summarize("error_rate") if hasattr(self, 'per_metrics_joint_beam') else per
                # mpd_f1_joint_greedy = self.mpd_metrics_joint.summarize("mpd_f1") if hasattr(self, 'mpd_metrics_joint') else mpd_f1
                # mpd_f1_joint_beam = self.mpd_metrics_joint_beam.summarize("mpd_f1") if hasattr(self, 'mpd_metrics_joint_beam') else mpd_f1

        if stage == sb.Stage.VALID and current_stage % 1 == 0:
            # Log stats
            valid_stats = {
                "loss": stage_loss,
                "ctc_loss": self.ctc_metrics.summarize("average"),
                "PER": per,
                "mpd_f1": mpd_f1,
                # "PER_joint_greedy": per_joint_greedy,
                # "PER_joint_beam": per_joint_beam,
                # "mpd_f1_joint_greedy": mpd_f1_joint_greedy,
                # "mpd_f1_joint_beam": mpd_f1_joint_beam,
            }
            
            self.hparams.train_logger.log_stats(
                stats_meta={
                    "epoch": epoch,
                    "lr_adam": self.adam_optimizer.param_groups[0]["lr"],
                    "lr_pretrained": self.pretrained_opt_class.param_groups[0]["lr"],
                },
                train_stats={"loss": self.train_loss},
                valid_stats=valid_stats,
            )
            # Save best 3 Models - prioritize joint beam search results
            improved = False
            
            # Use best joint decoding result for model selection
            best_per = per
            best_mpd_f1 = mpd_f1
            
            # Save best 3 PER models (lower is better)
            if best_per < self.best_per or len(self.best_per_list) < 3:
                ckpt_name = f"best_per_joint_{epoch:03d}_{best_per:.4f}.ckpt"
                self.checkpointer.save_and_keep_only(
                    meta={
                        "PER": per, 
                        "best_PER": best_per,
                        "mpd_f1": mpd_f1, 
                        "epoch": epoch
                    },
                    name=ckpt_name,
                    num_to_keep=5,
                    min_keys=["best_PER"]
                )
                self.best_per_list.append((best_per, epoch, ckpt_name))
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
            if best_mpd_f1 > self.best_mpd_f1 or len(self.best_mpd_f1_list) < 3:
                ckpt_name = f"best_mpdf1_joint_{epoch:03d}_{best_mpd_f1:.4f}.ckpt"
                self.checkpointer.save_and_keep_only(
                    meta={
                        "PER": per, 
                        "mpd_f1": mpd_f1,
                        "best_mpd_f1": best_mpd_f1,
                        "epoch": epoch
                    },
                    name=ckpt_name,
                    num_to_keep=5,
                    max_keys=["best_mpd_f1"]
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
            # if not files for joint decoding, create files
            if hasattr(self, 'per_metrics_joint'):
                self.per_metrics_joint = self.hparams.per_file.replace(".txt", "_joint.txt")
                with open(self.per_metrics_joint, "w") as w:
                    w.write("Joint CTC-Attention PER stats:\n")
                    self.per_metrics_joint.write_stats(w)
                    print(
                        "Joint CTC-Attention PER stats written to file",
                        self.per_metrics_joint,
                    )
            if hasattr(self, 'mpd_metrics_joint'):
                self.mpd_metrics_joint = self.hparams.mpd_file.replace(".txt", "_joint.txt")
                with open(self.mpd_metrics_joint, "w") as m:
                    m.write("Joint CTC-Attention MPD results and stats:\n")
                    self.mpd_metrics_joint.write_stats(m)
                    print(
                        "Joint CTC-Attention MPD results and stats written to file",
                        self.mpd_metrics_joint,
                    )
            # joint beam search results
            if hasattr(self, 'per_metrics_joint_beam'):
                self.per_metrics_joint_beam = self.hparams.per_file.replace(".txt", "_joint_beam.txt")
                with open(self.per_metrics_joint_beam, "w") as w:
                    w.write("Joint CTC-Attention PER stats (beam search):\n")
                    self.per_metrics_joint_beam.write_stats(w)
                    print(
                        "Joint CTC-Attention PER stats (beam search) written to file",
                        self.per_metrics_joint_beam,
                    )
            if hasattr(self, 'mpd_metrics_joint_beam'):
                self.mpd_metrics_joint_beam = self.hparams.mpd_file.replace(".txt", "_joint_beam.txt")
                with open(self.mpd_metrics_joint_beam, "w") as m:
                    m.write("Joint CTC-Attention MPD results and stats (beam search):\n")
                    self.mpd_metrics_joint_beam.write_stats(m)
                    print(
                        "Joint CTC-Attention MPD results and stats (beam search) written to file",
                        self.mpd_metrics_joint_beam,
                    )
                    