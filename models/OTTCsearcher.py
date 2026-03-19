import torch
from speechbrain.decoders.seq2seq import S2STransformerBeamSearcher
from speechbrain.utils.data_utils import undo_padding
import torch.nn.functional as F

class S2SSharedEncoderOTTCBeamSearcher(S2STransformerBeamSearcher):
    def __init__(
        self, 
        modules, 
        ottc_lin, 
        lm_weight_module,     # 新增: 用于生成 weights_logits 的网络层
        unigram_module,       # 新增: 用于生成 weights_labels 的模块/词表映射
        sinkhorn_cost_fn, 
        ottc_weight=0.5, 
        temperature=1.0, 
        **kwargs
    ):
        kwargs['return_topk'] = True 
        super().__init__(modules, temperature=temperature, **kwargs)
        
        self.ottc_lin = ottc_lin
        self.lm_weight_module = lm_weight_module
        self.unigram_module = unigram_module
        self.sinkhorn_cost_fn = sinkhorn_cost_fn
        self.ottc_weight = ottc_weight
    def create_attention_mask_from_input_sequence(self, lens_abs, max_len):
            """辅助函数：根据绝对长度生成布尔掩码"""
            batch_size = lens_abs.shape[0]
            seq_range = torch.arange(max_len, device=lens_abs.device).unsqueeze(0).expand(batch_size, -1)
            mask = seq_range < lens_abs.unsqueeze(1)
            return mask
    def forward(self, enc_states, wav_len, canonical_tokens=None):
            # 1. 运行 Transformer Beam Search
            topk_hyps, topk_lengths, topk_scores, topk_log_probs = super().forward(enc_states, wav_len)
            
            batch_size = enc_states.shape[0]
            
            # ====================================================
            # 2. 生成全局的 Dense 特征和声学边缘权重 (weights_logits)
            # ====================================================
            ottc_dense = self.ottc_lin(enc_states) 
            
            # 严格复刻你训练时的逻辑
            weights_logits_batch = self.lm_weight_module(enc_states).squeeze(-1) # (Batch, Time)
            lens_abs = (wav_len * enc_states.shape[1]).int()
            output_mask = self.create_attention_mask_from_input_sequence(lens_abs, enc_states.shape[1])
            
            weights_logits_batch = weights_logits_batch.masked_fill(output_mask == 0, -torch.inf)
            weights_logits_batch = F.softmax(weights_logits_batch, dim=-1)
            
            # 3. 开始 OTTC 重打分
            for b in range(batch_size):
                actual_audio_len = lens_abs[b].item()
                b_ottc_feat = ottc_dense[b:b+1, :actual_audio_len, :] 
                
                # 取出当前句子的有效权重 (1, Actual_Time)
                b_weight_logits = weights_logits_batch[b:b+1, :actual_audio_len]
                
                for k in range(self.topk):
                    hyp_len = int(topk_lengths[b, k] * topk_hyps.size(2))
                    hyp_seq = topk_hyps[b, k, :hyp_len].unsqueeze(0).long() # (1, Hyp_Len)
                    
                    # ====================================================
                    # 动态生成文本边缘权重 (weights_labels)
                    # ====================================================
                    # 这里假设你的 unigram_module 接收 token IDs 并输出 weight logits
                    w_labels_logits = self.unigram_module(hyp_seq).squeeze(-1)
                    b_weight_labels = F.softmax(w_labels_logits, dim=-1) # (1, Hyp_Len)
                    
                    # 转换为 One-hot
                    one_hot_hyp = F.one_hot(hyp_seq, num_classes=self.ottc_lin.out_features).float()
                    
                    # ====================================================
                    # 调用 Sinkhorn Cost
                    # ====================================================
                    ot_cost, _, _, _ = self.sinkhorn_cost_fn(
                        x=b_ottc_feat,
                        y=one_hot_hyp,
                        a=b_weight_logits,    # 使用学习到的声学信息密度
                        b=b_weight_labels,    # 使用学习到的文本 Unigram 分布
                        amask=None,
                        bmask=None,
                        euclidian=False,
                        jsd=False
                    )

                    # 5. 联合重打分
                    topk_scores[b, k] -= self.ottc_weight * ot_cost.item()
                    
            # 4. 根据更新后的 scores 重新排序并提取全局 Top-1
            sorted_scores, sorted_indices = torch.sort(topk_scores, dim=-1, descending=True)
            
            sorted_hyps = torch.zeros_like(topk_hyps)
            sorted_lengths = torch.zeros_like(topk_lengths)
            sorted_log_probs = torch.zeros_like(topk_log_probs)
            
            for b in range(batch_size):
                sorted_hyps[b] = topk_hyps[b, sorted_indices[b]]
                sorted_lengths[b] = topk_lengths[b, sorted_indices[b]]
                sorted_log_probs[b] = topk_log_probs[b, sorted_indices[b]]
                
            best_hyps_tensor = sorted_hyps[:, 0, :]
            best_lens = sorted_lengths[:, 0]
            best_scores = sorted_scores[:, 0]
            best_log_probs = sorted_log_probs[:, 0, :]

            hyps = undo_padding(best_hyps_tensor, best_lens)

            return hyps, best_lens, best_scores, best_log_probs