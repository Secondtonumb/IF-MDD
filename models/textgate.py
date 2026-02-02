import torch
import torch.nn as nn
import torch.nn.functional as F
from speechbrain.lobes.models.transformer.Transformer import TransformerDecoder
from utils.layers.utils import make_pad_mask
from speechbrain.nnet.attention import RelPosEncXL, RelPosMHAXL, RoPEMHA 

class TextGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.audio_gate_net = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        # 这是一个简单的 Attention 用来把 Text 对齐到 Audio 长度
        # 实际论文中可能使用更复杂的 MFA 对齐结果直接 expand
        # self.attn_layer = nn.MultiheadAttention(embed_dim=dim, num_heads=4, batch_first=True)

    def forward(self, q_audio, k_text, v_text):
        """
        q_audio: [Batch, 100, Dim] (长)
        k_text:  [Batch, 10, Dim]  (短)
        v_text:  [Batch, 10, Dim]
        """
        
        # --- 步骤 1: 对齐 (Alignment) ---
        # 利用标准 Attention 把 Text "拉伸" 成 Audio 的长度
        # Q=Audio, K=Text, V=Text
        # 输出 context 的形状会变成 [Batch, 100, Dim]，和 q_audio 一样长
        aligned_text, _ = self.attn_layer(query=q_audio, key=k_text, value=v_text)
        
        # 现在我们可以把 aligned_text 当作图里的 K 和 V (或者分开处理)
        # 为了对应图示，我们要用对齐后的文本特征
        k_aligned = aligned_text
        v_aligned = aligned_text 

        # --- 步骤 2: TextGate 核心逻辑 (你图里的部分) ---
        
        # 文本自门控 (k 和 v 现在都是 100 长度，可以点乘)
        text_info = v_aligned * torch.sigmoid(k_aligned)
        
        # 音频门控 (q 是 100 长度)
        audio_gate = self.audio_gate_net(q_audio)
        
        # 融合
        gated_interaction = text_info * audio_gate
        
        # 残差 (可以相加了，因为长度都是 100)
        output = q_audio + gated_interaction
        
        return output, gated_interaction

# fuse_net_layers: 2

# fuse_net: !new:speechbrain.lobes.models.transformer.Transformer.TransformerDecoder
#     num_layers: !ref <fuse_net_layers>
#     nhead: !ref <nhead>
#     d_ffn: !ref <d_ffn>
#     d_model: !ref <dnn_neurons>
#     activation: !ref <activation>
#     attention_type: "RelPosMHAXL"
#     causal: true

class TextGate_TransDec(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.audio_gate_net = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        # 这是一个简单的 Attention 用来把 Text 对齐到 Audio 长度
        # 实际论文中可能使用更复杂的 MFA 对齐结果直接 expand
        self.fuse_net = TransformerDecoder(
            num_layers=2,
            nhead=8,
            d_ffn=2048,
            d_model=dim,
            # activation="relu",
            attention_type="RelPosMHAXL",
            causal=True
        )

    def forward(self, q_audio, k_text, v_text, q_audio_lens=None, k_text_lens=None):
        """
        q_audio: [Batch, 100, Dim] (长)
        k_text:  [Batch, 10, Dim]  (短)
        v_text:  [Batch, 10, Dim]
        """
        
        # --- 步骤 1: 对齐 (Alignment) ---
        # 利用标准 Attention 把 Text "拉伸" 成 Audio 的长度
        # Q=Audio, K=Text, V=Text
        # 输出 context 的形状会变成 [Batch, 100, Dim]，和 q_audio 一样长
        # aligned_text, _ = self.attn_layer(query=q_audio, key=k_text, value=v_text)
        aligned_text, _,  fuse_attn  = self.fuse_net(tgt=q_audio, 
                                   memory=v_text,
                                   tgt_key_padding_mask=make_pad_mask(q_audio.shape[1] * q_audio_lens, maxlen=q_audio.shape[1]).to(q_audio.device) if q_audio_lens is not None else None,
                                   memory_key_padding_mask=make_pad_mask(k_text.shape[1] * k_text_lens, maxlen=k_text.shape[1]).to(k_text.device) if k_text_lens is not None else None,
                                   pos_embs_tgt=RelPosEncXL(emb_dim=self.dim)(q_audio).to(q_audio.device),
                                   pos_embs_src=RelPosEncXL(emb_dim=self.dim)(k_text).to(k_text.device)
                                   )
        
        # 现在我们可以把 aligned_text 当作图里的 K 和 V (或者分开处理)
        # 为了对应图示，我们要用对齐后的文本特征
        k_aligned = aligned_text
        v_aligned = aligned_text 

        # --- 步骤 2: TextGate 核心逻辑 (你图里的部分) ---
        
        # 文本自门控 (k 和 v 现在都是 100 长度，可以点乘)
        text_info = v_aligned * torch.sigmoid(k_aligned)
        
        # 音频门控 (q 是 100 长度)
        audio_gate = self.audio_gate_net(q_audio)
        
        # 融合
        gated_interaction = text_info * audio_gate
        
        # 残差 (可以相加了，因为长度都是 100)
        output = q_audio + gated_interaction
        
        return output, gated_interaction