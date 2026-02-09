
import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedSeq2SeqAttention (nn.Module):
    """
    Gated Attention机制用于调制seq2seq输出
    
    使用Mispronunciation信息作为gate来引导seq2seq logits学习perceived phoneme特征
    包含残差连接以保留原始信息
    
    Args:
        input_dim (int): seq2seq输出的特征维度 (C)
        gate_dim (int): gate特征维度 (e.g., 1 for binary, 4 for multi-class)
        hidden_dim (int): gate网络的隐藏层维度
        use_residual (bool): 是否使用残差连接
        gate_type (str): gate类型 ('sigmoid', 'softmax', 'tanh')
    """
    def __init__(self, input_dim, gate_dim=1, hidden_dim=128, use_residual=True, gate_type='sigmoid'):
        super().__init__()
        self.input_dim = input_dim
        self.gate_dim = gate_dim
        self.use_residual = use_residual
        self.gate_type = gate_type
        
        # 投影gate特征到与输出兼容的维度
        self.gate_proj = nn.Linear(gate_dim, hidden_dim)
        self.gate_fc = nn.Linear(hidden_dim, input_dim)
        
        # 门控网络
        self.gate_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # 融合网络（可选的额外处理）
        self.fusion_net = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        self.layer_norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, seq_logits, gate_features):
        """
        Args:
            seq_logits: [B, T_p+1, C] - seq2seq输出logits
            gate_features: [B, T_c, gate_dim] - mispronunciation特征
                          可以是h_mispro_bin [B, T_c, 1] 或 h_mispro_cls [B, T_c, 4]
        
        Returns:
            output: [B, T_p+1, C] - 经过gated attention调制的输出
        """
        B, T_p, C = seq_logits.shape
        B_gate, T_c, gate_dim = gate_features.shape
        
        # 1. 投影gate特征到与seq输出兼容的维度
        gate_proj = self.gate_proj(gate_features)  # [B, T_c, hidden_dim]
        gate_proj = F.relu(gate_proj)
        gate_weights = self.gate_fc(gate_proj)  # [B, T_c, C]
        
        # 2. 处理时间维度对齐 (T_c vs T_p+1)
        if T_c != T_p:
            # 使用线性插值将gate权重从T_c对齐到T_p
            gate_weights = gate_weights.transpose(1, 2)  # [B, C, T_c]
            gate_weights = F.interpolate(
                gate_weights, 
                size=T_p, 
                mode='linear', 
                align_corners=False
            )  # [B, C, T_p]
            gate_weights = gate_weights.transpose(1, 2)  # [B, T_p, C]
        
        # 3. 应用gate激活函数
        if self.gate_type == 'sigmoid':
            gate = torch.sigmoid(gate_weights)  # [B, T_p, C]
        elif self.gate_type == 'softmax':
            gate = F.softmax(gate_weights, dim=-1)  # [B, T_p, C]
        elif self.gate_type == 'tanh':
            gate = torch.tanh(gate_weights)  # [B, T_p, C]
        else:
            gate = torch.sigmoid(gate_weights)
        
        # 4. 生成额外的调制信号
        modulation = self.gate_net(seq_logits)  # [B, T_p, C]
        modulation = torch.tanh(modulation)
        
        # 5. 融合gate和调制信号
        gated_output = seq_logits * gate + modulation * (1 - gate)  # [B, T_p, C]
        
        # 6. 融合网络处理（可选增强）
        combined = torch.cat([seq_logits, gated_output], dim=-1)  # [B, T_p, 2*C]
        enhanced = self.fusion_net(combined)  # [B, T_p, C]
        enhanced = self.dropout(enhanced)
        
        # 7. 残差连接
        if self.use_residual:
            output = seq_logits + enhanced  # [B, T_p, C]
        else:
            output = enhanced
        
        # 8. 层正则化
        output = self.layer_norm(output)
        
        return output
