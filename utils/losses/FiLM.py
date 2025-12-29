import torch
import torch.nn as nn
import torch.nn.functional as F

class FiLMOnLogits(nn.Module):
    def __init__(self, cond_dim=384, n_class=44, hidden=128, gamma_max=2.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2 * n_class)  # -> [γ, β]
        )
        self.gamma_max = gamma_max

    def forward(self, fuse_feat, logits):
        """
        fuse_feat: [B,T,384]
        logits:    [B,T,44]  (音素 logits)
        return:    [B,T,44]  (调制后的 logits)
        """
        gb = self.mlp(fuse_feat)            # [B,T,88]
        gamma, beta = gb.chunk(2, dim=-1)   # [B,T,44], [B,T,44]
        # 让缩放稳定：γ ∈ (0, gamma_max)
        gamma = torch.sigmoid(gamma) * self.gamma_max
        import pdb; pdb.set_trace()
        return gamma * logits + beta
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class FiLMAndHiddenFuse(nn.Module):
    def __init__(
        self,
        cond_dim=384,      # fuse_feat 的维度
        n_class=44,        # 类别数
        hidden_mlp=128,    # 产生 FiLM 参数的中间层
        gamma_max=2.0,     # γ 的上限（用 sigmoid 映射到 (0, gamma_max)）
        temp=1.0,          # 将 logits -> prob 的温度
        gate_per_dim=True  # True: 每维门控；False: 标量门控
    ):
        super().__init__()
        # 1) 用 fuse_feat 生成 FiLM 参数(γ, β)，调制 logits
        self.film_mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_mlp),
            nn.ReLU(),
            nn.Linear(hidden_mlp, 2 * n_class)  # -> [γ, β]
        )
        self.gamma_max = gamma_max

        # 2) 将类别分布提升回隐藏维（n_class -> cond_dim）
        #    这里先 softmax 得到 prob 再线性投影，更数值稳。
        self.class2hid = nn.Linear(n_class, cond_dim, bias=True)
        self.temp = temp

        # 3) 与 fuse_feat 做门控融合得到 fused_hidden
        gate_out = cond_dim if gate_per_dim else 1
        self.gate_fc = nn.Linear(cond_dim * 2, gate_out)
        self.ln_fuse = nn.LayerNorm(cond_dim)
        self.ln_from_logits = nn.LayerNorm(cond_dim)
        self.gate_per_dim = gate_per_dim

    def forward(self, fuse_feat, logits):
        """
        fuse_feat: [B, T, cond_dim]   —— 早期 mispro 特征
        logits   : [B, T, n_class]    —— 音素 logits（未过 softmax）
        returns:
          fused_logits : [B, T, n_class]
          fused_hidden : [B, T, cond_dim]
          extras: dict( gamma, beta, gate )
        """
        B, T, _ = fuse_feat.shape

        # ---- 1) FiLM 调制 logits ----
        gb = self.film_mlp(fuse_feat)            # [B,T,2*n_class]
        gamma, beta = gb.chunk(2, dim=-1)        # [B,T,n_class], [B,T,n_class]
        gamma = torch.sigmoid(gamma) * self.gamma_max
        fused_logits = gamma * logits + beta     # 仍为 logits 语义

        # ---- 2) logits -> prob -> 提升到隐藏维 ----
        probs = F.softmax(logits / self.temp, dim=-1)   # [B,T,n_class]
        h_from_logits = self.class2hid(probs)           # [B,T,cond_dim]
        # 轻微标准化，便于与 fuse_feat 融合
        h_from_logits = self.ln_from_logits(h_from_logits)
        fuse_norm = self.ln_fuse(fuse_feat)

        # ---- 3) 门控融合得到 fused_hidden ----
        gate_inp = torch.cat([fuse_norm, h_from_logits], dim=-1)  # [B,T,2*cond_dim]
        gate = torch.sigmoid(self.gate_fc(gate_inp))               # [B,T,cond_dim] 或 [B,T,1]
        fused_hidden = gate * fuse_norm + (1.0 - gate) * h_from_logits

        return fused_logits, fused_hidden, {"gamma": gamma, "beta": beta, "gate": gate}
    
class TemperatureGating(nn.Module):
    def __init__(self, cond_dim=384, tau_min=0.7, tau_max=1.8):
        super().__init__()
        self.proj = nn.Linear(cond_dim, 1)
        self.tau_min, self.tau_max = tau_min, tau_max

    def forward(self, fuse_feat, logits):
        # 先产生一个标量温度（逐时刻）
        s = torch.sigmoid(self.proj(fuse_feat))  # [B,T,1] in (0,1)
        tau = self.tau_min + (self.tau_max - self.tau_min) * s
        return logits / tau  # 仍返回 logits
    
class BiasAdapter(nn.Module):
    def __init__(self, cond_dim=384, n_class=44, scale=1.0):
        super().__init__()
        self.proj = nn.Linear(cond_dim, n_class)
        self.scale = scale

    def forward(self, fuse_feat, logits):
        delta = self.proj(fuse_feat)  # [B,T,44]
        return logits + self.scale * delta