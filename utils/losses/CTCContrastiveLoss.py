import torch
import torch.nn as nn
import torch.nn.functional as F

class MDDContrastiveLoss(nn.Module):
    def __init__(self, ctc_loss, margin=16.0):
        super(MDDContrastiveLoss, self).__init__()
        self.margin = margin
        # reduction='none' 是为了手动处理 ln p(L|X)
        self.ctc_loss = ctc_loss
        
    def forward(self, log_probs, target_annot, target_ref, input_lengths, annot_lengths, ref_lengths):
        """
        Args:
            log_probs: 模型输出的 log_softmax 概率 [T, N, C]
            target_annot: 用户实际发音的标注 (L^e)
            target_ref: 标准文本的读音 (L)
            input_lengths: 输入序列长度
            annot_lengths: L^e 的长度
            ref_lengths: L 的长度
        """
        
        # 1. 计算 ln p(L^e | X) -> 实际上是 -CTCLoss
        # PyTorch 的 CTCLoss 返回的是 -ln p(L|X)
        neg_log_p_annot = self.ctc_loss(log_probs, target_annot, input_lengths, annot_lengths)
        log_p_annot = -neg_log_p_annot

        # 2. 计算 ln p(L | X)
        neg_log_p_ref = self.ctc_loss(log_probs, target_ref, input_lengths, ref_lengths)
        log_p_ref = -neg_log_p_ref

        # 3. 根据公式 (8) 计算 Contrastive Loss [cite: 183]
        # Contrast = max(ln p(L^e|X) - ln p(L|X) + margin, 0)
        # 注意：这里假设发生了误读。如果 L == L^e，通常这部分 loss 会趋于 margin 或被屏蔽。
        contrastive_loss = torch.clamp(log_p_annot - log_p_ref + self.margin, min=0.0)

        return contrastive_loss.mean()

# 使用示例
# criterion = MDDContrastiveLoss(nn.CTCLoss(blank=0, reduction='none', zero_infinity=True), margin=16)
# loss_ctc = nn.CTCLoss()(log_probs, target_annot, input_lengths, annot_lengths)
# loss_contrast = criterion(log_probs, target_annot, target_ref, ...)
# total_loss = loss_ctc + loss_contrast [cite: 158]