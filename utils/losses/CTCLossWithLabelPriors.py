import torch
from torch import Tensor, nn
import k2
from typing import List, Optional, Union, Tuple

class CTCLossWithLabelPriors(nn.Module):
    def __init__(self, prior_scaling_factor=0.0, ctc_implementation='k2', blank=0, reduction='sum', prior_momentum=0.9):
        super().__init__()
        
        self.ctc_implementation = ctc_implementation
        self.blank = blank
        self.reduction = reduction

        if ctc_implementation == 'k2':
            pass
        elif ctc_implementation == 'torch':
            self.ctc_loss = torch.nn.CTCLoss(blank=blank, reduction=reduction)
        elif ctc_implementation == 'ctc_primer':
            pass
        else:
            raise ValueError(f"ctc_implementation={ctc_implementation} is not supported")
    
        self.log_priors = None
        self.log_priors_sum = None
        self.num_samples = 0
        self.prior_scaling_factor = prior_scaling_factor  # This corresponds to the `alpha` hyper parameter in the paper
        
        # EMA smoothing for stable prior updates
        self.prior_momentum = prior_momentum  # 0.9 means 90% old + 10% new
        self.ema_log_priors = None  # Stores smoothed priors across epochs
    
    def forward(self, log_probs: Tensor, targets: Tensor, input_lengths: Tensor, target_lengths: Tensor, step_type="train") -> Tensor:
        '''
        log_probs: (T, N, C)
        '''
        if self.ctc_implementation == 'k2':
            return self.forward_k2(log_probs, targets, input_lengths, target_lengths, step_type=step_type)
        elif self.ctc_implementation == 'torch':
            return self.forward_torch(log_probs, targets, input_lengths, target_lengths, step_type=step_type)
        elif self.ctc_implementation == 'ctc_primer':
            return self.forward_ctc_primer(log_probs, targets, input_lengths, target_lengths, step_type=step_type)
        else:
            raise ValueError(f"ctc_implementation={self.ctc_implementation} is not supported")

    def forward_torch(self, log_probs: Tensor, targets: Tensor, input_lengths: Tensor, target_lengths: Tensor, step_type="train") -> Tensor:
        # Note:
        # Pytorch's ctc implementation is not suitable to use with label priors.
        # It will lead to incorrect gradients.
        # Check out this issue for more details: https://github.com/pytorch/pytorch/issues/122243
        # Here, we will only use pytorch's ctc loss without label priors as a sanity check.

        loss = self.ctc_loss(log_probs, targets, input_lengths.int(), target_lengths.int())
        return loss

    def forward_ctc_primer(self, log_probs: Tensor, targets: Tensor, input_lengths: Tensor, target_lengths: Tensor, step_type="train") -> Tensor:
        try:
            import ctc as ctc_primer
        except ImportError:
            raise ImportError(
                "Please install a python implementation of CTC loss from 'https://github.com/vadimkantorov/ctc'." \
                + "For example:" \
                + "  git clone https://github.com/vadimkantorov/ctc.git ctc_python" \
                + "  export PYTHONPATH=<path-to-ctc-python>:$PYTHONPATH" \
            )
        
        # Accumulate label priors for this epoch
        log_probs = log_probs.permute(1, 0, 2)  # (T, N, C) -> (N, T, C)
        if step_type == "train":
            log_probs_flattened = []
            for lp, le in zip(log_probs, input_lengths):
                log_probs_flattened.append(lp[:int(le.item())])      
            log_probs_flattened = torch.cat(log_probs_flattened, 0)

            # Note, the log_probs here is already log_softmax'ed.
            T = log_probs_flattened.size(0)
            self.num_samples += T
            log_batch_priors_sum = torch.logsumexp(log_probs_flattened, dim=0, keepdim=True)
            log_batch_priors_sum = log_batch_priors_sum.detach()
            if self.log_priors_sum is None:
                self.log_priors_sum = log_batch_priors_sum
            else:
                _temp = torch.stack([self.log_priors_sum, log_batch_priors_sum], dim=-1)
                self.log_priors_sum = torch.logsumexp(_temp, dim=-1)

        # Apply the label priors
        if self.log_priors is not None and self.prior_scaling_factor > 0:
            log_probs = log_probs - self.log_priors * self.prior_scaling_factor

        # Compute CTC loss
        log_probs = log_probs.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)
        loss = ctc_primer.ctc_loss(
            log_probs, 
            targets.to(torch.int64), 
            input_lengths.to(torch.int64), 
            target_lengths.to(torch.int64), 
            blank=self.blank, 
            reduction='none'
        ).sum()
        return loss

    def encode_supervisions(
        self, targets, target_lengths, input_lengths
    ) -> Tuple[torch.Tensor, Union[List[str], List[List[int]]]]:
        # https://github.com/k2-fsa/icefall/blob/master/icefall/utils.py#L181
        
        batch_size = targets.size(0)
        supervision_segments = torch.stack(
            (
                torch.arange(batch_size),
                torch.zeros(batch_size),
                input_lengths.cpu(),
            ),
            1,
        ).to(torch.int32)

        indices = torch.argsort(supervision_segments[:, 2], descending=True)
        supervision_segments = supervision_segments[indices]

        # Be careful: the targets here are already padded! We need to remove paddings from it
        res = targets[indices].tolist()
        res_lengths = target_lengths[indices].tolist()
        res = [l[:l_len] for l, l_len in zip(res, res_lengths)]

        return supervision_segments, res, indices

    def forward_k2(self, log_probs: Tensor, targets: Tensor, input_lengths: Tensor, target_lengths: Tensor, step_type="train") -> Tensor:
        supervision_segments, token_ids, indices = self.encode_supervisions(targets, target_lengths, input_lengths)

        decoding_graph = k2.ctc_graph(token_ids, modified=False, device=log_probs.device)

        # TODO: graph compiler for multiple pronunciations

        # Accumulate label priors for this epoch
        log_probs = log_probs.permute(1, 0, 2)  # (T, N, C) -> (N, T, C)
        if step_type == "train":
            log_probs_flattened = []
            for lp, le in zip(log_probs, input_lengths):
                log_probs_flattened.append(lp[:int(le.item())])      
            log_probs_flattened = torch.cat(log_probs_flattened, 0)

            # Note, the log_probs here is already log_softmax'ed.
            T = log_probs_flattened.size(0)
            self.num_samples += T
            log_batch_priors_sum = torch.logsumexp(log_probs_flattened, dim=0, keepdim=True)
            log_batch_priors_sum = log_batch_priors_sum.detach()
            if self.log_priors_sum is None:
                self.log_priors_sum = log_batch_priors_sum
            else:
                _temp = torch.stack([self.log_priors_sum, log_batch_priors_sum], dim=-1)
                self.log_priors_sum = torch.logsumexp(_temp, dim=-1)

        # Apply the label priors
        if self.log_priors is not None and self.prior_scaling_factor > 0:
            log_probs = log_probs - self.log_priors * self.prior_scaling_factor

        # Compute CTC loss
        dense_fsa_vec = k2.DenseFsaVec(
            log_probs,  # (N, T, C)
            supervision_segments,
        )

        loss = k2.ctc_loss(
            decoding_graph=decoding_graph,
            dense_fsa_vec=dense_fsa_vec,
            output_beam=10,
            reduction=self.reduction,
            use_double_scores=True,
        )
            
        return loss