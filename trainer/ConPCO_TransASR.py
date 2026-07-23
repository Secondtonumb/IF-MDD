# Define custom ASR's TGT Head to ensure normalized embedding

import speechbrain as sb
from speechbrain.lobes.models.transformer.TransformerASR import TransformerASR

from dataclasses import dataclass
from typing import Any, Optional

import torch  # noqa 42
from torch import nn

from speechbrain.dataio.dataio import length_to_mask
from speechbrain.lobes.models.transformer.Transformer import (
    NormalizedEmbedding,
    TransformerInterface,
    get_key_padding_mask,
    get_lookahead_mask,
)
from speechbrain.lobes.models.transformer.TransformerASR import (
    make_transformer_src_tgt_masks,
)
from speechbrain.nnet.activations import Swish
from speechbrain.nnet.containers import ModuleList
from speechbrain.nnet.linear import Linear
from speechbrain.utils.dynamic_chunk_training import DynChunkTrainConfig
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


class _IdentityEncoder(nn.Module):
    """Preserve the no-op encoder used by zero-layer legacy recipes."""

    def __init__(self, output_hidden_states=False):
        super().__init__()
        self.output_hidden_states = output_hidden_states

    def forward(
        self,
        src,
        src_mask=None,
        src_key_padding_mask=None,
        pos_embs=None,
        **kwargs,
    ):
        if self.output_hidden_states:
            return src, [], [src]
        return src, []


class ConPCO_TransformerASR(TransformerASR):
    def __init__(
        self,
        tgt_vocab,
        input_size,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ffn=2048,
        dropout=0.1,
        activation=nn.ReLU,
        positional_encoding="fixed_abs_sine",
        normalize_before=False,
        kernel_size: Optional[int] = 31,
        bias: bool = True,
        encoder_module: str = "transformer",
        conformer_activation: type = Swish,
        branchformer_activation: type = nn.GELU,
        attention_type: str = "regularMHA",
        max_length: int = 2500,
        causal: Optional[bool] = None,
        csgu_linear_units: int = 3072,
        gate_activation: type = nn.Identity,
        use_linear_after_conv: bool = False,
        output_hidden_states=False,
        layerdrop_prob=0.0,
        custom_src_module=None,
        custom_tgt_module=None,
        custom_encoder=None,
        custom_decoder=None,
        encoder_proj_decoder=None,
        post_encoder_reduction_factor=1,
    ):
        if causal is None:
            logger.warning(
                "`causal` not specified for `TransformerASR`, assuming `True` for compatibility. "
                "We strongly recommend that you explicitly set this. "
                "If you are using a model or recipe defined before v1.0, it might now be BROKEN! "
                "If so, please see https://github.com/speechbrain/speechbrain/issues/2604"
            )
            causal = True

        super().__init__(
            tgt_vocab=tgt_vocab,
            input_size=input_size,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            d_ffn=d_ffn,
            dropout=dropout,
            activation=activation,
            positional_encoding=positional_encoding,
            normalize_before=normalize_before,
            kernel_size=kernel_size,
            bias=bias,
            encoder_module=encoder_module,
            conformer_activation=conformer_activation,
            branchformer_activation=branchformer_activation,
            attention_type=attention_type,
            max_length=max_length,
            causal=causal,
            csgu_linear_units=csgu_linear_units,
            gate_activation=gate_activation,
            use_linear_after_conv=use_linear_after_conv,
            output_hidden_states=output_hidden_states,
            layerdrop_prob=layerdrop_prob,
        )
        # init first

        # reset parameters using xavier_normal_
        self._init_params()

        # Recent SpeechBrain versions leave ``encoder`` undefined for a
        # zero-layer recipe. This model still calls it to obtain decoder memory.
        if not hasattr(self, "encoder"):
            self.encoder = _IdentityEncoder(output_hidden_states)
        
        if custom_src_module is not None:
            self.custom_src_module = custom_src_module
        else:
            self.custom_src_module = ModuleList(
                Linear(
                    input_size=input_size,
                    n_neurons=d_model,
                    bias=True,
                    combine_dims=False,
                ),
                torch.nn.Dropout(dropout),
            )
        if custom_encoder is not None:
            # import pdb; pdb.set_trace()
            self.encoder = custom_encoder
        if encoder_proj_decoder is not None:
            self.encoder_proj_decoder = encoder_proj_decoder
            self.post_encoder_reduction_factor = post_encoder_reduction_factor
        else:
            self.encoder_proj_decoder = None

        if num_decoder_layers > 0:
            if custom_decoder is not None:
                self.decoder = custom_decoder
                
            if custom_tgt_module is not None:
                self.custom_tgt_module = custom_tgt_module
            else:
                self.custom_tgt_module = ModuleList(
                    NormalizedEmbedding(d_model, tgt_vocab),
                )
                # LayerNorm,
                # torch.nn.LayerNorm(d_model),

        # reset parameters using xavier_normal_
        # self._init_params()
        
    def forward(self, src, tgt, wav_len=None, pad_idx=0):
        """
        Arguments
        ---------
        src : torch.Tensor
            The sequence to the encoder.
        tgt : torch.Tensor
            The sequence to the decoder.
        wav_len: torch.Tensor, optional
            Torch Tensor of shape (batch, ) containing the relative length to padded length for each example.
        pad_idx : int, optional
            The index for <pad> token (default=0).

        Returns
        -------
        encoder_out : torch.Tensor
            The output of the encoder.
        decoder_out : torch.Tensor
            The output of the decoder
        hidden_state_lst : list, optional
            The output of the hidden layers of the encoder.
            Only works if output_hidden_states is set to true.
        """

        # reshape the src vector to [Batch, Time, Fea] is a 4d vector is given
        if src.ndim == 4:
            bz, t, ch1, ch2 = src.shape
            src = src.reshape(bz, t, ch1 * ch2)

        (
            src_key_padding_mask,
            tgt_key_padding_mask,
            src_mask,
            tgt_mask,
        ) = make_transformer_src_tgt_masks(
            src, tgt, wav_len, causal=self.causal, pad_idx=pad_idx
        )

        src = self.custom_src_module(src)
        # add pos encoding to queries if are sinusoidal ones else
        if (
            self.attention_type == "hypermixing"
            or self.attention_type == "RoPEMHA"
        ):
            pos_embs_encoder = None
        elif self.attention_type == "RelPosMHAXL":
            pos_embs_encoder = self.positional_encoding(src)
        elif self.positional_encoding_type == "fixed_abs_sine":
            src = src + self.positional_encoding(src)
            pos_embs_encoder = None
        
        
        outputs = self.encoder(
            src=src,
            src_mask=src_mask,
            src_key_padding_mask=src_key_padding_mask,
            pos_embs=pos_embs_encoder,
        )
        

        # if encoder only, we return the output of the encoder
        if tgt is None:
            return outputs
        

    
        if self.output_hidden_states:
            encoder_out, _, hidden_states = outputs
        else:
            encoder_out, _ = outputs
        
        # add conv projector to decoder
        # import pdb; pdb.set_trace()
        if self.encoder_proj_decoder is not None:
            encoder_proj = self.encoder_proj_decoder(encoder_out)
            
        tgt = self.custom_tgt_module(tgt)

        if (
            self.attention_type == "RelPosMHAXL"
            or self.attention_type == "RoPEMHA"
        ):
            tgt = tgt + self.positional_encoding_decoder(tgt)
            pos_embs_encoder = None
            pos_embs_target = None
        elif (
            self.positional_encoding_type == "fixed_abs_sine"
            or self.attention_type == "hypermixing"
        ):
            tgt = tgt + self.positional_encoding(tgt)
            pos_embs_target = None
            pos_embs_encoder = None
        # import pdb; pdb.set_trace()
        if self.encoder_proj_decoder is not None:
            # apply new src_key_padding_mask, shirnked by post_encoder_reduction_factor
            if self.post_encoder_reduction_factor >= 1:
                (
                    src_key_padding_mask_proj,
                    tgt_key_padding_mask,
                    src_mask_proj,
                    tgt_mask,
                ) = make_transformer_src_tgt_masks(
                    encoder_proj, tgt, wav_len, causal=self.causal, pad_idx=pad_idx
                )
            # import pdb; pdb.set_trace()
            decoder_out, _, _ = self.decoder(
                tgt=tgt,
                memory=encoder_proj,
                memory_mask=None,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=src_key_padding_mask_proj,
                pos_embs_tgt=pos_embs_target,
                pos_embs_src=pos_embs_encoder,
            )
        else:
            decoder_out, _, _ = self.decoder(
                tgt=tgt,
                memory=encoder_out,
                memory_mask=None,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=src_key_padding_mask,
                pos_embs_tgt=pos_embs_target,
                pos_embs_src=pos_embs_encoder,
            )

        if self.output_hidden_states:
            if self.encoder_proj_decoder is not None:
                return encoder_out, encoder_proj, hidden_states, decoder_out
            else:
                return encoder_out, hidden_states, decoder_out
        else:
            if self.encoder_proj_decoder is not None:
                return encoder_out, encoder_proj, decoder_out
            else:
                return encoder_out, decoder_out