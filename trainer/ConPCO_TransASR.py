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
from speechbrain.nnet.activations import Swish
from speechbrain.nnet.containers import ModuleList
from speechbrain.nnet.linear import Linear
from speechbrain.utils.dynamic_chunk_training import DynChunkTrainConfig
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)

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
        self._init_params()
        
