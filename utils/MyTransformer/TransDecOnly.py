from speechbrain.decoders.seq2seq import S2STransformerBeamSearcher, S2SBaseSearcher
import torch
from speechbrain.decoders.utils import (
    _update_mem,
    inflate_tensor,
    mask_by_condition,
)
from speechbrain.lobes.models.transformer.Transformer import (
    NormalizedEmbedding,
    TransformerInterface,
    get_key_padding_mask,
    get_lookahead_mask,
)
from speechbrain.lobes.models.transformer.Transformer import TransformerDecoder
from speechbrain.lobes.models.transformer.TransformerASR import make_transformer_src_tgt_masks
from speechbrain.lobes.models.transformer.TransformerASR import TransformerASR

from speechbrain.dataio.dataio import length_to_mask


class S2STransDecOnlyBeamSearcher(S2STransformerBeamSearcher):
    def __init__(self, modules, temperature=1.0, **kwargs):
        super().__init__(modules, temperature=temperature, **kwargs)
        self.model = modules[0]
        self.fc = modules[1]
        self.softmax = torch.nn.LogSoftmax(dim=-1)
        self.temperature = temperature
    
    def forward_step(self, inp_tokens, memory, enc_states, enc_lens):
        """Performs a step in the implemented beamsearcher."""
        memory = _update_mem(inp_tokens, memory)
        pred, attn = self.model(memory, enc_states, enc_lens)
        prob_dist = self.softmax(self.fc(pred) / self.temperature)
        return prob_dist[:, -1, :], memory, attn

class TransformerDecoderOnlyASR(TransformerASR):
    @torch.no_grad()
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

        # src = self.custom_src_module(src)
        # # add pos encoding to queries if are sinusoidal ones else
        # if (
        #     self.attention_type == "hypermixing"
        #     or self.attention_type == "RoPEMHA"
        # ):
        #     pos_embs_encoder = None
        # elif self.attention_type == "RelPosMHAXL":
        #     pos_embs_encoder = self.positional_encoding(src)
        # elif self.positional_encoding_type == "fixed_abs_sine":
        #     src = src + self.positional_encoding(src)
        #     pos_embs_encoder = None

        # No Encoder' use in as
        encoder_out = src

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

        # if self.output_hidden_states:
        #     return encoder_out, hidden_states, decoder_out
        # else:
        return encoder_out, decoder_out
    
    def decode(self, tgt, encoder_out, enc_len=None):
        """This method implements a decoding step for the transformer model.

        Arguments
        ---------
        tgt : torch.Tensor
            The sequence to the decoder.
        encoder_out : torch.Tensor
            Hidden output of the encoder.
        enc_len : torch.LongTensor
            The actual length of encoder states.

        Returns
        -------
        prediction
        """
        tgt_mask = get_lookahead_mask(tgt)
        src_key_padding_mask = None
        if enc_len is not None:
            src_key_padding_mask = (1 - length_to_mask(enc_len)).bool()

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

        prediction, self_attns, multihead_attns = self.decoder(
            tgt,
            encoder_out,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=src_key_padding_mask,
            pos_embs_tgt=pos_embs_target,
            pos_embs_src=pos_embs_encoder,
        )
        return prediction, multihead_attns[-1]