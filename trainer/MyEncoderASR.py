from speechbrain.inference.ASR import EncoderASR
from speechbrain.decoders.ctc import TorchAudioCTCPrefixBeamSearcher
from speechbrain.decoders.ctc import CTCHypothesis
import torch
import speechbrain
import functools
import matplotlib.pyplot as plt

class MyEncoderASR(EncoderASR):
    def transcribe_batch(self, wavs, wav_lens):
        """Transcribes the input audio into a sequence of words

        The waveforms should already be in the model's desired format.
        You can call:
        ``normalized = EncoderASR.normalizer(signal, sample_rate)``
        to get a correctly converted signal in most cases.

        Arguments
        ---------
        wavs : torch.Tensor
            Batch of waveforms [batch, time, channels] or [batch, time]
            depending on the model.
        wav_lens : torch.Tensor
            Lengths of the waveforms relative to the longest one in the
            batch, tensor of shape [batch]. The longest one should have
            relative length 1.0 and others len(waveform) / max_length.
            Used for ignoring padding.

        Returns
        -------
        list
            Each waveform in the batch transcribed.
        tensor
            Each predicted token id.
        """
        with torch.no_grad():
            wav_lens = wav_lens.to(self.device)
            encoder_out = self.encode_batch(wavs, wav_lens)
            # frame level logits.
            predictions = self.decoding_function(encoder_out, wav_lens)
            is_ctc_text_encoder_tokenizer = isinstance(
                self.tokenizer, speechbrain.dataio.encoder.CTCTextEncoder
            )
            if isinstance(self.hparams.decoding_function, functools.partial):
                if is_ctc_text_encoder_tokenizer:
                    predicted_words = [
                        " ".join(self.tokenizer.decode_ndim(token_seq))
                        for token_seq in predictions
                    ]
                else:
                    predicted_words = [
                        self.tokenizer.decode_ids(token_seq)
                        for token_seq in predictions
                    ]
            else:
                predicted_words = [hyp[0].text for hyp in predictions]
        return predicted_words, predictions

class MyCTCPrefixBeamSearcher(TorchAudioCTCPrefixBeamSearcher):
    def decode_beams(self, log_probs, wav_len):
        """Decode log_probs using TorchAudio CTC decoder.

        If `using_cpu_decoder=True` then log_probs and wav_len are moved to CPU before decoding.
        When using CUDA CTC decoder, the timestep information is not available. Therefore, the timesteps
        in the returned hypotheses are set to None.

        Make sure that the input are in the log domain. The decoder will fail to decode
        logits or probabilities. The input should be the log probabilities of the CTC output.

        Arguments
        ---------
        log_probs : torch.Tensor
            The log probabilities of the input audio.
            Shape: (batch_size, seq_length, vocab_size)
        wav_len : torch.Tensor, default: None
            The speechbrain-style relative length. Shape: (batch_size,)
            If None, then the length of each audio is assumed to be seq_length.

        Returns
        -------
        list of list of CTCHypothesis
            The decoded hypotheses. The outer list is over the batch dimension, and the inner list is over the topk dimension.
        """
        if wav_len is not None:
            wav_len = log_probs.size(1) * wav_len
        else:
            wav_len = torch.tensor(
                [log_probs.size(1)] * log_probs.size(0),
                device=log_probs.device,
                dtype=torch.int32,
            )

        if wav_len.dtype != torch.int32:
            wav_len = wav_len.to(torch.int32)

        if log_probs.dtype != torch.float32:
            raise ValueError("log_probs must be float32.")

        # When using CPU decoder, we need to move the log_probs and wav_len to CPU
        if self.using_cpu_decoder and log_probs.is_cuda:
            log_probs = log_probs.cpu()

        if self.using_cpu_decoder and wav_len.is_cuda:
            wav_len = wav_len.cpu()

        if not log_probs.is_contiguous():
            raise RuntimeError("log_probs must be contiguous.")

        results = self._ctc_decoder(log_probs, wav_len)

        tokens_preds = []
        words_preds = []
        scores_preds = []
        timesteps_preds = []

        # over batch dim
        for i in range(len(results)):
            if self.using_cpu_decoder:
                preds = [
                    results[i][j].tokens.tolist()
                    for j in range(len(results[i]))
                ]
                preds = [
                    [self.tokens[token] for token in tokens] for tokens in preds
                ]
                tokens_preds.append(preds)

                timesteps = [
                    results[i][j].timesteps.tolist()
                    for j in range(len(results[i]))
                ]
                timesteps_preds.append(timesteps)

            else:
                # no timesteps is available for CUDA CTC decoder
                timesteps = [None for _ in range(len(results[i]))]
                timesteps_preds.append(timesteps)

                preds = [results[i][j].tokens for j in range(len(results[i]))]
                preds = [
                    [self.tokens[token] for token in tokens] for tokens in preds
                ]
                tokens_preds.append(preds)

            words = [results[i][j].words for j in range(len(results[i]))]
            words_preds.append(words)

            scores = [results[i][j].score for j in range(len(results[i]))]
            scores_preds.append(scores)

        hyps = []
        for (
            batch_index,
            (batch_text, batch_score, batch_timesteps),
        ) in enumerate(zip(tokens_preds, scores_preds, timesteps_preds)):
            hyps.append([])
            for text, score, timestep in zip(
                batch_text, batch_score, batch_timesteps
            ):
                hyps[batch_index].append(
                    CTCHypothesis(
                        text=text,
                        last_lm_state=None,
                        score=score,
                        lm_score=score,
                        text_frames=timestep,
                    )
                )
        return hyps

def plot_alignments(waveform, emission, tokens, timesteps, sample_rate):
    t = torch.arange(waveform.size(0)) / sample_rate
    ratio = waveform.size(0) / emission.size(1) / sample_rate

    chars = []
    words = []
    word_start = None
    for token, timestep in zip(tokens, timesteps * ratio):
        if token == "|":
            if word_start is not None:
                words.append((word_start, timestep))
            word_start = None
        else:
            chars.append((token, timestep))
            if word_start is None:
                word_start = timestep

    num_axes = len(waveform) // sample_rate + 1
    plt.figure(figsize=[num_axes*10, 5])
    fig, axes = plt.subplots(num_axes, 1)

    def _plot(ax, xlim):
        ax.plot(t, waveform)
        for token, timestep in chars:
            ax.annotate(token.upper(), (timestep, 0.5))
        for word_start, word_end in words:
            ax.axvspan(word_start, word_end, alpha=0.1, color="red")
        ax.set_ylim(-0.6, 0.7)
        ax.set_yticks([0])
        ax.grid(True, axis="y")
        ax.set_xlim(xlim)

    for i in range(0, num_axes):
        _plot(axes[i], (i, i+1))
    
    axes[num_axes-1].set_xlabel("time (sec)")
    fig.tight_layout()
    
    return fig