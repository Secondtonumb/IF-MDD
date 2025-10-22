from huggingface_hub import hf_hub_download
import importlib.util
from trainer.MyEncoderASR import MyEncoderASR
from trainer.MyEncoderASR import MyCTCPrefixBeamSearcher
from trainer.MyEncoderASR import plot_alignments
import torch
import numpy as np

# # Customized Encoder ASR 
path = hf_hub_download(repo_id="Haopeng/CTC_for_IF-MDD", filename="MyEncoderASR.py")

# Dyanamic import
spec = importlib.util.spec_from_file_location("MyEncoderASR", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# Simple Transcribe
asr_model = MyEncoderASR.from_hparams(source="./pretrained_models/CTC_pretrained_IF_MDD", hparams_file="inference.yaml")
x = asr_model.transcribe_file("examples/arctic_b0503.wav")

# print(x)
# sil f ao r dh ah t w eh n t iy th t ay m sil dh ae t sil iy v n ih n sil d iy t uw m eh n sh uw k sil hh ae n s sil

# Get CTC Probabililty
waveform = asr_model.load_audio("examples/arctic_b0503.wav")
batch = waveform.unsqueeze(0)
rel_length = torch.tensor([1.0])

ctc_p = asr_model.encode_batch(batch, rel_length)
# print(ctc_p.shape)
# torch.Size([1, 221, 44])

# CTC Tokens
ctc_id = ctc_p.argmax(-1)

# Get verbose CTC output
searcher = MyCTCPrefixBeamSearcher(
    tokens=list(dict(sorted(asr_model.tokenizer.ind2lab.items())).values()),
    blank_index=asr_model.tokenizer.lab2ind["<blank>"],
    sil_index=asr_model.tokenizer.lab2ind["<blank>"]
)

# import pdb; pdb.set_trace()
hyps = searcher(ctc_p, rel_length)

# print(hyps)
''' [[CTCHypothesis(
    text=['sil', 'f', 'ao', 'r', 'dh', 'ah', 't', 'w', 'eh', 'n', 't', 'iy', 'th', 't', 'ay', 'm', 'sil', 'dh', 'ae', 't', 'sil', 'iy', 'v', 'n', 'ih', 'n', 'sil', 'd', 'iy', 't', 'uw', 'm', 'eh', 'n', 'sh', 'uw', 'k', 'sil', 'hh', 'ae', 'n', 's', 'sil'], 
    last_lm_state=None,
    score=-9.480588230616718,
    lm_score=-9.480588230616718,
    text_frames=[1, 8, 13, 15, 18, 21, 26, 28, 33, 35, 40, 44, 48, 56, 67, 75, 82, 90, 96, 100, 103, 109, 112, 116, 121, 125, 130, 134, 137, 144, 150, 155, 161, 163, 170, 176, 180, 183, 187, 196, 200, 209, 218]
    )]]
'''
print(hyps)

# Plot CTC output
emission = ctc_p
predicted_tokens = hyps[0][0].text
timesteps = torch.tensor(hyps[0][0].text_frames, dtype=torch.int32)

fig = plot_alignments(waveform=waveform,
                      emission=ctc_p,
                      tokens=predicted_tokens,
                      timesteps = torch.tensor(hyps[0][0].text_frames, dtype=torch.int32),
                      sample_rate=16000)

# fig.savefig("phoneme_wav_1.png")