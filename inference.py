import sys
from pathlib import Path

# Add .speechbrain to Python path
speechbrain_path = Path.home() / ".speechbrain"
if speechbrain_path.exists():
    sys.path.insert(0, str(speechbrain_path))

from huggingface_hub import hf_hub_download
import importlib.util
from trainer.MyEncoderASR import MyEncoderASR
from trainer.MyEncoderASR import MyCTCPrefixBeamSearcher
from trainer.MyEncoderASR import MyCTCBeamSearcher

from speechbrain.decoders.ctc import TorchAudioCTCPrefixBeamSearcher, CTCPrefixBeamSearcher, CTCBeamSearcher

from trainer.MyEncoderASR import plot_alignments
import torch
import numpy as np
import tgt
import itertools

# file_id = "arctic_b0503"
# file_id= "DOS_F01_S6_001"
file_id = "0002-0000"

# # Customized Encoder ASR 
path = hf_hub_download(repo_id="Haopeng/CTC_for_IF-MDD", filename="MyEncoderASR.py")

# Dyanamic import
spec = importlib.util.spec_from_file_location("MyEncoderASR", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# Simple Transcribe
asr_model = MyEncoderASR.from_hparams(source="/home/kevingenghaopeng/MDD/IF-MDD/pretrained_models/CTC_for_IF-MDD", hparams_file="inference.yaml")
# asr_model = MyEncoderASR.from_hparams(source="/home/kevingenghaopeng/MDD/IF-MDD/pretrained_models/CTCwithLP", hparams_file="inference.yaml")
x = asr_model.transcribe_file(f"examples/{file_id}.wav")

# print(x)
# sil f ao r dh ah t w eh n t iy th t ay m sil dh ae t sil iy v n ih n sil d iy t uw m eh n sh uw k sil hh ae n s sil

# Get CTC Probabililty
waveform = asr_model.load_audio(f"examples/{file_id}.wav")
batch = waveform.unsqueeze(0)
rel_length = torch.tensor([1.0])

ctc_p = asr_model.encode_batch(batch, rel_length)
# print(ctc_p.shape)
# torch.Size([1, 221, 44])

# CTC Tokens
ctc_id = ctc_p.argmax(-1)

# Get verbose CTC output
# import pdb; pdb.set_trace()
searcher = MyCTCPrefixBeamSearcher(
    tokens=list(dict(sorted(asr_model.tokenizer.ind2lab.items())).values()),
    blank_index=asr_model.tokenizer.lab2ind["<blank>"],
    sil_index=asr_model.tokenizer.lab2ind["sil"],
)

# import pdb; pdb.set_trace()
hyps = searcher(ctc_p, rel_length)

s = TorchAudioCTCPrefixBeamSearcher(
    tokens=list(dict(sorted(asr_model.tokenizer.ind2lab.items())).values()),
    blank_index=asr_model.tokenizer.lab2ind["<blank>"],
    sil_index=asr_model.tokenizer.lab2ind["sil"],
)
hyps_ = s(ctc_p, rel_length)
import pdb; pdb.set_trace()

# s1 = CTCPrefixBeamSearcher(
#      blank_index=asr_model.tokenizer.lab2ind["<blank>"],
#      vocab_list= list(dict(sorted(asr_model.tokenizer.ind2lab.items())).values())
# )
# hyps__ = s1(ctc_p, rel_length)

# searcher = CTCBeamSearcher(
#     blank_index=asr_model.tokenizer.lab2ind["<blank>"],
#     vocab_list=list(dict(sorted(asr_model.tokenizer.ind2lab.items())).values())
# )

# new_hyps = searcher(ctc_p, rel_length)

searcher = MyCTCBeamSearcher(
    blank_index=asr_model.tokenizer.lab2ind["<blank>"],
    vocab_list=list(dict(sorted(asr_model.tokenizer.ind2lab.items())).values()),
)
new_hyps_ = searcher(ctc_p, rel_length)

import pdb; pdb.set_trace()


# # print(hyps)
# ''' [[CTCHypothesis(
#     text=['sil', 'f', 'ao', 'r', 'dh', 'ah', 't', 'w', 'eh', 'n', 't', 'iy', 'th', 't', 'ay', 'm', 'sil', 'dh', 'ae', 't', 'sil', 'iy', 'v', 'n', 'ih', 'n', 'sil', 'd', 'iy', 't', 'uw', 'm', 'eh', 'n', 'sh', 'uw', 'k', 'sil', 'hh', 'ae', 'n', 's', 'sil'], 
#     last_lm_state=None,
#     score=-9.480588230616718,
#     lm_score=-9.480588230616718,
#     text_frames=[1, 8, 13, 15, 18, 21, 26, 28, 33, 35, 40, 44, 48, 56, 67, 75, 82, 90, 96, 100, 103, 109, 112, 116, 121, 125, 130, 134, 137, 144, 150, 155, 161, 163, 170, 176, 180, 183, 187, 196, 200, 209, 218]
#     )]]
# '''
print(hyps)

# # Plot CTC output
# emission = ctc_p
# predicted_tokens = hyps[0][0].text
# timesteps = torch.tensor(hyps[0][0].text_frames, dtype=torch.int32)

# fig = plot_alignments(waveform=waveform,
#                       emission=ctc_p,
#                       tokens=predicted_tokens,
#                       timesteps = torch.tensor(hyps[0][0].text_frames, dtype=torch.int32),
#                       sample_rate=16000)

# fig.savefig("phoneme_wav_1.png")

# # Create TextGrid output
# def create_textgrid_from_ctc(predicted_tokens, text_frames, sample_rate=16000):
#     """
#     Create a TextGrid object from CTC predictions.
    
#     Args:
#         predicted_tokens: List of predicted token strings
#         text_frames: List of frame indices for each token
#         sample_rate: Audio sample rate (default: 16000)
    
#     Returns:
#         tgt.TextGrid object
#     """
#     # Calculate timestamps (converting frames to seconds)
#     # Assuming frame rate: each frame is approximately 20ms (320 samples at 16kHz)
#     frame_shift = 320  # samples per frame
#     timestamps = [frame * frame_shift / sample_rate for frame in text_frames]
    
#     # Add start time (0.0) and calculate intervals
#     start_times = [0.0] + timestamps[:-1]
#     end_times = timestamps
    
#     # Get total duration
#     total_duration = end_times[-1] if end_times else 0.0
    
#     # Create TextGrid
#     textgrid = tgt.TextGrid()
    
#     # Create phoneme tier
#     phoneme_tier = tgt.IntervalTier(name="phonemes", start_time=0.0, end_time=total_duration)
    
#     for i, (token, start, end) in enumerate(zip(predicted_tokens, start_times, end_times)):
#         interval = tgt.Interval(start_time=start, end_time=end, text=token)
#         phoneme_tier.add_interval(interval)
    
#     textgrid.add_tier(phoneme_tier)
    
#     return textgrid

# # Generate TextGrid
# textgrid = create_textgrid_from_ctc(
#     predicted_tokens=hyps[0][0].text,
#     text_frames=hyps[0][0].text_frames,
#     sample_rate=16000
# )

# # Save TextGrid to file
# output_textgrid_path = f"examples/{file_id}.TextGrid"
# tgt.io.write_to_file(textgrid, output_textgrid_path, format='long')
# print(f"TextGrid saved to: {output_textgrid_path}")

# # Print intervals in JSON-like format (similar to main.py)
# def textgrid_to_dict(textgrid):
#     """Convert TextGrid to dictionary format."""
#     response_data = {}
#     for tier in textgrid.tiers:
#         if not isinstance(tier, tgt.IntervalTier):
#             continue
        
#         intervals_data = []
#         for interval in tier.intervals:
#             if interval.text:  # Skip empty intervals
#                 intervals_data.append({
#                     "start": round(interval.start_time, 4),
#                     "end": round(interval.end_time, 4),
#                     "content": interval.text,
#                 })
#         response_data[tier.name] = intervals_data
#     return response_data

# # Print as JSON-like format
# intervals_dict = textgrid_to_dict(textgrid)
# print("\nTextGrid intervals:")
# for tier_name, intervals in intervals_dict.items():
#     print(f"\n{tier_name}:")
#     for interval in intervals:
#         print(f"  {interval['start']:.4f} - {interval['end']:.4f}: {interval['content']}")

# # Generate interactive HTML visualization
# from generate_interactive_html import generate_interactive_html

# audio_file = f"examples/{file_id}.wav"
# output_html = f"examples/{file_id}_viewer.html"

# # Get phoneme intervals
# phoneme_intervals = intervals_dict.get('phonemes', [])

# print(f"\n🎨 Generating interactive HTML visualization...")
# generate_interactive_html(audio_file, phoneme_intervals, output_html)
# print(f"\n🌐 在浏览器中打开: {output_html}")

# import pdb; pdb.set_trace()
# from speechbrain.alignment.ctc_segmentation import CTCSegmentation
# import pdb; pdb.set_trace()

# # model 
# # aligner = CTCSegmentation(asr_model, kaldi_style_text=False, text_converter="classic")
# # using example file included in the SpeechBrain repository
# from speechbrain.inference.ASR import EncoderDecoderASR
# from speechbrain.alignment.ctc_segmentation import CTCSegmentation
# # load an ASR model
# pre_trained = "speechbrain/asr-transformer-transformerlm-librispeech"
# asr_model = EncoderDecoderASR.from_hparams(source=pre_trained)
# aligner = CTCSegmentation(asr_model, kaldi_style_text=False)
# # load data
# audio_path = "/home/kevingenghaopeng/MDD/IF-MDD/examples/{file_id}.wav"
from speechbrain.inference import EncoderASR
from speechbrain.integrations.k2_fsa.align import CTCAligner
from speechbrain.utils.fetching import fetch

# wav = fetch("example.wav", source="speechbrain/asr-wav2vec2-commonvoice-en")
# asr_model = EncoderASR.from_hparams(source="speechbrain/asr-wav2vec2-librispeech", savedir="pretrained_models/asr-wav2vec2-librispeech")
# import pdb; pdb.set_trace()
# asr_model.transcribe_file(str(wav))
# aligner = CTCAligner(model=asr_model, tokenizer=asr_model.tokenizer, device=asr_model.device)
# alignment = aligner.align_audio_to_words(wav, "THE BIRCH CANOE SLID ON THE SMOOTH PLANKS", frame_shift=0.02)
# print(alignment)

from speechbrain.integrations.k2_fsa.align import CTCAligner
# from speechbrain.integrations.k2_fsa.align import CTCAlignerforPhonemes

# import pdb;pdb.set_trace()

aligner = CTCAligner(model=asr_model, tokenizer=asr_model.tokenizer, device=asr_model.device)
# import pdb; pdb.set_trace()

# alignments = aligner.align_audio_to_words(audio_file=f"examples/{file_id}.wav",
#                                          transcript=x.split(" "),
#                                          frame_shift=0.02)
alignments = aligner.align_audio_to_tokens(audio_file=f"examples/{file_id}.wav",
                                         transcript=x.split(" "))
# pdb.set_trace()

# log_probs, log_prob_len, targets = aligner.get_log_prob_and_targets(audio_files=f"examples/{file_id}.wav", transcripts=x.split(" "))
# alignments = aligner.align(log_probs, log_prob_len, targets)
# print(alignments)
# word_alignments = aligner.get_word_alignment(alignments, transcripts=x)
# pdb.set_trace()

print(alignments)

# ========== CTC Alignment to Timestamps Converter ==========

def ctc_alignment_to_timestamps(alignment, tokenizer, frame_shift_ms=20):
    """
    Convert frame-level CTC alignment to phoneme timestamps.
    
    In CTC format, blank frames belong to the previous phoneme's segment.
    
    Args:
        alignment: List[int] - Frame-level alignment (phoneme IDs), 0 is blank
        tokenizer: The tokenizer with ind2lab mapping
        frame_shift_ms: Frame shift in milliseconds (default: 20ms)
    
    Returns:
        dict with:
            - 'text': List[str] - phoneme sequence
            - 'text_frames': List[int] - start frame of each phoneme
            - 'timestamps': List[Tuple[float, float, str]] - (start_sec, end_sec, phoneme)
    """
    text = []
    text_frames = []
    timestamps = []
    
    if not alignment:
        return {'text': [], 'text_frames': [], 'timestamps': []}
    
    # Track current phoneme and its start frame
    current_phone_id = None
    start_frame = 0
    
    for frame_idx, phone_id in enumerate(alignment):
        # Blank frame - belongs to previous phoneme's duration, so continue
        if phone_id == 0:
            continue
        
        # New phoneme detected (different from current)
        if phone_id != current_phone_id:
            # Save previous phoneme if exists
            if current_phone_id is not None:
                phoneme = tokenizer.ind2lab[current_phone_id]
                text.append(phoneme)
                text_frames.append(start_frame)
                
                # End time includes all frames up to (but not including) current frame
                # This includes any blank frames after the last occurrence of the phoneme
                start_sec = start_frame * frame_shift_ms / 1000.0
                end_sec = (frame_idx - 1) * frame_shift_ms / 1000.0
                timestamps.append((start_sec, end_sec, phoneme))
            
            # Start tracking new phoneme
            current_phone_id = phone_id
            start_frame = frame_idx
    
    # Handle last phoneme (extend to end of sequence, including trailing blanks)
    if current_phone_id is not None:
        phoneme = tokenizer.ind2lab[current_phone_id]
        text.append(phoneme)
        text_frames.append(start_frame)
        
        start_sec = start_frame * frame_shift_ms / 1000.0
        end_sec = (len(alignment) - 1) * frame_shift_ms / 1000.0
        timestamps.append((start_sec, end_sec, phoneme))
    
    return {
        'text': text,
        'text_frames': text_frames,
        'timestamps': timestamps
    }


# Example usage with your alignment
# example_alignment = [9, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21, 0, 0, 0, 0, 0, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 18, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33, 33, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 9, 0, 0, 0, 0, 0, 0, 0, 14, 0, 0, 0, 0, 30, 22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 26, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 35, 0, 0, 0, 0, 0, 0, 0, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33, 33, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 41, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 0, 0, 0, 0, 0, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 38, 38, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21, 0, 0, 0, 0, 0, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 38, 0, 0, 0, 0, 0, 0, 0, 26, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33, 0, 0, 0, 9, 9, 9, 9, 9, 9]

result = ctc_alignment_to_timestamps(alignments, asr_model.tokenizer, frame_shift_ms=20)
print(f"\nPhoneme sequence: {result['text']}")
print(f"Start frames: {result['text_frames']}")
print(f"Timestamps: {result['timestamps'][:5]}...")  # Show first 5

# ========== Generate Comparison HTML ==========

from generate_comparison_html import generate_comparison_html

print("\n🎨 Generating comparison visualization...")

# Method 1: CTC Prefix Beam Search results
intervals_method1 = []
text_frames = hyps[0][0].text_frames
text = hyps[0][0].text
for i in range(len(text)):
    start_frame = text_frames[i]
    # End frame is the start of next phoneme, or total frames
    if i + 1 < len(text_frames):
        end_frame = text_frames[i+1] - 1
    else:
        # Last phoneme extends to end of audio
        end_frame = len(ctc_p[0]) - 1
    
    start_sec = start_frame * 0.02
    end_sec = end_frame * 0.02
    intervals_method1.append({
        'start': round(start_sec, 4),
        'end': round(end_sec, 4),
        'content': text[i]
    })

# Method 2: K2 Forced Alignment results
intervals_method2 = [
    {
        'start': round(start, 4),
        'end': round(end, 4),
        'content': phoneme
    }
    for start, end, phoneme in result['timestamps']
]

output_html = f"examples/{file_id}_comparison.html"

generate_comparison_html(
    audio_path=f"examples/{file_id}.wav",
    intervals_data1=intervals_method1,
    intervals_data2=intervals_method2,
    output_html_path=output_html,
    method1_name="CTC Prefix Beam Search",
    method2_name="K2 Forced Alignment"
)

print(f"\n🌐 在浏览器中打开: {output_html}")

