#!/usr/bin/env python3
"""
run_test.py — offline CTC / OTTC inference, forced alignment, and evaluation.

Loads a checkpoint produced by run_exp.py, runs batched CTC log-probability
extraction, performs CTC Viterbi forced alignment against reference phone
sequences, writes Kaldi-format phone CTMs, and reports TSE and ACC against
ground-truth CTMs.

Default test corpora (mirrors run_ctc_align_wavlm.sh):
  - timit_dev
  - timit_test
  - buckeye_train

Usage
-----
    python run_test.py \
        --checkpoint_dir wavlm-large/train.100/checkpoint-114160 \
        --encoder_name wavlm-large \
        --loss_type ottc \
        --output_dir ./test_results

    # Single corpus, explicit GPU:
    python run_test.py \
        --checkpoint_dir /abs/path/to/exp \
        --encoder_name wavlm-large \
        --corpora timit_dev timit_test \
        --gpu 2

    # Pass a specific checkpoint sub-folder directly:
    python run_test.py \
        --checkpoint_dir .../wavlm-large/20260507/checkpoint-12000 \
        --encoder_name wavlm-large
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Data I/O  (mirrors ctc_align_and_eval.py)
# ─────────────────────────────────────────────────────────────────────────────

def read_wav_scp(path: str) -> Dict[str, str]:
    wav_map = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                wav_map[parts[0]] = parts[1]
    return wav_map


def read_phone_text(path: str) -> Dict[str, List[str]]:
    """Read text.phn: each line is  'uttid ph1 ph2 ...'"""
    phone_map = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                phone_map[parts[0]] = parts[1:]
    return phone_map


def read_ctm(path: str) -> Dict[str, List[Tuple[float, float, str]]]:
    """Read Kaldi CTM: uttid channel start dur label"""
    ctm: Dict[str, List] = defaultdict(list)
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                uttid = parts[0]
                start = float(parts[2])
                dur   = float(parts[3])
                label = parts[4]
                ctm[uttid].append((start, start + dur, label))
    return dict(ctm)


def write_ctm(ctm: Dict[str, List[Tuple[float, float, str]]], path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        for uttid in sorted(ctm):
            for start, end, label in ctm[uttid]:
                f.write(f"{uttid} 1 {start:.4f} {end - start:.4f} {label}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Metrics  (mirrors ctc_align_and_eval.py)
# ─────────────────────────────────────────────────────────────────────────────

_SIL = {"SIL", "sil", "SP", "sp", "SPN", "spn", ""}


def _filter_sil(entries: List[Tuple[float, float, str]]) -> List[Tuple[float, float, str]]:
    return [(s, e, l) for s, e, l in entries if l not in _SIL and l.upper() not in _SIL]


def compute_tse(
    ref:  Dict[str, List[Tuple[float, float, str]]],
    pred: Dict[str, List[Tuple[float, float, str]]],
    utterances: List[str],
) -> Dict[str, float]:
    utt_tse = {}
    for utt in utterances:
        if utt not in ref or utt not in pred:
            continue
        ref_f = _filter_sil(ref[utt])
        prd_f = _filter_sil(pred[utt])   # filter SIL from both sides
        if not ref_f or len(ref_f) != len(prd_f):
            continue
        tse = sum(
            abs(rs - ps) + abs(re - pe)
            for (rs, re, _), (ps, pe, _) in zip(ref_f, prd_f)
        )
        utt_tse[utt] = tse / len(ref_f)
    return utt_tse


def compute_acc(
    ref:  Dict[str, List[Tuple[float, float, str]]],
    pred: Dict[str, List[Tuple[float, float, str]]],
    utterances: List[str],
    tau: int = 20,
) -> Dict[str, float]:
    tau_sec = tau / 1000.0
    utt_acc = {}
    for utt in utterances:
        if utt not in ref or utt not in pred:
            continue
        ref_f = _filter_sil(ref[utt])
        prd_f = _filter_sil(pred[utt])   # filter SIL from both sides
        if not ref_f or len(ref_f) != len(prd_f):
            continue
        correct = sum(
            1 for (rs, re, _), (ps, pe, _) in zip(ref_f, prd_f)
            if abs(rs - ps) <= tau_sec and abs(re - pe) <= tau_sec
        )
        utt_acc[utt] = correct / len(ref_f) * 100.0
    return utt_acc


# ─────────────────────────────────────────────────────────────────────────────
# CTC Viterbi forced alignment
# ─────────────────────────────────────────────────────────────────────────────

_NEG_INF = float("-inf")


def ctc_forced_align(
    log_probs: np.ndarray,   # (T, V)  log-softmax
    targets:   List[int],    # phone-id sequence, no blanks
    blank:     int = 0,
) -> List[Tuple[int, int]]:
    """
    Standard CTC forward Viterbi with greedy traceback.
    Returns [(start_frame, end_frame), ...] for each target token.
    """
    T, V = log_probs.shape
    S = len(targets)
    # Expanded label sequence: blank t0 blank t1 ... tS blank
    expanded = [blank]
    for t in targets:
        expanded += [t, blank]
    L = len(expanded)   # 2S+1

    # Forward log-probabilities
    alpha = np.full((T, L), _NEG_INF, dtype=np.float64)
    alpha[0, 0] = log_probs[0, blank]
    if L > 1:
        alpha[0, 1] = log_probs[0, expanded[1]]

    for t in range(1, T):
        for s in range(L):
            lbl = expanded[s]
            a = alpha[t - 1, s]
            if s > 0:
                a = np.logaddexp(a, alpha[t - 1, s - 1])
            if s > 1 and expanded[s] != expanded[s - 2]:
                a = np.logaddexp(a, alpha[t - 1, s - 2])
            alpha[t, s] = a + log_probs[t, lbl]

    # Greedy backward traceback
    path = [-1] * T
    # Start at the last valid state
    s = L - 1 if alpha[T - 1, L - 1] >= alpha[T - 1, L - 2] else L - 2
    for t in range(T - 1, -1, -1):
        path[t] = s
        if t == 0:
            break
        best_s, best_v = s, alpha[t - 1, s]
        if s > 0 and alpha[t - 1, s - 1] > best_v:
            best_s, best_v = s - 1, alpha[t - 1, s - 1]
        if s > 1 and expanded[s] != expanded[s - 2] and alpha[t - 1, s - 2] > best_v:
            best_s = s - 2
        s = best_s

    # Collect spans: token idx k lives at expanded position 2k+1
    spans = []
    for k in range(S):
        pos    = 2 * k + 1
        frames = [t for t in range(T) if path[t] == pos]
        if frames:
            spans.append((frames[0], frames[-1]))
        else:
            # Fallback: equal-length split
            chunk = T // S
            spans.append((k * chunk, min((k + 1) * chunk - 1, T - 1)))
    return spans


def spans_to_ctm(
    spans:             List[Tuple[int, int]],
    phones:            List[str],
    audio_len_samples: int,
    hop_length:        int = 320,
    sr:                int = 16000,
) -> List[Tuple[float, float, str]]:
    frame_dur      = hop_length / sr
    audio_duration = audio_len_samples / sr
    result = []
    for (sf, ef), ph in zip(spans, phones):
        start_sec = sf * frame_dur
        end_sec   = min((ef + 1) * frame_dur, audio_duration)
        result.append((start_sec, end_sec, ph))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Model + processor loading  (mirrors run_exp.py)
# ─────────────────────────────────────────────────────────────────────────────

def _find_latest_checkpoint(directory: str) -> Optional[str]:
    prefix = "checkpoint-"
    candidates = [
        d for d in os.listdir(directory)
        if d.startswith(prefix) and os.path.isdir(os.path.join(directory, d))
    ]
    if not candidates:
        return None
    return os.path.join(directory, max(candidates, key=lambda x: int(x[len(prefix):])))


def load_model_and_processor(args):
    """
    Resolve the checkpoint path, detect wav2vec2 vs wavlm, load the model
    and its accompanying Wav2Vec2Processor.
    """
    from ottc.config.path_config import LARGE_MODELS_PATH
    from ottc.models.sequence.wav2vec2_ctc  import wvForCTC
    from ottc.models.sequence.wav2vec2_ottc import wvForOTTC
    from ottc.models.sequence.wavlm_ctc     import wlForCTC
    from ottc.models.sequence.wavlm_ottc    import wlForOTTC
    from transformers import Wav2Vec2Processor, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, AutoConfig

    ckpt_root = args.checkpoint_dir
    if not os.path.isabs(ckpt_root):
        ckpt_root = str(LARGE_MODELS_PATH / "large_models_results" / ckpt_root)

    # If ckpt_root already is a checkpoint-NNNN dir, use it directly;
    # otherwise find the latest sub-checkpoint.
    if os.path.basename(ckpt_root).startswith("checkpoint-"):
        ckpt_path = ckpt_root
    else:
        ckpt_path = _find_latest_checkpoint(ckpt_root) or ckpt_root
    logger.info(f"Loading checkpoint: {ckpt_path}")

    # Detect model type from the base encoder config
    encoder_path = str(LARGE_MODELS_PATH / args.encoder_name)
    base_cfg = AutoConfig.from_pretrained(encoder_path)
    if base_cfg.model_type == "wavlm":
        ModelCls = wlForOTTC if args.loss_type == "ottc" else wlForCTC
    else:
        ModelCls = wvForOTTC if args.loss_type == "ottc" else wvForCTC

    model = ModelCls.from_pretrained(ckpt_path)
    model.eval()

    # Processor: search in order: explicit arg → checkpoint dir → experiment root
    search_paths = []
    if args.processor_dir:
        search_paths.append(args.processor_dir)
    search_paths += [ckpt_path, ckpt_root]

    # Build the processor manually to avoid AutoConfig choking on the custom
    # "wavlm-ottc" / "wav2vec2-ottc" model_type in the checkpoint config.json.
    # The tokenizer (vocab.json) lives in the checkpoint; the feature extractor
    # config comes from the base encoder.

    vocab_dir = None
    for p in search_paths:
        if os.path.exists(os.path.join(p, "vocab.json")):
            vocab_dir = p
            break
    if vocab_dir is None:
        raise FileNotFoundError(
            f"vocab.json not found in any of: {search_paths}. "
            "Specify --processor_dir explicitly."
        )
    logger.info(f"Loading tokenizer from {vocab_dir}")
    tokenizer        = Wav2Vec2CTCTokenizer.from_pretrained(vocab_dir)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(encoder_path)
    processor         = Wav2Vec2Processor(feature_extractor=feature_extractor,
                                          tokenizer=tokenizer)

    return model, processor


# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────────────────────

SR         = 16_000
HOP_LENGTH = 320   # CNN stride for WavLM-large / Wav2Vec2-large (~20 ms/frame)


@torch.no_grad()
def get_log_probs(
    model, processor, audio: np.ndarray, device: torch.device
) -> np.ndarray:
    """Return (T, V) log-softmax probabilities for one utterance (CTC models)."""
    inputs = processor(audio, sampling_rate=SR, return_tensors="pt", padding=False)
    input_values = inputs.input_values.to(device)
    out = model(input_values=input_values)
    log_probs = F.log_softmax(out.logits[0], dim=-1).cpu().numpy()
    return log_probs


@torch.no_grad()
def get_ottc_weights(
    model, processor, audio: np.ndarray, device: torch.device
) -> np.ndarray:
    """Return p_weights (T_valid,) from the OTTC lm_weight head.

    Bypasses model.forward() which requires labels and crashes when labels=None.
    Works for both wlForOTTC (model.wavlm) and wvForOTTC (model.wav2vec2).
    """
    inputs = processor(audio, sampling_rate=SR, return_tensors="pt", padding=False)
    input_values = inputs.input_values.to(device)
    attention_mask = torch.ones(input_values.shape[:2], dtype=torch.long, device=device)

    # Detect backbone (wavlm vs wav2vec2)
    if hasattr(model, "wavlm"):
        backbone        = model.wavlm
        backbone_config = model.wavlm_config
    else:
        backbone        = model.wav2vec2
        backbone_config = model.wav2vec2_config

    hidden_states = backbone(
        input_values,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=False,
    )[0]                                         # (1, T, H)
    hidden_states = model.dropout(hidden_states)

    # Valid frame count (CNN downsampling)
    input_lengths = model._get_feat_extract_output_lengths(
        backbone_config, attention_mask.sum(-1)
    ).to(torch.long)
    T       = hidden_states.shape[1]
    T_valid = int(input_lengths[0].item())

    output_mask = torch.zeros(1, T, device=device)
    output_mask[0, :T_valid] = 1.0

    raw_w = model.lm_weight(hidden_states).squeeze(-1)          # (1, T)
    raw_w = raw_w.masked_fill(output_mask == 0, -torch.inf)
    p_weights = F.softmax(raw_w, dim=1)[0, :T_valid].cpu().numpy()  # (T_valid,)
    return p_weights


def ottc_segment(p_weights: np.ndarray, S: int) -> List[Tuple[int, int]]:
    """Duration-based OTTC segmentation (Approach 3 from the paper).

    The lm_weight head predicts per-frame probability mass p_weights (sums to 1).
    With uniform token weights q_u = 1/S the 1-D optimal-transport plan matches the
    two CDFs, assigning each frame to the token whose CDF bucket it falls into:

        End_Frame(u) = searchsorted(cumsum(p_weights), (u+1)/S)

    This is algebraically equivalent to running batched_ottc_loss_bucketized with
    uniform q_weights and reading off index_qy_ for every frame.
    """
    T = len(p_weights)
    if S == 0 or T == 0:
        return []
    cum_p      = np.clip(np.cumsum(p_weights), 0.0, 1.0)
    token_cdf  = np.arange(1, S + 1) / S            # [1/S, 2/S, ..., 1.0]
    end_frames = np.searchsorted(cum_p, token_cdf, side="left")
    end_frames = np.minimum(end_frames, T - 1)

    spans: List[Tuple[int, int]] = []
    start = 0
    for u in range(S):
        end = max(int(end_frames[u]), start)
        spans.append((start, end))
        start = end + 1
    # Ensure the last token covers remaining valid frames
    if spans:
        spans[-1] = (spans[-1][0], T - 1)
    return spans


def phones_to_ids(phones: List[str], processor) -> Optional[List[int]]:
    """Map phone strings to tokenizer IDs; return None if any unknown.

    Tries several normalisation forms in order so that both TIMIT-style
    (lowercase, no stress: 'eh') and CMU ARPAbet-style (uppercase, with
    stress markers: 'EH2') phone labels can be looked up in the same vocab:
      exact → uppercase → lowercase → strip-stress → strip-stress-upper → strip-stress-lower
    """
    vocab = processor.tokenizer.get_vocab()
    ids = []
    for ph in phones:
        stripped = ph.rstrip("012")
        tid = None
        for candidate in (ph, ph.upper(), ph.lower(),
                          stripped, stripped.upper(), stripped.lower()):
            tid = vocab.get(candidate)
            if tid is not None:
                break
        if tid is None:
            return None
        ids.append(tid)
    return ids


# ─────────────────────────────────────────────────────────────────────────────
# Per-corpus pipeline
# ─────────────────────────────────────────────────────────────────────────────

def process_corpus(
    wav_scp_path:  str,
    text_phn_path: str,
    ref_ctm_path:  Optional[str],
    output_dir:    str,
    model,
    processor,
    device:        torch.device,
    loss_type:     str = "ottc",
) -> dict:
    import librosa

    os.makedirs(output_dir, exist_ok=True)

    blank_id   = processor.tokenizer.pad_token_id   # CTC blank = pad token
    wav_map    = read_wav_scp(wav_scp_path)
    phone_map  = read_phone_text(text_phn_path)
    ref_ctm    = read_ctm(ref_ctm_path) if ref_ctm_path and os.path.exists(ref_ctm_path) else None
    utterances = sorted(set(wav_map) & set(phone_map))

    logger.info(f"  {len(utterances)} utterances")

    pred_ctm: Dict[str, List[Tuple[float, float, str]]] = {}
    skipped = 0

    for uttid in tqdm(utterances, desc="  aligning", ncols=90):

        print('uttid: ', uttid)
        wav_path = wav_map[uttid]
        phones   = phone_map[uttid]

        if not os.path.exists(wav_path):
            logger.warning(f"  [SKIP] audio not found: {wav_path}")
            skipped += 1
            continue

        try:
            audio, _ = librosa.load(wav_path, sr=SR)

            if loss_type == "ottc":
                # ── OTTC duration-based segmentation ────────────────────────
                # No vocab lookup needed: we only need the phone sequence (count S).
                # The lm_weight head predicts per-frame probability mass p_weights.
                # 1-D OT with uniform token weights gives boundaries via CDF matching.
                if len(phones) == 0:
                    skipped += 1
                    continue
                p_weights = get_ottc_weights(model, processor, audio, device)
                print('p_weights: ', p_weights.shape)
                print(p_weights)
                input()
                spans = ottc_segment(p_weights, len(phones))
                pred_ctm[uttid] = spans_to_ctm(spans, phones, len(audio), HOP_LENGTH, SR)

            else:
                # ── CTC Viterbi forced alignment ─────────────────────────────
                phone_ids = phones_to_ids(phones, processor)
                if phone_ids is None:
                    vocab = processor.tokenizer.get_vocab()
                    unknown = next(
                        (p for p in phones
                         if all(vocab.get(c) is None
                                for c in (p, p.upper(), p.lower(),
                                          p.rstrip("012"), p.rstrip("012").upper(),
                                          p.rstrip("012").lower()))),
                        "?"
                    )
                    logger.warning(f"  [SKIP] unknown phone '{unknown}' in {uttid}")
                    skipped += 1
                    continue
                if len(phone_ids) == 0:
                    skipped += 1
                    continue
                log_probs = get_log_probs(model, processor, audio, device)
                T = log_probs.shape[0]
                if T < 2 * len(phone_ids) + 1:
                    logger.warning(
                        f"  [SKIP] {uttid}: {T} frames < 2×{len(phone_ids)}+1 "
                        f"required for CTC forced alignment"
                    )
                    skipped += 1
                    continue
                spans = ctc_forced_align(log_probs, phone_ids, blank=blank_id)
                pred_ctm[uttid] = spans_to_ctm(spans, phones, len(audio), HOP_LENGTH, SR)

        except Exception as exc:
            logger.error(f"  [ERROR] {uttid}: {exc}")
            skipped += 1
            continue

    logger.info(f"  Aligned {len(pred_ctm)}/{len(utterances)}  ({skipped} skipped)")

    ctm_path = os.path.join(output_dir, "phone_ctm")
    write_ctm(pred_ctm, ctm_path)
    logger.info(f"  Phone CTM → {ctm_path}")

    metrics: dict = {
        "num_utterances": len(utterances),
        "num_aligned":    len(pred_ctm),
    }

    if ref_ctm is not None:
        aligned_utts = sorted(pred_ctm)

        tse_map = compute_tse(ref_ctm, pred_ctm, aligned_utts)
        if tse_map:
            avg_tse = sum(tse_map.values()) / len(tse_map) * 1000  # → ms
            metrics["phone_tse_ms"] = round(avg_tse, 3)
            logger.info(f"  Phone TSE : {avg_tse:.2f} ms  (n={len(tse_map)})")

        metrics["phone_acc"] = {}
        for tau in [10, 20, 25, 30, 40, 50]:
            acc_map = compute_acc(ref_ctm, pred_ctm, aligned_utts, tau)
            if acc_map:
                avg_acc = sum(acc_map.values()) / len(acc_map)
                metrics["phone_acc"][f"tau_{tau}"] = round(avg_acc, 3)
                logger.info(f"  Phone ACC τ={tau:2d} ms : {avg_acc:.2f}%")

    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"  Results → {results_path}")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    BENNEVIS = "/home/jtli/projects/ASR/BenNevis/egs/librispeech/asr3"

    DEFAULT_CORPORA = [
        {
            "name":     "timit_dev",
            "wav_scp":  f"{BENNEVIS}/data_timit/dev/wav.scp",
            "text_phn": f"{BENNEVIS}/data_timit/dev/text.phn",
            "ref_ctm":  f"{BENNEVIS}/data_timit/dev/phone_ctm_new",
        },
        {
            "name":     "timit_test",
            "wav_scp":  f"{BENNEVIS}/data_timit/test/wav.scp",
            "text_phn": f"{BENNEVIS}/data_timit/test/text.phn",
            "ref_ctm":  f"{BENNEVIS}/data_timit/test/phone_ctm_new",
        },
        {
            "name":     "buckeye_train",
            "wav_scp":  f"{BENNEVIS}/data_buckeye/train/wav.scp",
            "text_phn": f"{BENNEVIS}/data_buckeye/train/text.phn",
            "ref_ctm":  f"{BENNEVIS}/data_buckeye/train/phone_ctm_new",
        },
    ]

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # ── Model ────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--checkpoint_dir", type=str, required=True,
        help=(
            "Experiment output directory (contains checkpoint-* sub-folders), "
            "or a specific checkpoint-NNNN folder."
        ),
    )
    parser.add_argument(
        "--encoder_name", type=str, default="wavlm-large",
        help="Encoder name used during training (matches large_models/ subdir).",
    )
    parser.add_argument(
        "--loss_type", type=str, default="ottc", choices=["ctc", "ottc"],
        help="Loss type used during training.",
    )
    parser.add_argument(
        "--processor_dir", type=str, default=None,
        help="Override path to the Wav2Vec2Processor directory (vocab.json).",
    )
    # ── Runtime ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--device", type=str, default=None,
        help="cuda / cpu (autodetected if omitted).",
    )
    parser.add_argument(
        "--gpu", type=int, default=0,
        help="GPU index when --device is not set explicitly.",
    )
    # ── Data ─────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--output_dir", type=str, default="./test_results",
        help="Root output directory; per-corpus sub-dirs are created inside.",
    )
    parser.add_argument(
        "--corpora", type=str, nargs="*", default=None,
        help=(
            "Corpus names to process (default: all three). "
            "Choices: timit_dev timit_test buckeye_train"
        ),
    )
    # ── Custom corpus override ────────────────────────────────────────────────
    parser.add_argument("--wav_scp",  type=str, default=None,
                        help="wav.scp for a single custom corpus.")
    parser.add_argument("--text_phn", type=str, default=None,
                        help="text.phn for a single custom corpus.")
    parser.add_argument("--ref_ctm",  type=str, default=None,
                        help="Reference phone CTM for a single custom corpus.")
    parser.add_argument("--corpus_name", type=str, default="custom",
                        help="Name tag for a single custom corpus.")

    args = parser.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    logger.info(f"Device: {device}")

    # Model
    model, processor = load_model_and_processor(args)
    model.to(device)
    logger.info("Model ready.")

    # Resolve corpus list
    if args.wav_scp:
        # Single custom corpus supplied on CLI
        corpora = [{
            "name":     args.corpus_name,
            "wav_scp":  args.wav_scp,
            "text_phn": args.text_phn,
            "ref_ctm":  args.ref_ctm,
        }]
    else:
        corpora = DEFAULT_CORPORA
        if args.corpora:
            allowed = set(args.corpora)
            corpora = [c for c in corpora if c["name"] in allowed]

    # Run
    all_results: dict = {}
    for corpus in corpora:
        name = corpus["name"]
        logger.info(f"\n{'='*60}\n  {name}\n{'='*60}")

        if not os.path.exists(corpus["wav_scp"]):
            logger.warning(f"[SKIP] wav.scp not found: {corpus['wav_scp']}")
            continue
        if not os.path.exists(corpus["text_phn"]):
            logger.warning(f"[SKIP] text.phn not found: {corpus['text_phn']}")
            continue

        metrics = process_corpus(
            wav_scp_path  = corpus["wav_scp"],
            text_phn_path = corpus["text_phn"],
            ref_ctm_path  = corpus.get("ref_ctm"),
            output_dir    = os.path.join(args.output_dir, name),
            model         = model,
            processor     = processor,
            device        = device,
            loss_type     = args.loss_type,
        )
        all_results[name] = metrics

    # Merged summary JSON
    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nSummary → {summary_path}")

    # Print results table
    print("\n" + "=" * 72)
    print(f"{'Corpus':<20} {'#aligned':>8} {'TSE(ms)':>9} "
          f"{'A@20':>7} {'A@25':>7} {'A@50':>7}")
    print("=" * 72)
    for name, m in all_results.items():
        n_al  = m.get("num_aligned", "?")
        tse   = f"{m['phone_tse_ms']:.1f}" if "phone_tse_ms" in m else "N/A"
        a20   = f"{m['phone_acc']['tau_20']:.1f}" if "phone_acc" in m and "tau_20" in m["phone_acc"] else "N/A"
        a25   = f"{m['phone_acc']['tau_25']:.1f}" if "phone_acc" in m and "tau_25" in m["phone_acc"] else "N/A"
        a50   = f"{m['phone_acc']['tau_50']:.1f}" if "phone_acc" in m and "tau_50" in m["phone_acc"] else "N/A"
        print(f"{name:<20} {n_al:>8} {tse:>9} {a20:>7} {a25:>7} {a50:>7}")
    print("=" * 72)


if __name__ == "__main__":
    main()
