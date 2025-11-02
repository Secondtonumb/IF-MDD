"""
Prepare SpeechOcean762 (original release) into JSON annotations compatible with IF-MDD.

Inputs (Kaldi-style folders under --src_dir):
  - train/{wav.scp, utt2spk, text}
  - test/{wav.scp, utt2spk, text}
  - resource/{scores.json, text-phone, lexicon.txt}
  - WAVE/SPEAKERXXXX/*.WAV

Outputs (under --output_dir):
  - train.json       # full train split
  - train-train.json # 90% of train for training
  - train-dev.json   # 10% of train for validation
  - test.json        # full test split

Canonical/perceived phonemes are normalized to the ARPAbet set used by
exp_l2arctic/EMA_ctc/save/label_encoder.txt, adding leading/trailing 'sil'.

Author: Copilot
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random


try:
    import soundfile as sf
except Exception as e:  # pragma: no cover
    sf = None
try:
    import librosa  # fallback for duration
except Exception:
    librosa = None


# -----------------------------
# Helpers for phoneme handling
# -----------------------------

_PHONEME_MAP = {
    # Vowels
    "aa": "aa", "ae": "ae", "ah": "ah", "ao": "ao", "aw": "aw",
    "ax": "ah",  # map schwa to ah (matches label set)
    "ay": "ay", "eh": "eh", "er": "er", "ey": "ey",
    "ih": "ih", "iy": "iy", "ow": "ow", "oy": "oy", "uh": "uh",
    "uw": "uw",
    # Consonants
    "b": "b", "ch": "ch", "d": "d", "dh": "dh", "f": "f",
    "g": "g", "hh": "hh", "jh": "jh", "k": "k", "l": "l",
    "m": "m", "n": "n", "ng": "ng", "p": "p", "r": "r",
    "s": "s", "sh": "sh", "t": "t", "th": "th", "v": "v",
    "w": "w", "y": "y", "z": "z", "zh": "zh",
    # Specials
    "sil": "sil", "sp": "sil", "spn": "sil",
}


def _strip_bio_token(token: str) -> str:
    """Convert tokens like 'IY0_E' or 'K_B' to plain symbol 'IY0'/'K'."""
    if "_" in token:
        token, _ = token.split("_", 1)
    return token


def _cmu_to_arpabet(sym: str) -> str:
    """Lowercase, drop stress digits, and map to our ARPAbet inventory."""
    # Remove stress digits 0/1/2
    sym = re.sub(r"[012]", "", sym)
    sym = sym.lower()
    return _PHONEME_MAP.get(sym, sym)


def normalize_phoneme_seq(seq: List[str]) -> List[str]:
    """Normalize a list of CMU-ish phones (with possible stress) to target set.

    - Remove BIO suffixes (_B/_I/_E/_S)
    - Remove stress digits
    - Lowercase
    - Map ax->ah, sp/spn->sil
    """
    out: List[str] = []
    for tok in seq:
        base = _strip_bio_token(tok)
        out.append(_cmu_to_arpabet(base))
    return out


def add_silence_boundaries(seq: List[str]) -> List[str]:
    # Deprecated default; kept for optional use via flag
    return ["sil", *seq, "sil"]


def deduplicate_except_sil(seq: List[str]) -> List[str]:
    if not seq:
        return seq
    result = [seq[0]]
    for s in seq[1:]:
        if s == "sil" or s != result[-1]:
            result.append(s)
    return result


# -----------------------------
# Data loading (Kaldi-style)
# -----------------------------

def _read_key_val_tsv(path: Path) -> Dict[str, str]:
    mp: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "\t" in line:
                k, v = line.split("\t", 1)
            else:
                # space separated fallback
                k, v = line.split(None, 1)
            mp[k] = v
    return mp


def _read_text_phone(path: Path) -> Dict[str, List[List[str]]]:
    """Read resource/text-phone.

    Returns a dict: utt_id -> list of word-level phone lists (each element is a
    list of tokens like ['W_B','IY0_E'] for a word). Order is preserved.
    """
    by_utt: Dict[str, List[Tuple[int, List[str]]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            # Format: 000010011.0<TAB>W_B IY0_E
            try:
                left, phones_str = line.split("\t", 1)
            except ValueError:
                continue
            if "." in left:
                utt, idx_str = left.split(".", 1)
                try:
                    idx = int(idx_str)
                except ValueError:
                    idx = 0
            else:
                utt, idx = left, 0
            phones = phones_str.strip().split()
            by_utt[utt].append((idx, phones))
    # sort by idx and strip to phones only
    out: Dict[str, List[List[str]]] = {}
    for utt, items in by_utt.items():
        items_sorted = [p for _, p in sorted(items, key=lambda x: x[0])]
        out[utt] = items_sorted
    return out


def _read_scores(path: Path) -> Dict[str, dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _compute_duration_seconds(wav_path: Path) -> float:
    # Preferred: soundfile.info (fast, reads header only)
    if sf is not None:
        try:
            info = sf.info(str(wav_path))
            if info.samplerate and info.frames:
                return round(info.frames / float(info.samplerate), 3)
        except Exception:
            pass
    # Fallback: librosa
    if librosa is not None:
        try:
            d = librosa.get_duration(path=str(wav_path))
            return round(float(d), 3)
        except Exception:
            pass
    return 0.0


@dataclass
class SplitData:
    wav_scp: Dict[str, str]
    utt2spk: Dict[str, str]
    text: Dict[str, str]


def _load_split(split_dir: Path) -> SplitData:
    return SplitData(
        wav_scp=_read_key_val_tsv(split_dir / "wav.scp"),
        utt2spk=_read_key_val_tsv(split_dir / "utt2spk"),
        text=_read_key_val_tsv(split_dir / "text"),
    )


def _flatten_word_phones(word_phones: List[List[str]]) -> List[str]:
    phones: List[str] = []
    for wp in word_phones:
        phones.extend(wp)
    return phones


def build_json_for_split(
    split: SplitData,
    text_phone: Dict[str, List[List[str]]],
    scores: Dict[str, dict],
    src_root: Path,
    allowed_phones: set[str] | None = None,
    outer_sil: bool = False,
) -> Dict[str, dict]:
    data: Dict[str, dict] = {}
    missing_tp, missing_scores, phone_mismatch = 0, 0, 0

    # Determine absolute WAVE root for joining
    for utt, rel_wav in split.wav_scp.items():
        wav_path = (src_root / rel_wav).resolve()
        spk_id = split.utt2spk.get(utt, "unknown")
        text = split.text.get(utt, "")

        # Build phones from scores.json when available (preferred)
        s_obj = scores.get(utt)
        canonical_seq: List[str] = []
        perceived_seq: List[str] = []
        perceived_sources: List[str] = []  # 'canonical' or 'del' for <del> mapping
        acc_scores: Optional[List[float]] = None
        mispro_events: List[dict] = []

        if s_obj is not None:
            acc_scores = []
            words_list = s_obj.get("words", []) or []
            for w_idx, w in enumerate(words_list):
                w_phones_raw: List[str] = w.get("phones", []) or []
                w_acc = w.get("phones-accuracy", []) or []
                # mispronunciations per word: map index->pronounced-phone
                mis_map: Dict[int, str] = {}
                for m in w.get("mispronunciations", []) or []:
                    try:
                        mi = int(m.get("index", -1))
                    except Exception:
                        mi = -1
                    pr = m.get("pronounced-phone")
                    if mi >= 0 and pr:
                        mis_map[mi] = pr

                # normalize canonical phones for this word
                w_canon_norm = [
                    _cmu_to_arpabet(_strip_bio_token(p)) for p in w_phones_raw
                ]

                # ensure accuracy length
                if len(w_acc) < len(w_canon_norm):
                    w_acc = list(w_acc) + [2.0] * (len(w_canon_norm) - len(w_acc))
                acc_scores.extend([float(x) for x in w_acc[: len(w_canon_norm)]])

                # build perceived using mispron when index present (apply always if provided)
                for p_idx, (c_ph_raw, c_ph, acc) in enumerate(
                    zip(w_phones_raw, w_canon_norm, w_acc)
                ):
                    use_ph = c_ph
                    source = "canonical"
                    if p_idx in mis_map:
                        pronounced_raw = mis_map[p_idx]
                        pronounced_base = _strip_bio_token(pronounced_raw)
                        base_lower = pronounced_base.lower()
                        # special cases: <unk>, * suffix, <del>
                        if base_lower == "<unk>" or base_lower.endswith("*"):
                            use_ph_norm = "err"
                        elif base_lower == "<del>":
                            use_ph_norm = "sil"
                            source = "del"
                        else:
                            use_ph_norm = _cmu_to_arpabet(base_lower)

                        use_ph = use_ph_norm
                        mispro_events.append(
                            {
                                "word_index": w_idx,
                                "phone_index": p_idx,
                                "canonical_phone": c_ph,
                                "pronounced_phone": use_ph_norm,
                                "pronounced_raw": pronounced_raw,
                                "accuracy": float(acc),
                            }
                        )

                    canonical_seq.append(c_ph)
                    perceived_seq.append(use_ph)
                    perceived_sources.append(source)
        else:
            # Fallback to resource/text-phone
            missing_scores += 1
            if utt not in text_phone:
                missing_tp += 1
                word_phones = []
            else:
                word_phones = text_phone[utt]
            flat_tokens = _flatten_word_phones(word_phones)
            canonical_seq = normalize_phoneme_seq(flat_tokens)
            perceived_seq = list(canonical_seq)
            perceived_sources = ["canonical"] * len(perceived_seq)

        # Validate phones
        if allowed_phones is not None:
            bad = [p for p in canonical_seq if p not in allowed_phones]
            if bad:
                phone_mismatch += 1

        # Duration
        duration = _compute_duration_seconds(wav_path)

        # Optionally add outer sil
        if outer_sil:
            canonical_seq = add_silence_boundaries(canonical_seq)
            perceived_seq = add_silence_boundaries(perceived_seq)
            perceived_sources = ["canonical"] + perceived_sources + ["canonical"]

        # train target: remove deletions (<del> mapped to sil) and deduplicate
        perceived_target_seq: List[str] = []
        for ph, src in zip(perceived_seq, perceived_sources):
            if src == "del":
                continue  # skip deletions in train target
            perceived_target_seq.append(ph)
        perceived_target = deduplicate_except_sil(perceived_target_seq)

        entry = {
            "wav": str(wav_path),
            "duration": duration,
            "spk_id": str(spk_id),
            "canonical_aligned": " ".join(canonical_seq),
            "perceived_aligned": " ".join(perceived_seq),
            "perceived_train_target": " ".join(perceived_target),
            "wrd": text,
            "total_score": s_obj.get("total") if s_obj else None,
            "accuracy_score": s_obj.get("accuracy") if s_obj else None,
            "fluency_score": s_obj.get("fluency") if s_obj else None,
            "completeness_score": s_obj.get("completeness") if s_obj else None,
            "prosodic_score": s_obj.get("prosodic") if s_obj else None,
        }
        if acc_scores:
            entry["accuracy_scores"] = acc_scores
        if mispro_events:
            entry["mispronunciations"] = mispro_events

        data[str(wav_path)] = entry

    if missing_tp or missing_scores or phone_mismatch:
        print(
            f"[build_json] missing text-phone: {missing_tp}, missing scores: {missing_scores}, "
            f"phone mismatch (per-utt): {phone_mismatch}"
        )
    return data


def read_allowed_phones_from_label_encoder(path: Path) -> set[str]:
    allowed: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("==="):
                continue
            if "=>" in line:
                # format: 'zh' => 35
                k, _ = line.split("=>", 1)
                k = k.strip().strip("'")
                allowed.add(k)
    return allowed


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Prepare SpeechOcean762 (original release)")
    parser.add_argument("--src_dir", type=str, default="./speechocean762", help="Root of original dataset")
    parser.add_argument("--output_dir", type=str, default="./data/speechocean762_orig", help="Where to write JSONs")
    parser.add_argument("--label_encoder", type=str,
                        default="./exp_l2arctic/EMA_ctc/save/label_encoder.txt",
                        help="Label encoder to validate phoneme inventory")
    parser.add_argument("--train_dev_ratio", type=float, default=0.9, help="Train/dev split ratio for train.json")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for speaker-balanced split")
    parser.add_argument("--outer_sil", action="store_true", help="If set, add sil at start and end of sequences")
    args = parser.parse_args()

    src_root = Path(args.src_dir).resolve()
    out_root = Path(args.output_dir).resolve()

    # Read inventories and resources
    print(f"Reading resources from: {src_root}")
    text_phone = _read_text_phone(src_root / "resource" / "text-phone")
    scores = _read_scores(src_root / "resource" / "scores.json")
    allowed = read_allowed_phones_from_label_encoder(Path(args.label_encoder).resolve())

    # Load splits
    train_split = _load_split(src_root / "train")
    test_split = _load_split(src_root / "test")

    # Build JSONs
    print("Building train.json ...")
    train_json = build_json_for_split(train_split, text_phone, scores, src_root, allowed, outer_sil=args.outer_sil)
    print("Building test.json ...")
    test_json = build_json_for_split(test_split, text_phone, scores, src_root, allowed, outer_sil=args.outer_sil)

    # Save full files
    save_json(train_json, out_root / "train.json")
    save_json(test_json, out_root / "test.json")

    # Derive train-train.json and train-dev.json
    # Perform a random speaker-balanced split: group utterances by speaker,
    # shuffle speakers with a seed and assign whole speakers to train until
    # the accumulated utterance count reaches the target proportion. This
    # guarantees no speaker overlap between train and dev.
    random.seed(args.seed)
    spk2utts: Dict[str, List[str]] = defaultdict(list)
    for utt_key, entry in train_json.items():
        spk = entry.get("spk_id", "unknown")
        spk2utts[spk].append(utt_key)

    speakers = list(spk2utts.keys())
    random.shuffle(speakers)

    total_utts = len(train_json)
    target_train_utts = int(total_utts * args.train_dev_ratio)

    train_train: Dict[str, dict] = {}
    train_dev: Dict[str, dict] = {}
    acc = 0
    for spk in speakers:
        utts = spk2utts[spk]
        # If we haven't reached the target number of train utterances,
        # put this entire speaker into train; otherwise into dev.
        if acc < target_train_utts:
            for u in utts:
                train_train[u] = train_json[u]
            acc += len(utts)
        else:
            for u in utts:
                train_dev[u] = train_json[u]

    # Edge case: if train_dev is empty (e.g., ratio==1.0), move the last
    # speaker from train to dev to ensure dev is non-empty.
    if len(train_dev) == 0 and len(speakers) > 1:
        last_spk = speakers[-1]
        for u in spk2utts[last_spk]:
            train_dev[u] = train_train.pop(u, train_json[u])

    save_json(train_train, out_root / "train-train.json")
    save_json(train_dev, out_root / "train-dev.json")

    # Quick inventory check
    ph_set = set()
    for v in train_json.values():
        ph_set.update(v["canonical_aligned"].split())
    unknown = sorted([p for p in ph_set if p not in allowed])
    if unknown:
        print(f"⚠️ Unknown phones wrt label encoder: {unknown}")
    else:
        print("Phoneme inventory validated against label encoder.")

    print(f"\n✅ Done. Wrote JSONs to: {out_root}")
    print("Update hparams if needed, e.g.,")
    print(f"  data_folder_save: \"{out_root}\"")
    print(f"  train_annotation: \"{out_root}/train-train.json\"")
    print(f"  valid_annotation: \"{out_root}/train-dev.json\"")
    print(f"  test_annotation:  \"{out_root}/test.json\"")


if __name__ == "__main__":
    main()
