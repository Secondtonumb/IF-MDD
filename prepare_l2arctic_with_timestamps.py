"""
Prepare L2-ARCTIC into JSON annotations with phoneme-level timestamps for IF-MDD.

This generator matches the expected schema used by TimestampDataIOPrep:
- wav: absolute path to audio file
- duration: seconds (float)
- spk_id: speaker id (folder name)
- wrd: transcript text (string)
- canonical_aligned: space-separated canonical phones (aligned form)
- perceived_aligned: space-separated perceived phones (aligned form)
- perceived_train_target: space-separated perceived phones for training target
- canonical_starts: list[float] per-canonical phone start times (seconds)
- canonical_ends: list[float] per-canonical phone end times (seconds)
- target_starts: list[float] per-target phone start times (seconds)
- target_ends: list[float] per-target phone end times (seconds)

Inputs (L2-ARCTIC layout):
  <SRC_DIR>/<SPK>/wav/*.wav
  <SRC_DIR>/<SPK>/annotation/*.TextGrid
  <SRC_DIR>/<SPK>/transcript/*.txt

Notes:
- We follow the old script's normalization rules for L2-ARCTIC, including
  handling of the artificial "sil" and annotation tags (a/s/d) in marks.
- "aligned" strings preserve artificial silences to maintain 1-1 canonical vs
  perceived alignment; training targets remove artificial/repetitive sil.

Example:
  python prepare_l2arctic_with_timestamps.py \
      --src_dir /path/to/L2-ARCTIC \
      --output_dir ./data/l2arctic_ts \
      --test_spks TLV NJS TNI TXHC ZHAA YKWK
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
import re
import copy

try:
    import soundfile as sf
except Exception:
    sf = None

try:
    import librosa
except Exception:
    librosa = None

from textgrid import TextGrid, IntervalTier

# -----------------------------
# Helpers
# -----------------------------

ARPABET_PATH_DEFAULT = "utils/arpa_phonemes"


def _read_arpa_phonemes(path: Path) -> List[str]:
    if not path.exists():
        # Fallback common set; not strictly needed for generation
        return [
            "aa","ae","ah","ao","aw","ay","eh","er","ey","ih","iy","ow","oy","uh","uw",
            "b","ch","d","dh","f","g","hh","jh","k","l","m","n","ng","p","r","s","sh","t","th","v","w","y","z","zh","sil","err"
        ]
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tok = line.split()[0]
            out.append(tok)
    return out


def is_sil(s: str) -> bool:
    return s.lower() in {"sil", "sp", "spn", "pau", ""}


def normalize_phone(
    s: str,
    is_rm_annotation: bool = True,
    is_phoneme_canonical: bool = True,
    keep_artificial_sil: bool = False,
) -> Optional[str]:
    """Normalize L2-ARCTIC phone mark into our target inventory.

    L2-ARCTIC phone marks can be like "AH,AX,A" or "B,B,S" where the third tag is
    an annotation code: 'a' (artificial sil), 's' (substitution), 'd' (deletion).

    Behavior mirrors the legacy script:
    - Return "sil" for silence-like labels.
    - If only one token, use it (mapping ax->ah).
    - If is_rm_annotation:
        * keep_artificial_sil=True: choose canonical (idx0) or perceived (idx1)
        * keep_artificial_sil=False: drop artificial sil and deletions depending on mode
    - If not is_rm_annotation: keep annotations; we still lowercase and map ax->ah.
    """
    t = s.lower()
    # Keep only letters and commas
    t = re.sub(r"[^a-z,]", "", t)
    if is_sil(t):
        return "sil"
    if len(t) == 0:
        raise ValueError(f"Invalid phone mark: {s}")

    parts = t.split(",")
    if len(parts) == 1:
        p0 = parts[0]
        if p0 == "ax":
            return "ah"
        return p0

    # parts has at least 2 (canonical, perceived [, tag])
    cano = parts[0]
    perc = parts[1] if len(parts) >= 2 else parts[0]
    tag = parts[2] if len(parts) >= 3 else ""

    if is_rm_annotation:
        if keep_artificial_sil:
            # Keep alignment 1-1 between canonical and perceived
            out = cano if is_phoneme_canonical else perc
            return "ah" if out == "ax" else out
        else:
            if is_phoneme_canonical:
                if tag in {"s", "d"}:  # keep canonical
                    out = cano
                elif tag == "a":        # artificial sil -> drop
                    return None
                else:
                    out = cano
            else:
                if tag in {"s", "a"}:  # keep perceived
                    out = perc
                elif tag == "d":        # deletion -> drop
                    return None
                else:
                    out = perc
            return "ah" if out == "ax" else out
    else:
        # Keep annotations (seldom used in this project); still map ax->ah
        if parts[0] == "ax":
            return "ah"
        return parts[0]


def normalize_word(s: str) -> str:
    return s.strip()


def normalize_tier_mark(
    tier: IntervalTier,
    mode: str = "NormalizePhoneCanonical",
    keep_artificial_sil: bool = False,
) -> IntervalTier:
    tier = copy.deepcopy(tier)
    tier_out = IntervalTier(name=tier.name, minTime=tier.minTime, maxTime=tier.maxTime)

    if mode not in {
        "NormalizePhoneCanonical",
        "NormalizePhonePerceived",
        "NormalizePhoneAnnotation",
        "NormalizeWord",
    }:
        raise ValueError(f"Invalid mode: {mode}")

    for itv in tier.intervals:
        if mode == "NormalizePhoneCanonical":
            p = normalize_phone(itv.mark, True, True, keep_artificial_sil)
        elif mode == "NormalizePhonePerceived":
            p = normalize_phone(itv.mark, True, False, keep_artificial_sil)
        elif mode == "NormalizePhoneAnnotation":
            p = normalize_phone(itv.mark, False)
        elif mode == "NormalizeWord":
            p = normalize_word(itv.mark)
        else:
            p = None
        if p is None:
            continue
        if p == "ax":
            p = "ah"
        itv_new = copy.deepcopy(itv)
        itv_new.mark = p
        tier_out.addInterval(itv_new)
    return tier_out


def _tier_to_list(tier: IntervalTier) -> List[str]:
    return [interval.mark for interval in tier]


def _tier_to_time_tuples(tier: IntervalTier) -> List[Tuple[float, float, str]]:
    return [(interval.minTime, interval.maxTime, interval.mark) for interval in tier.intervals]


def _remove_repetitive_sil(seq: List[str]) -> List[str]:
    if not seq:
        return seq
    out = [seq[0]]
    for s in seq[1:]:
        if s == "sil" and out[-1] == "sil":
            continue
        out.append(s)
    return out


def _remove_repetitive_sil_intervals(itvs: List[Tuple[float, float, str]]) -> List[Tuple[float, float, str]]:
    out: List[Tuple[float, float, str]] = []
    for s, e, m in itvs:
        if m == "sil" and out and out[-1][2] == "sil":
            # merge with previous sil (extend end time)
            ps, pe, pm = out[-1]
            out[-1] = (ps, max(pe, e), pm)
        else:
            out.append((s, e, m))
    return out


def _compute_duration_seconds(wav_path: Path) -> float:
    if sf is not None:
        try:
            info = sf.info(str(wav_path))
            if info.samplerate and info.frames:
                return round(info.frames / float(info.samplerate), 3)
        except Exception:
            pass
    if librosa is not None:
        try:
            d = librosa.get_duration(path=str(wav_path))
            return round(float(d), 3)
        except Exception:
            pass
    return 0.0


# -----------------------------
# Core builders
# -----------------------------

def get_phoneme_strings(
    tg: TextGrid,
    keep_artificial_sil: bool,
    rm_repetitive_sil: bool,
) -> Tuple[str, str]:
    phone_tier: IntervalTier = tg.getFirst("phones")
    perc_tier = normalize_tier_mark(phone_tier, "NormalizePhonePerceived", keep_artificial_sil)
    cano_tier = normalize_tier_mark(phone_tier, "NormalizePhoneCanonical", keep_artificial_sil)
    perc_list = _tier_to_list(perc_tier)
    cano_list = _tier_to_list(cano_tier)
    if rm_repetitive_sil:
        perc_list = _remove_repetitive_sil(perc_list)
        cano_list = _remove_repetitive_sil(cano_list)
    return " ".join(cano_list), " ".join(perc_list)


def get_phoneme_intervals(
    tg: TextGrid,
    keep_artificial_sil: bool,
    rm_repetitive_sil: bool,
) -> Tuple[List[Tuple[float, float, str]], List[Tuple[float, float, str]]]:
    phone_tier: IntervalTier = tg.getFirst("phones")
    cano_tier = normalize_tier_mark(phone_tier, "NormalizePhoneCanonical", keep_artificial_sil)
    perc_tier = normalize_tier_mark(phone_tier, "NormalizePhonePerceived", keep_artificial_sil)
    cano_itvs = _tier_to_time_tuples(cano_tier)
    perc_itvs = _tier_to_time_tuples(perc_tier)
    if rm_repetitive_sil:
        cano_itvs = _remove_repetitive_sil_intervals(cano_itvs)
        perc_itvs = _remove_repetitive_sil_intervals(perc_itvs)
    return cano_itvs, perc_itvs


# -----------------------------
# Dataset builders
# -----------------------------

def _load_text(text_file: Path) -> str:
    try:
        return text_file.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def make_json_for_speakers(src_root: Path, spks: List[str]) -> Dict[str, dict]:
    json_data: Dict[str, dict] = defaultdict(dict)
    for spk in spks:
        wav_dir = src_root / spk / "wav"
        tg_dir = src_root / spk / "annotation"
        txt_dir = src_root / spk / "transcript"
        for tg_path in sorted(tg_dir.glob("*.TextGrid")):
            try:
                tg = TextGrid(); tg.read(str(tg_path))
            except Exception:
                continue
            base = tg_path.stem
            wav_path = wav_dir / f"{base}.wav"
            txt_path = txt_dir / f"{base}.txt"
            if not wav_path.exists():
                continue
            # Core strings: aligned (keep artificial sil, keep repeats)
            cano_aligned, perc_aligned = get_phoneme_strings(
                tg, keep_artificial_sil=True, rm_repetitive_sil=False
            )
            # Training target: remove artificial sil + collapse repeats
            _cano_target, perc_target = get_phoneme_strings(
                tg, keep_artificial_sil=False, rm_repetitive_sil=True
            )
            # Intervals for canonical/target with the same respective settings
            cano_itvs, _ = get_phoneme_intervals(
                tg, keep_artificial_sil=False, rm_repetitive_sil=True
            )
            _, target_itvs = get_phoneme_intervals(
                tg, keep_artificial_sil=False, rm_repetitive_sil=True
            )
            # Assemble
            duration = _compute_duration_seconds(wav_path)
            entry = {
                "wav": str(wav_path.resolve()),
                "duration": duration,
                "spk_id": spk,
                "canonical_aligned": cano_aligned,
                "perceived_aligned": perc_aligned,
                "perceived_train_target": perc_target,
                "wrd": _load_text(txt_path),
                "canonical_starts": [round(s, 4) for s, e, _ in cano_itvs],
                "canonical_ends": [round(e, 4) for s, e, _ in cano_itvs],
                "target_starts": [round(s, 4) for s, e, _ in target_itvs],
                "target_ends": [round(e, 4) for s, e, _ in target_itvs],
            }
            json_data[str(wav_path.resolve())] = entry
    return json_data


def speaker_splits_from_metadata(meta_path: Path, test_spks: Optional[List[str]]) -> Tuple[List[str], List[str]]:
    total_spks: List[str] = []
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                parts = line.strip().split()
                if not parts:
                    continue
                total_spks.append(parts[0])
    # Fallback: infer speakers from folders
    if not total_spks:
        total_spks = [p.name for p in Path(meta_path).parent.glob("*") if p.is_dir()]

    test_spks = test_spks or ["TLV", "NJS", "TNI", "TXHC", "ZHAA", "YKWK"]
    train_spks = [s for s in total_spks if s not in set(test_spks)]
    return train_spks, test_spks


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Prepare L2-ARCTIC with phoneme timestamps for IF-MDD")
    parser.add_argument("--src_dir", type=str, required=True, help="Root of L2-ARCTIC")
    parser.add_argument("--output_dir", type=str, default="./data/l2arctic_ts", help="Output directory")
    parser.add_argument("--metadata_l2arctic", type=str, default="data/metadata_l2arctic", help="Speaker metadata file (name dialect gender)")
    parser.add_argument("--test_spks", type=str, nargs="*", default=None, help="Explicit test speaker IDs")
    parser.add_argument("--arpa_phonemes", type=str, default=ARPABET_PATH_DEFAULT, help="Path to ARPA phoneme list (optional)")
    args = parser.parse_args()

    src_root = Path(args.src_dir).resolve()
    out_root = Path(args.output_dir).resolve()

    # Load ARPA list (not strictly necessary here but validates normalization if needed)
    _ = _read_arpa_phonemes(Path(args.arpa_phonemes))

    train_spks, test_spks = speaker_splits_from_metadata(Path(args.metadata_l2arctic), args.test_spks)

    print(f"Building train set for {len(train_spks)} speakers -> {out_root}")
    train_json = make_json_for_speakers(src_root, train_spks)
    save_json(train_json, out_root / "train.json")

    print(f"Building test set for {len(test_spks)} speakers -> {out_root}")
    test_json = make_json_for_speakers(src_root, test_spks)
    save_json(test_json, out_root / "test.json")

    # Optional: split train into train-train/dev by speakers (keep whole speakers together)
    # Here we simply reuse the same split by default; downstream can create dev as needed.
    # If you want a dev set, you can pass a subset of train speakers as test_spks and run twice.

    print("\n✅ Done. JSONs with timestamps written to:")
    print(f"  {out_root}/train.json")
    print(f"  {out_root}/test.json")
    print("\nUse these with TimestampDataIOPrep by pointing:")
    print(f"  data_folder_save: \"{out_root}\"")
    print(f"  train_annotation: \"{out_root}/train.json\"")
    print(f"  valid_annotation: \"{out_root}/test.json\"  # or provide your own dev split")
    print(f"  test_annotation:  \"{out_root}/test.json\"")


if __name__ == "__main__":
    main()
