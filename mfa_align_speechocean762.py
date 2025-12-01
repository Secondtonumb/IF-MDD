"""
Run Montreal Forced Aligner on SpeechOcean762 and export word/phoneme timestamps into JSON.

- Reads one or more existing SO762 JSONs (e.g., data/speechocean762_with_word_scores/test.json)
- Builds an MFA corpus folder with symlinked wavs and .lab transcripts
- Runs MFA alignment using english_us_arpa (configurable)
- Parses TextGrids to extract word and phone tiers
- Outputs a new JSON with added timestamp fields:
    - mfa_phone_aligned, mfa_phone_starts, mfa_phone_ends
    - mfa_word_aligned,  mfa_word_starts,  mfa_word_ends
  Additionally mirrors phone timestamps into canonical_starts/canonical_ends for downstream pipelines

Usage (example):
  python mfa_align_speechocean762.py \
    --input-json data/speechocean762_with_word_scores/test.json \
    --output-json data/speechocean762_with_word_scores/test_with_mfa.json \
    --corpus-dir data/so762_mfa_corpus \
    --mfa-output-dir data/so762_mfa_textgrids \
    --jobs 8 --limit 50

Notes:
- Requires `mfa` CLI installed and accessible on PATH.
- Default dictionary/acoustic_model: english_us_arpa (ARPA phoneme set).
- By default, removes stress digits (e.g., ao1 -> ao) and filters out sil/sp/spn from phones.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

from collections import OrderedDict

try:
    from textgrid import TextGrid
except ImportError:
    raise SystemExit("Please install textgrid: pip install textgrid")


ARPA_SIL_SET = {"sil", "sp", "spn", "pau", ""}


def remove_stress(p: str) -> str:
    return re.sub(r"\d+", "", p)


def is_sil(p: str) -> bool:
    return p.lower() in ARPA_SIL_SET


def build_corpus_from_jsons(json_paths: List[Path], corpus_dir: Path, limit: int | None = None) -> List[Tuple[Path, Path, str]]:
    """Create MFA corpus folder with speaker subfolders and .lab files.
    Returns list of tuples: (wav_symlink, lab_path, utt_id) in the same order of processing.
    The utt_id is the basename without extension.
    """
    if corpus_dir.exists():
        shutil.rmtree(corpus_dir)
    corpus_dir.mkdir(parents=True, exist_ok=True)
    # import pdb; pdb.set_trace()
    items: List[Tuple[Path, Path, str]] = []
    count = 0
    for jp in json_paths:
        data = json.load(open(jp))
        for key, obj in data.items():
            wav = Path(obj["wav"]) if "wav" in obj else Path(key)
            
            # if not wav.exists():
            #     # Skip missing audio
            #     continue
            # speaker id: prefer explicit spk_id, else infer from path
            spk = str(obj.get("spk_id") or wav.parent.name or "SPK")
            text = str(obj.get("wrd", "")).strip()
            if not text:
                continue

            # Keep folder structure: corpus_dir/SPK/<utt>.WAV and .lab
            spk_dir = corpus_dir / f"{spk}"
            spk_dir.mkdir(parents=True, exist_ok=True)

            utt_id = wav.stem  # e.g., 000030012
            wav_link = spk_dir / f"{utt_id}{wav.suffix}"
            lab_path = spk_dir / f"{utt_id}.lab"
            # Create symlink for wav (fallback to copy if symlink fails)
    
            try:
                if wav_link.exists():
                    wav_link.unlink()
                os.symlink(wav, wav_link)
            except Exception:
                shutil.copy2(wav, wav_link)
            
            # Write transcript
            lab_path.write_text(text + "\n", encoding="utf-8")

            items.append((wav_link, lab_path, utt_id))
            count += 1
            if limit is not None and count >= limit:
                return items
        
    return items


def run_mfa_align(corpus_dir: Path, output_dir: Path, dictionary: str, acoustic_model: str, jobs: int = 4) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "mfa", "align",
        str(corpus_dir),
        dictionary,
        acoustic_model,
        str(output_dir),
        "-j", str(jobs),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_textgrid(tg_path: Path, remove_sil: bool = True) -> Dict:
    tg = TextGrid.fromFile(str(tg_path))

    # Find word and phone tiers
    word_tier = None
    phone_tier = None
    for tier in tg.tiers:
        name = tier.name.lower()
        if word_tier is None and name in {"words", "word"}:
            word_tier = tier
        if phone_tier is None and name in {"phones", "phone"}:
            phone_tier = tier
    if phone_tier is None:
        raise ValueError(f"No phone tier in {tg_path}")

    # Phones
    phones: List[str] = []
    p_starts: List[float] = []
    p_ends: List[float] = []
    for it in phone_tier:
        ph = remove_stress(str(it.mark).strip().lower())
        if not ph:
            continue
        if remove_sil and is_sil(ph):
            continue
        phones.append(ph)
        p_starts.append(float(it.minTime))
        p_ends.append(float(it.maxTime))

    # Words (optional tier)
    words: List[str] = []
    w_starts: List[float] = []
    w_ends: List[float] = []
    if word_tier is not None:
        for it in word_tier:
            w = str(it.mark).strip()
            if not w:
                continue
            # MFA sometimes uses <UNK> or silences; filter empties
            if remove_sil and w.lower() in ARPA_SIL_SET:
                continue
            words.append(w)
            w_starts.append(float(it.minTime))
            w_ends.append(float(it.maxTime))
    # Word -> phone span index mapping (inclusive start, exclusive end like python slicing)
    word_phone_start_idx: List[int] = []
    word_phone_end_idx: List[int] = []
    if words:
        # For each word interval, find phones whose midpoint lies within the word boundaries
        for ws, we in zip(w_starts, w_ends):
            # Collect indices of phones overlapping [ws, we]
            indices = [i for i, (ps, pe) in enumerate(zip(p_starts, p_ends)) if not (pe <= ws or ps >= we)]
            if indices:
                word_phone_start_idx.append(indices[0])
                word_phone_end_idx.append(indices[-1] + 1)  # exclusive end
            else:
                # No phones found (edge case), duplicate previous or set to last end
                last_end = word_phone_end_idx[-1] if word_phone_end_idx else 0
                word_phone_start_idx.append(last_end)
                word_phone_end_idx.append(last_end)

    return {
        "mfa_phone_aligned": " ".join(phones),
        "mfa_phone_starts": [round(x, 3) for x in p_starts],
        "mfa_phone_ends": [round(x, 3) for x in p_ends],
        "mfa_word_aligned": " ".join(words) if words else None,
        "mfa_word_starts": [round(x, 3) for x in w_starts] if words else None,
        "mfa_word_ends": [round(x, 3) for x in w_ends] if words else None,
        "mfa_word_phone_start_idx": word_phone_start_idx if words else None,
        "mfa_word_phone_end_idx": word_phone_end_idx if words else None,
    }


def integrate_mfa_into_json(input_json: Path, textgrid_root: Path, output_json: Path, remove_sil: bool = True) -> None:
    """For each entry in input_json, locate its TextGrid, parse, and attach MFA fields.
    Saves an updated JSON to output_json.
    """
    data = json.load(open(input_json))
    out: Dict[str, Dict] = OrderedDict()

    # Build map of available grids for quick lookup
    # Expect path: textgrid_root/SPEAKER/UTT.TextGrid
    available = {}
    for spk_dir in textgrid_root.glob("*"):
        if not spk_dir.is_dir():
            continue
        for tg in spk_dir.glob("*.TextGrid"):
            available[(spk_dir.name, tg.stem)] = tg

    missing = 0
    for key, obj in data.items():
        wav = Path(obj.get("wav", key))
        spk = str(obj.get("spk_id") or wav.parent.name)
        utt = wav.stem
        res = obj.copy()

        tg_path = available.get((spk, utt))
        if tg_path and tg_path.exists():
            parsed = parse_textgrid(tg_path, remove_sil=remove_sil)
            res.update(parsed)
            # Mirror canonical_* for downstream pipelines
            if parsed.get("mfa_phone_starts"):
                res["canonical_starts"] = parsed["mfa_phone_starts"]
                res["canonical_ends"] = parsed["mfa_phone_ends"]
        else:
            missing += 1
        out[key] = res

    output_json.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(output_json, "w"), ensure_ascii=False, indent=2)
    print(f"Saved: {output_json} (missing grids: {missing})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-json", nargs="+", required=True, help="Input SO762 JSON file(s)")
    ap.add_argument("--output-json", required=True, help="Output JSON with MFA timestamps")
    ap.add_argument("--corpus-dir", default="data/so762_mfa_corpus", help="Staging corpus dir for MFA")
    ap.add_argument("--mfa-output-dir", default="data/so762_mfa_textgrids", help="MFA TextGrid output dir")
    ap.add_argument("--dictionary", default="english_us_arpa", help="MFA dictionary name or path")
    ap.add_argument("--acoustic-model", default="english_us_arpa", help="MFA acoustic model name or path")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="Limit number of utterances for quick run; 0 for all")
    ap.add_argument("--skip-align", action="store_true", help="Skip MFA alignment step and only parse existing TextGrids")
    ap.add_argument("--keep-sil", action="store_true", help="Keep sil/sp/spn phones in output")
    args = ap.parse_args()

    input_jsons = [Path(p) for p in args.input_json]
    corpus_dir = Path(args.corpus_dir)
    textgrid_dir = Path(args.mfa_output_dir)
    output_json = Path(args.output_json)

    limit = args.limit if args.limit and args.limit > 0 else None

    # Build corpus
    items = build_corpus_from_jsons(input_jsons, corpus_dir, limit=limit)
    if not items:
        raise SystemExit("No utterances prepared. Check input JSON paths.")

    # Run MFA
    if not args.skip_align:
        run_mfa_align(corpus_dir, textgrid_dir, args.dictionary, args.acoustic_model, jobs=args.jobs)

    # Integrate back to JSON
    # For now, process only the first input_json to produce a matching output
    integrate_mfa_into_json(input_jsons[0], textgrid_dir, output_json, remove_sil=not args.keep_sil)


if __name__ == "__main__":
    main()
