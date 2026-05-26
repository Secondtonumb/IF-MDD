#!/usr/bin/env python3
"""Create integrated LibriSpeech training/eval scripts for Buckeye/TIMIT tests."""

from __future__ import annotations

import argparse
import stat
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LOSSES = ["ctc", "crctc", "ottc", "crottc"]
SOURCE_HPARAM_DIR = "hparam_librispeech"
TARGET_HPARAM_DIR = "hparam_librispeech_integrated"
EXP_ROOT = "exp_librispeech_small_integrated"

EVAL_DATASETS = {
    "librispeech_test": "data/librispeech_alignments/json/test.json",
    "librispeech_test_clean": "data/librispeech_alignments/json/test_clean.json",
    "librispeech_test_other": "data/librispeech_alignments/json/test_other.json",
    "timit_dev": "data/timit_alignments_39phone/json/dev.json",
    "timit_test": "data/timit_alignments_39phone/json/test.json",
    "buckeye": "data/buckeye_alignments_39phone/json/train_dev_test.json",
}


def rewrite_hparam(text: str) -> str:
    text = text.replace("wandb_project: \"librispeech_small\"", "wandb_project: \"librispeech_small_integrated\"")
    text = text.replace("exp_librispeech_small/", f"{EXP_ROOT}/")

    out = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# test_annotation:"):
            continue
        if stripped.startswith("test_annotation:"):
            out.append("test_annotation: !ref <data_folder_save>/test.json")
        else:
            out.append(line)
    return "\n".join(out) + "\n"


def write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)


def train_single_script(loss: str) -> str:
    hparam = f"{TARGET_HPARAM_DIR}/phnmonossl_{loss}.yaml"
    return f"""#!/usr/bin/env bash
#------ qsub option --------#
#PBS -q regular-g
#PBS -l select=1:mpiprocs=4
#PBS -l walltime=24:00:00
#PBS -W group_list=gm64
#PBS -j oe

set -e -o pipefail

source ~/.bashrc
cd /home/m64000/work/IF-MDD
conda activate sb_k2

DEVICE="${{DEVICE:-cuda}}"
BATCH_SIZE="${{BATCH_SIZE:-24}}"
EPOCHS="${{EPOCHS:-100}}"

python train.py {hparam} \\
  --mode train+eval \\
  --device="${{DEVICE}}" \\
  --batch_size="${{BATCH_SIZE}}" \\
  --number_of_epochs="${{EPOCHS}}"
"""


def train_group_script() -> str:
    configs = " \\\n  ".join(f"{TARGET_HPARAM_DIR}/phnmonossl_{loss}.yaml" for loss in LOSSES)
    return f"""#!/usr/bin/env bash
#------ qsub option --------#
#PBS -q regular-g
#PBS -l select=1:mpiprocs=4
#PBS -l walltime=24:00:00
#PBS -W group_list=gm64
#PBS -j oe

set -e -o pipefail

source ~/.bashrc
cd /home/m64000/work/IF-MDD
conda activate sb_k2

DEVICE="${{DEVICE:-cuda}}"
BATCH_SIZE="${{BATCH_SIZE:-24}}"
EPOCHS="${{EPOCHS:-100}}"

for CONFIG in \\
  {configs}
do
  python train.py "${{CONFIG}}" \\
    --mode train+eval \\
    --device="${{DEVICE}}" \\
    --batch_size="${{BATCH_SIZE}}" \\
    --number_of_epochs="${{EPOCHS}}"
done
"""


def eval_script() -> str:
    dataset_lines = "\n".join(f"  {name}={path}" for name, path in EVAL_DATASETS.items())
    configs = " ".join(f"{TARGET_HPARAM_DIR}/phnmonossl_{loss}.yaml" for loss in LOSSES)
    return f"""#!/usr/bin/env bash
#------ qsub option --------#
#PBS -q regular-g
#PBS -l select=1:mpiprocs=4
#PBS -l walltime=08:00:00
#PBS -W group_list=gm64
#PBS -j oe

set -e -o pipefail

source ~/.bashrc
cd /home/m64000/work/IF-MDD
conda activate sb_k2

DEVICE="${{DEVICE:-cuda}}"
CONFIGS="${{CONFIGS:-{configs}}}"
DATASETS="${{DATASETS:-librispeech_test_clean librispeech_test_other timit_dev timit_test buckeye}}"

declare -A ANNOTATIONS=(
{dataset_lines}
)

for CONFIG in ${{CONFIGS}}; do
  for DATASET in ${{DATASETS}}; do
    ANNOTATION="${{ANNOTATIONS[$DATASET]:-}}"
    if [ -z "${{ANNOTATION}}" ]; then
      echo "Unknown DATASET=${{DATASET}}"
      exit 1
    fi
    echo "============================================================"
    echo "Eval CONFIG=${{CONFIG}} DATASET=${{DATASET}}"
    echo "Annotation=${{ANNOTATION}}"
    echo "============================================================"
    python train.py "${{CONFIG}}" \\
      --mode eval \\
      --device="${{DEVICE}}" \\
      --test_annotation "${{ANNOTATION}}"
  done
done
"""


def fa_script() -> str:
    return """#!/usr/bin/env bash
#------ qsub option --------#
#PBS -q regular-g
#PBS -l select=1:mpiprocs=4
#PBS -l walltime=24:00:00
#PBS -W group_list=gm64
#PBS -j oe

set -e -o pipefail

source ~/.bashrc
cd /home/m64000/work/IF-MDD
conda activate sb_k2

DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LIMIT="${LIMIT:-0}"
FA_BACKEND="${FA_BACKEND:-native-k2}"
BLANK_POLICY="${BLANK_POLICY:-previous}"
OUT_DIR="${OUT_DIR:-fa_research/results/librispeech_buckeye_timit_integrated_fa}"
FIG_DIR="${FIG_DIR:-fa_research/figures/librispeech_buckeye_timit_integrated_fa}"
DATASETS="${DATASETS:-timit_dev timit_test buckeye}"
MODELS="${MODELS:-LIBRISPEECH_CTC LIBRISPEECH_OTTC LIBRISPEECH_CRCTC LIBRISPEECH_CROTTC}"

# Optional filters, e.g.:
#   DATASETS="timit_test buckeye" MODELS="LIBRISPEECH_CTC LIBRISPEECH_CROTTC" qsub ...
# Dataset aliases also accept explicit name=/path values supported by evaluate_json_alignment_fa.py.
read -r -a DATASET_ARGS <<< "${DATASETS}"
read -r -a MODEL_ARGS <<< "${MODELS}"

python fa_research/scripts/evaluate_json_alignment_fa.py \\
  --datasets "${DATASET_ARGS[@]}" \\
  --models "${MODEL_ARGS[@]}" \\
  --output-dir "${OUT_DIR}" \\
  --figures-dir "${FIG_DIR}" \\
  --batch-size "${BATCH_SIZE}" \\
  --device "${DEVICE}" \\
  --limit "${LIMIT}" \\
  --fa-backend "${FA_BACKEND}" \\
  --blank-policy "${BLANK_POLICY}" \\
  --tolerances 0.01 0.02 0.025 0.03 0.04 0.05 \\
  --save-alignments

python fa_research/scripts/merge_alignment_shards.py \\
  --shards-dir "${OUT_DIR}/alignment_shards" \\
  --output "${OUT_DIR}/alignments.jsonl" \\
  --datasets "${DATASET_ARGS[@]}" \\
  --models "${MODEL_ARGS[@]}"

python fa_research/scripts/recompute_json_fa_run_test_style_metrics.py \\
  --alignments "${OUT_DIR}/alignments.jsonl" \\
  --output-dir "${OUT_DIR}" \\
  --tolerances-ms 10 20 25 30 40 50

python fa_research/scripts/plot_cross_domain_run_test_style_matrix_svg.py \\
  --matrix "${OUT_DIR}/run_test_style_matrix.csv" \\
  --output "${FIG_DIR}/run_test_style_matrix_paper_like.svg" \\
  --column

if printf '%s\\n' "${DATASET_ARGS[@]}" | grep -qx buckeye; then
  python fa_research/scripts/compare_buckeye_with_li_table3.py \\
    --matrix "${OUT_DIR}/run_test_style_matrix.csv" \\
    --output-csv "${OUT_DIR}/buckeye_li_table3_comparison.csv" \\
    --output-md "${OUT_DIR}/buckeye_li_table3_comparison.md"
fi

echo "Saved FA alignments: ${OUT_DIR}/alignments.jsonl"
echo "Saved FA matrix: ${OUT_DIR}/run_test_style_matrix.csv"
echo "Saved FA figure: ${FIG_DIR}/run_test_style_matrix_paper_like.svg"
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.repo_root
    target_hparams = root / TARGET_HPARAM_DIR
    target_hparams.mkdir(parents=True, exist_ok=True)

    for loss in LOSSES:
        source = root / SOURCE_HPARAM_DIR / f"phnmonossl_{loss}.yaml"
        target = target_hparams / f"phnmonossl_{loss}.yaml"
        target.write_text(rewrite_hparam(source.read_text(encoding="utf-8")), encoding="utf-8")
        write_executable(root / f"run_librispeech_integrated_{loss}.sh", train_single_script(loss))

    write_executable(root / "run_librispeech_integrated.sh", train_group_script())
    write_executable(root / "run_librispeech_buckeye_timit_eval_integrated.sh", eval_script())
    write_executable(root / "run_librispeech_buckeye_timit_fa_integrated.sh", fa_script())


if __name__ == "__main__":
    main()
