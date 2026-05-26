#!/usr/bin/env python3
"""Create hparams and qsub scripts for pruned no-overlap triphone models."""

from __future__ import annotations

import argparse
import json
import stat
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LOSSES = ["ctc", "crctc", "ottc", "crottc"]


def read_output_neurons(data_dir: Path) -> int:
    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    return int(metadata["output_neurons"])


def rewrite_hparam(
    text: str,
    data_dir: str,
    output_neurons: int,
    exp_root: str,
    exp_subdir: str,
) -> str:
    for old_data_dir in [
        "data/context_phone_nooverlap_times/l2arctic_triphone",
        "data/context_phone_uniformid_nooverlap_times/l2arctic_triphone",
    ]:
        text = text.replace(
            f"lab_enc_file: ./{old_data_dir}/label_encoder.txt",
            f"lab_enc_file: ./{data_dir}/label_encoder.txt",
        )
        text = text.replace(
            f'data_folder_save: "./{old_data_dir}"',
            f'data_folder_save: "./{data_dir}"',
        )
    for old_exp_dir in [
        "exp_l2arctic_context_nooverlap/triphone/",
        "exp_l2arctic_context_uniformid_nooverlap/triphone/",
    ]:
        text = text.replace(old_exp_dir, f"{exp_root}/{exp_subdir}/")

    lines = []
    for line in text.splitlines():
        if line.startswith("output_neurons:"):
            lines.append(f"output_neurons: {output_neurons} # pruned triphone metadata.json output_neurons")
        else:
            lines.append(line)
    return "\n".join(lines) + "\n"


def write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)


def single_run_script(hparam_dir: str, data_dir: str, loss: str) -> str:
    return f"""#!/usr/bin/env bash
#------ qsub option --------#
#PBS -q regular-g
#PBS -l select=1:mpiprocs=4
#PBS -l walltime=10:00:00
#PBS -W group_list=gm64
#PBS -j oe

set -e -o pipefail

source ~/.bashrc
cd /home/m64000/work/IF-MDD
conda activate sb_k2

DEVICE="${{DEVICE:-cuda}}"
BATCH_SIZE="${{BATCH_SIZE:-24}}"
EPOCHS="${{EPOCHS:-300}}"

python fa_research/scripts/check_json_split_overlap.py \\
  --train {data_dir}/train-train.json \\
  --valid {data_dir}/train-dev.json \\
  --test {data_dir}/test.json

python train.py {hparam_dir}/phnmonossl_{loss}.yaml \\
  --mode train+eval \\
  --device="${{DEVICE}}" \\
  --batch_size="${{BATCH_SIZE}}" \\
  --number_of_epochs="${{EPOCHS}}"
"""


def group_run_script(hparam_dir: str, data_dir: str) -> str:
    configs = " \\\n  ".join(f"{hparam_dir}/phnmonossl_{loss}.yaml" for loss in LOSSES)
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
EPOCHS="${{EPOCHS:-300}}"

python fa_research/scripts/check_json_split_overlap.py \\
  --train {data_dir}/train-train.json \\
  --valid {data_dir}/train-dev.json \\
  --test {data_dir}/test.json

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


def repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-hparams", type=Path, default=REPO_ROOT / "hparams_l2_nooverlap_tri")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "data/context_phone_nooverlap_times/l2arctic_triphone_pruned_min10",
    )
    parser.add_argument("--target-hparams", type=Path, default=REPO_ROOT / "hparams_l2_nooverlap_tri_pruned_min10")
    parser.add_argument("--script-prefix", default="run_l2_nooverlap_tri_pruned_min10")
    parser.add_argument("--exp-root", default="exp_l2arctic_context_nooverlap_pruned")
    parser.add_argument("--exp-subdir", default="triphone_pruned_min10")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_hparams = repo_path(args.source_hparams)
    data_dir = repo_path(args.data_dir)
    target_hparams = repo_path(args.target_hparams)
    data_dir_rel = data_dir.relative_to(REPO_ROOT).as_posix()
    hparam_dir_rel = target_hparams.relative_to(REPO_ROOT).as_posix()
    output_neurons = read_output_neurons(data_dir)

    target_hparams.mkdir(parents=True, exist_ok=True)
    for loss in LOSSES:
        src = source_hparams / f"phnmonossl_{loss}.yaml"
        dst = target_hparams / f"phnmonossl_{loss}.yaml"
        dst.write_text(
            rewrite_hparam(
                src.read_text(encoding="utf-8"),
                data_dir_rel,
                output_neurons,
                args.exp_root,
                args.exp_subdir,
            ),
            encoding="utf-8",
        )
        write_executable(
            REPO_ROOT / f"{args.script_prefix}_{loss}.sh",
            single_run_script(hparam_dir_rel, data_dir_rel, loss),
        )

    write_executable(
        REPO_ROOT / f"{args.script_prefix}.sh",
        group_run_script(hparam_dir_rel, data_dir_rel),
    )


if __name__ == "__main__":
    main()
