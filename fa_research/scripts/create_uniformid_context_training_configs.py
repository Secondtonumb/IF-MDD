#!/usr/bin/env python3
"""Create UniformID no-overlap context-phone YAMLs and qsub scripts."""

from __future__ import annotations

import argparse
import json
import stat
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LOSSES = ["ctc", "crctc", "ottc", "crottc"]

VARIANTS = {
    "dp": {
        "source_hparams": "hparams_l2_dp",
        "target_hparams": "hparams_l2_uniformid_dp",
        "data_dir": "data/context_phone_uniformid_nooverlap_times/l2arctic_diphone",
        "old_data_dirs": [
            "data/context_phone/l2arctic_diphone",
            "data/context_phone_nooverlap_times/l2arctic_diphone",
        ],
        "old_exp_dirs": [
            "exp_l2arctic_context/diphone/",
            "exp_l2arctic_context_nooverlap/diphone/",
        ],
        "output_subdir": "diphone",
        "mode": "diphone",
    },
    "tri": {
        "source_hparams": "hparams_l2_tri",
        "target_hparams": "hparams_l2_uniformid_tri",
        "data_dir": "data/context_phone_uniformid_nooverlap_times/l2arctic_triphone",
        "old_data_dirs": [
            "data/context_phone/l2arctic_triphone",
            "data/context_phone_nooverlap_times/l2arctic_triphone",
        ],
        "old_exp_dirs": [
            "exp_l2arctic_context/triphone/",
            "exp_l2arctic_context_nooverlap/triphone/",
        ],
        "output_subdir": "triphone",
        "mode": "triphone",
    },
    "bwt": {
        "source_hparams": "hparams_l2_bwt",
        "target_hparams": "hparams_l2_uniformid_bwt",
        "data_dir": "data/context_phone_uniformid_nooverlap_times/l2arctic_between_word_triphone",
        "old_data_dirs": [
            "data/context_phone/l2arctic_between_word_triphone",
            "data/context_phone_nooverlap_times/l2arctic_between_word_triphone",
        ],
        "old_exp_dirs": [
            "exp_l2arctic_context/between_word_triphone/",
            "exp_l2arctic_context_nooverlap/between_word_triphone/",
        ],
        "output_subdir": "between_word_triphone",
        "mode": "between_word_triphone",
    },
    "bwu": {
        "source_hparams": "hparams_l2_wpu",
        "target_hparams": "hparams_l2_uniformid_bwu",
        "data_dir": "data/context_phone_uniformid_nooverlap_times/l2arctic_between_word_uniphone",
        "old_data_dirs": [
            "data/context_phone/l2arctic_word_position_uniphone",
            "data/context_phone_nooverlap_times/l2arctic_between_word_uniphone",
        ],
        "old_exp_dirs": [
            "exp_l2arctic_context/word_position_uniphone/",
            "exp_l2arctic_context_nooverlap/between_word_uniphone/",
        ],
        "output_subdir": "between_word_uniphone",
        "mode": "word_position_uniphone",
    },
}


def read_output_neurons(root: Path, data_dir: str) -> int:
    metadata = json.loads((root / data_dir / "metadata.json").read_text(encoding="utf-8"))
    return int(metadata["output_neurons"])


def replace_yaml(text: str, variant: dict, output_neurons: int, exp_root: str) -> str:
    for old_data_dir in variant["old_data_dirs"]:
        text = text.replace(
            f"lab_enc_file: ./{old_data_dir}/label_encoder.txt",
            f"lab_enc_file: ./{variant['data_dir']}/label_encoder.txt",
        )
        text = text.replace(
            f'data_folder_save: "./{old_data_dir}"',
            f'data_folder_save: "./{variant["data_dir"]}"',
        )
    for old_exp_dir in variant["old_exp_dirs"]:
        text = text.replace(old_exp_dir, f"{exp_root}/{variant['output_subdir']}/")

    for old_mode in [
        "diphone",
        "triphone",
        "between_word_triphone",
        "word_position_uniphone",
    ]:
        text = text.replace(
            f'context_phone_mode: "{old_mode}"',
            f'context_phone_mode: "{variant["mode"]}"',
        )

    out_lines = []
    for line in text.splitlines():
        if line.startswith("output_neurons:"):
            out_lines.append(
                f"output_neurons: {output_neurons} # uniformid metadata.json output_neurons"
            )
        else:
            out_lines.append(line)
    return "\n".join(out_lines) + "\n"


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
    configs = " \\\n+  ".join(f"{hparam_dir}/phnmonossl_{loss}.yaml" for loss in LOSSES)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--exp-root", default="exp_l2arctic_context_uniformid_nooverlap")
    parser.add_argument("--script-prefix", default="run_l2_uniformid")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.repo_root
    for variant_key, variant in VARIANTS.items():
        output_neurons = read_output_neurons(root, variant["data_dir"])
        target_dir = root / variant["target_hparams"]
        target_dir.mkdir(parents=True, exist_ok=True)
        hparam_dir_rel = variant["target_hparams"]
        data_dir_rel = variant["data_dir"]
        for loss in LOSSES:
            src = root / variant["source_hparams"] / f"phnmonossl_{loss}.yaml"
            dst = target_dir / f"phnmonossl_{loss}.yaml"
            dst.write_text(
                replace_yaml(src.read_text(encoding="utf-8"), variant, output_neurons, args.exp_root),
                encoding="utf-8",
            )
            write_executable(
                root / f"{args.script_prefix}_{variant_key}_{loss}.sh",
                single_run_script(hparam_dir_rel, data_dir_rel, loss),
            )

        write_executable(
            root / f"{args.script_prefix}_{variant_key}.sh",
            group_run_script(hparam_dir_rel, data_dir_rel),
        )


if __name__ == "__main__":
    main()
