#!/usr/bin/env python3
"""Create no-overlap context-phone training YAMLs and qsub scripts."""

from __future__ import annotations

import argparse
import stat
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LOSSES = ["ctc", "crctc", "ottc", "crottc"]

VARIANTS = {
    "dp": {
        "source_hparams": "hparams_l2_dp",
        "target_hparams": "hparams_l2_nooverlap_dp",
        "data_dir": "data/context_phone_nooverlap_times/l2arctic_diphone",
        "old_data_dir": "data/context_phone/l2arctic_diphone",
        "output_subdir": "diphone",
        "output_neurons": 1473,
        "mode": "diphone",
        "prepare_mode": "diphone",
    },
    "tri": {
        "source_hparams": "hparams_l2_tri",
        "target_hparams": "hparams_l2_nooverlap_tri",
        "data_dir": "data/context_phone_nooverlap_times/l2arctic_triphone",
        "old_data_dir": "data/context_phone/l2arctic_triphone",
        "output_subdir": "triphone",
        "output_neurons": 14042,
        "mode": "triphone",
        "prepare_mode": "triphone",
    },
    "bwt": {
        "source_hparams": "hparams_l2_bwt",
        "target_hparams": "hparams_l2_nooverlap_bwt",
        "data_dir": "data/context_phone_nooverlap_times/l2arctic_between_word_triphone",
        "old_data_dir": "data/context_phone/l2arctic_between_word_triphone",
        "output_subdir": "between_word_triphone",
        "output_neurons": 19290,
        "mode": "between_word_triphone",
        "prepare_mode": "between-word-triphone",
    },
    "bwu": {
        "source_hparams": "hparams_l2_wpu",
        "target_hparams": "hparams_l2_nooverlap_bwu",
        "data_dir": "data/context_phone_nooverlap_times/l2arctic_between_word_uniphone",
        "old_data_dir": "data/context_phone/l2arctic_word_position_uniphone",
        "output_subdir": "between_word_uniphone",
        "output_neurons": 184,
        "mode": "word_position_uniphone",
        "prepare_mode": "between-word-uniphone",
    },
}


def replace_yaml(text: str, variant: dict) -> str:
    text = text.replace(
        f"lab_enc_file: ./{variant['old_data_dir']}/label_encoder.txt",
        f"lab_enc_file: ./{variant['data_dir']}/label_encoder.txt",
    )
    text = text.replace(
        f'data_folder_save: "./{variant["old_data_dir"]}"',
        f'data_folder_save: "./{variant["data_dir"]}"',
    )
    text = text.replace(
        f"exp_l2arctic_context/{variant['output_subdir']}/",
        f"exp_l2arctic_context_nooverlap/{variant['output_subdir']}/",
    )
    if variant["old_data_dir"].endswith("l2arctic_word_position_uniphone"):
        text = text.replace(
            "exp_l2arctic_context/word_position_uniphone/",
            "exp_l2arctic_context_nooverlap/between_word_uniphone/",
        )
    text = text.replace(
        'context_phone_mode: "diphone"',
        f'context_phone_mode: "{variant["mode"]}"',
    )
    text = text.replace(
        'context_phone_mode: "triphone"',
        f'context_phone_mode: "{variant["mode"]}"',
    )
    text = text.replace(
        'context_phone_mode: "between_word_triphone"',
        f'context_phone_mode: "{variant["mode"]}"',
    )
    text = text.replace(
        'context_phone_mode: "word_position_uniphone"',
        f'context_phone_mode: "{variant["mode"]}"',
    )

    out_lines = []
    for line in text.splitlines():
        if line.startswith("output_neurons:"):
            out_lines.append(
                f"output_neurons: {variant['output_neurons']} # no-overlap metadata.json output_neurons"
            )
        else:
            out_lines.append(line)
    return "\n".join(out_lines) + "\n"


def write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)


def single_run_script(variant_key: str, variant: dict, loss: str) -> str:
    hparam = f"{variant['target_hparams']}/phnmonossl_{loss}.yaml"
    data_dir = variant["data_dir"]
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

python train.py {hparam} \\
  --mode train+eval \\
  --device="${{DEVICE}}" \\
  --batch_size="${{BATCH_SIZE}}" \\
  --number_of_epochs="${{EPOCHS}}"
"""


def group_run_script(variant_key: str, variant: dict) -> str:
    data_dir = variant["data_dir"]
    configs = " \\\n  ".join(
        f"{variant['target_hparams']}/phnmonossl_{loss}.yaml" for loss in LOSSES
    )
    return f"""#!/usr/bin/env bash
#------ qsub option --------#
#PBS -q regular-g
#PBS -l select=1:mpiprocs=4
#PBS -l walltime=40:00:00
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.repo_root
    for variant_key, variant in VARIANTS.items():
        target_dir = root / variant["target_hparams"]
        target_dir.mkdir(parents=True, exist_ok=True)
        for loss in LOSSES:
            src = root / variant["source_hparams"] / f"phnmonossl_{loss}.yaml"
            dst = target_dir / f"phnmonossl_{loss}.yaml"
            dst.write_text(replace_yaml(src.read_text(encoding="utf-8"), variant), encoding="utf-8")

            write_executable(
                root / f"run_l2_nooverlap_{variant_key}_{loss}.sh",
                single_run_script(variant_key, variant, loss),
            )

        write_executable(
            root / f"run_l2_nooverlap_{variant_key}.sh",
            group_run_script(variant_key, variant),
        )


if __name__ == "__main__":
    main()
