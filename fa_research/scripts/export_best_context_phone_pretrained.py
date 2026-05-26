#!/usr/bin/env python3
"""Export best L2-Arctic context-phone checkpoints for inference.

The training checkpoints already contain all files needed by SpeechBrain's
Pretrainer.  This script selects the best checkpoint for each context-phone
variant and loss recipe, then materializes a reference-style pretrained model
directory with an inference.yaml.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path


DEFAULT_REPO_ROOT = Path("/home/m64000/work/IF-MDD")
REPO_ROOT = DEFAULT_REPO_ROOT if DEFAULT_REPO_ROOT.exists() else Path(__file__).resolve().parents[2]

VARIANTS = {
    "dp": {
        "display": "Diphone",
        "mode": "diphone",
        "exp_name": "diphone",
        "data_dir": "data/context_phone/l2arctic_diphone",
        "hparams_dir": "hparams_l2_dp",
    },
    "tri": {
        "display": "Triphone",
        "mode": "triphone",
        "exp_name": "triphone",
        "data_dir": "data/context_phone/l2arctic_triphone",
        "hparams_dir": "hparams_l2_tri",
    },
    "bwt": {
        "display": "Between-word triphone",
        "mode": "between_word_triphone",
        "exp_name": "between_word_triphone",
        "data_dir": "data/context_phone/l2arctic_between_word_triphone",
        "hparams_dir": "hparams_l2_bwt",
    },
    "wpu": {
        "display": "Word-position uniphone",
        "mode": "word_position_uniphone",
        "exp_name": "word_position_uniphone",
        "data_dir": "data/context_phone/l2arctic_word_position_uniphone",
        "hparams_dir": "hparams_l2_wpu",
    },
}

LOSSES = ("ctc", "crctc", "ottc", "crottc")


@dataclass(frozen=True)
class CheckpointInfo:
    path: Path
    epoch: int
    per: float
    mpd_f1: float
    source: str = "valid_ckpt"


def parse_scalar_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip()
    return values


def parse_epoch_from_name(path: Path) -> int:
    match = re.search(r"CKPT\+(\d+)_", path.name)
    if not match:
        return -1
    return int(match.group(1))


def find_checkpoints(save_dir: Path) -> list[CheckpointInfo]:
    checkpoints: list[CheckpointInfo] = []
    for ckpt_yaml in save_dir.glob("CKPT+*.ckpt/CKPT.yaml"):
        values = parse_scalar_file(ckpt_yaml)
        try:
            per = float(values["PER"])
            mpd_f1 = float(values["mpd_f1"])
        except (KeyError, ValueError):
            continue
        checkpoints.append(
            CheckpointInfo(
                path=ckpt_yaml.parent,
                epoch=parse_epoch_from_name(ckpt_yaml.parent),
                per=per,
                mpd_f1=mpd_f1,
            )
        )
    return checkpoints


def choose_best(checkpoints: list[CheckpointInfo], metric: str) -> CheckpointInfo:
    if not checkpoints:
        raise ValueError("no checkpoints found")
    if metric == "mpd_f1":
        return max(checkpoints, key=lambda item: (item.mpd_f1, -item.per, item.epoch))
    if metric == "PER":
        return min(checkpoints, key=lambda item: (item.per, -item.mpd_f1, -item.epoch))
    raise ValueError(f"unsupported metric: {metric}")


TEST_LINE_RE = re.compile(
    r"Epoch loaded: (?P<epoch>\d+) - test loss: (?P<loss>[^,]+), "
    r"test PER: (?P<per>[0-9.eE+-]+), test mpd_f1: (?P<mpd_f1>[0-9.eE+-]+)"
)


def checkpoint_for_epoch(save_dir: Path, epoch: int) -> Path | None:
    matches = sorted(save_dir.glob(f"CKPT+{epoch:03d}_*.ckpt"))
    return matches[0] if matches else None


def find_test_results(exp_dir: Path) -> list[CheckpointInfo]:
    log_path = exp_dir / "log.txt"
    if not log_path.exists():
        return []
    results: list[CheckpointInfo] = []
    save_dir = exp_dir / "save"
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = TEST_LINE_RE.search(line)
        if not match:
            continue
        epoch = int(match.group("epoch"))
        ckpt = checkpoint_for_epoch(save_dir, epoch)
        path = ckpt if ckpt is not None else save_dir / f"CKPT+{epoch:03d}_MISSING.ckpt"
        results.append(
            CheckpointInfo(
                path=path,
                epoch=epoch,
                per=float(match.group("per")),
                mpd_f1=float(match.group("mpd_f1")),
                source="test_log",
            )
        )
    return results


def materialize_file(src: Path, dst: Path, link_mode: str) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if link_mode in {"hardlink", "auto"}:
        try:
            os.link(src, dst)
            return "hardlink"
        except OSError:
            if link_mode == "hardlink":
                raise
    shutil.copy2(src, dst)
    return "copy"


def materialize_checkpoint(src_dir: Path, dst_dir: Path, link_mode: str) -> dict[str, str]:
    dst_dir.mkdir(parents=True, exist_ok=True)
    methods: dict[str, str] = {}
    for src in sorted(src_dir.iterdir()):
        if src.is_file():
            methods[src.name] = materialize_file(src, dst_dir / src.name, link_mode)
    return methods


def inference_yaml(
    *,
    model_dir: Path,
    loss: str,
    variant_key: str,
    variant: dict[str, str],
    output_neurons: int,
    best: CheckpointInfo,
    repo_root: Path,
    selected_by: str,
) -> str:
    pretrained_root = repo_root / "pretrained_models"
    return f"""# Inference config for L2-Arctic {variant['display']} {loss.upper()} acoustic model.
prefix: "{loss}"
ctc_loss_type: "{loss}"
encoder_type: "none"
context_phone_mode: "{variant['mode']}"
context_phone_variant: "{variant_key}"

source_checkpoint: "{best.path}"
selected_by: "{selected_by}"
best_epoch: {best.epoch}
best_per: {best.per:.10g}
best_mpd_f1: {best.mpd_f1:.10g}

pretrained_models_path: {pretrained_root}
perceived_ssl_model: "wavlm_large"
canonical_ssl_model: Null
ENCODER_DIM: 1024
feature_fusion: "PhnMonoSSL"
blend_alpha: 0.5

output_folder: {model_dir}
save_folder: {model_dir}
train_log: !ref <output_folder>/inference_log.txt

training_target: "target"

perceived_ssl: !apply:trainer.AutoSSLoader.AutoSSLLoader
    model_name: !ref <perceived_ssl_model>
    freeze: !ref <freeze_perceived_ssl>
    freeze_feature_extractor: !ref <freeze_perceived_feature_extractor>
    save_path: !ref <pretrained_models_path>
    output_all_hiddens: False
preceived_ssl_emb_layer: -1

enc: !new:torch.nn.Sequential
  - !new:speechbrain.lobes.models.VanillaNN.VanillaNN
     input_shape: [null, null, !ref <ENCODER_DIM>]
     activation: !ref <activation>
     dnn_blocks: !ref <dnn_layers>
     dnn_neurons: !ref <dnn_neurons>
  - !new:torch.nn.LayerNorm
     normalized_shape: !ref <dnn_neurons>

ctc_lin: !new:speechbrain.nnet.linear.Linear
    input_size: !ref <dnn_neurons>
    n_neurons: !ref <output_neurons>

lm_weight: !new:speechbrain.nnet.linear.Linear
    input_size: !ref <dnn_neurons>
    n_neurons: 1

activation: !name:torch.nn.LeakyReLU
dnn_layers: 2
dnn_neurons: 384
freeze_perceived_ssl: False
freeze_canonical_ssl: False
freeze_perceived_feature_extractor: True
freeze_canonical_feature_extractor: True

log_softmax: !new:speechbrain.nnet.activations.Softmax
    apply_log: True

output_neurons: {output_neurons}
blank_index: 0

model: !new:torch.nn.ModuleList
    - [!ref <enc>, !ref <ctc_lin>, !ref <lm_weight>]

epoch_counter: !new:speechbrain.utils.epoch_loop.EpochCounter
    limit: 300

pretrainer: !new:speechbrain.utils.parameter_transfer.Pretrainer
    collect_in: !ref <save_folder>/
    loadables:
        perceived_ssl: !ref <perceived_ssl>
        model: !ref <model>
        tokenizer: !ref <tokenizer>

encoder: !new:speechbrain.nnet.containers.LengthsCapableSequential
    perceived_ssl: !ref <perceived_ssl>
    enc: !ref <enc>
    ctc_lin: !ref <ctc_lin>
    log_softmax: !ref <log_softmax>

decoding_function: !name:speechbrain.decoders.ctc_greedy_decode
    blank_id: !ref <blank_index>

tokenizer: !new:speechbrain.dataio.encoder.TextEncoder
    load_from_file: !ref <save_folder>/label_encoder.txt

modules:
    encoder: !ref <encoder>
"""


def load_metadata(data_dir: Path) -> dict:
    return json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
    )
    parser.add_argument("--metric", choices=("mpd_f1", "PER"), default="mpd_f1")
    parser.add_argument(
        "--selection-source",
        choices=("test-log", "valid-ckpt"),
        default="test-log",
        help="test-log uses final test PER/MDD from log.txt; valid-ckpt uses CKPT.yaml validation metrics.",
    )
    parser.add_argument(
        "--link-mode",
        choices=("auto", "hardlink", "copy"),
        default="auto",
        help="auto tries hardlinks first and falls back to copies.",
    )
    parser.add_argument("--variants", nargs="+", default=sorted(VARIANTS))
    parser.add_argument("--losses", nargs="+", default=list(LOSSES))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = REPO_ROOT
    output_root = args.output_root
    if output_root is None:
        suffix = "test_selected" if args.selection_source == "test-log" else "valid_selected"
        output_root = repo_root / "pretrained_models" / f"l2arctic_context_phone_acou_model_{suffix}"
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    manifest: dict[str, object] = {
        "source": "exp_l2arctic_context",
        "selection_source": args.selection_source,
        "selected_by": args.metric,
        "output_root": str(output_root),
        "models": {},
        "missing": [],
        "warnings": [],
    }
    rows: list[str] = [
        "variant,mode,loss,best_epoch,best_per,best_mpd_f1,source_checkpoint,pretrained_dir"
    ]
    missing_rows: list[str] = [
        "variant,mode,loss,reason,best_epoch,best_per,best_mpd_f1,source_checkpoint"
    ]
    warning_rows: list[str] = [
        "variant,mode,loss,reason,missing_epoch,missing_per,missing_mpd_f1,exported_epoch,exported_per,exported_mpd_f1"
    ]

    for variant_key in args.variants:
        if variant_key not in VARIANTS:
            raise ValueError(f"unknown variant: {variant_key}")
        variant = VARIANTS[variant_key]
        metadata = load_metadata(repo_root / variant["data_dir"])
        output_neurons = int(metadata["output_neurons"])
        label_encoder = repo_root / metadata["label_encoder"]
        variant_manifest: dict[str, object] = {
            "display": variant["display"],
            "mode": variant["mode"],
            "data_dir": str(repo_root / variant["data_dir"]),
            "output_neurons": output_neurons,
            "losses": {},
        }

        for loss in args.losses:
            if loss not in LOSSES:
                raise ValueError(f"unknown loss: {loss}")
            exp_dir = (
                repo_root
                / "exp_l2arctic_context"
                / variant["exp_name"]
                / f"wavlm_large_None_PhnMonoSSL_{loss}"
            )
            save_dir = exp_dir / "save"
            if args.selection_source == "test-log":
                candidates = find_test_results(exp_dir)
                if not candidates:
                    reason = "missing_test_result"
                    manifest["missing"].append(
                        {"variant": variant_key, "loss": loss, "reason": reason}
                    )
                    missing_rows.append(
                        ",".join([variant_key, variant["mode"], loss, reason, "", "", "", ""])
                    )
                    print(f"{variant_key}/{loss}: missing test result")
                    continue
                best_overall = choose_best(candidates, args.metric)
                exportable_candidates = [candidate for candidate in candidates if candidate.path.exists()]
                if not exportable_candidates:
                    reason = "test_checkpoint_missing"
                    manifest["missing"].append(
                        {
                            "variant": variant_key,
                            "loss": loss,
                            "reason": reason,
                            "best_epoch": best_overall.epoch,
                            "best_per": best_overall.per,
                            "best_mpd_f1": best_overall.mpd_f1,
                            "source_checkpoint": str(best_overall.path),
                        }
                    )
                    missing_rows.append(
                        ",".join(
                            [
                                variant_key,
                                variant["mode"],
                                loss,
                                reason,
                                str(best_overall.epoch),
                                f"{best_overall.per:.10g}",
                                f"{best_overall.mpd_f1:.10g}",
                                str(best_overall.path),
                            ]
                        )
                    )
                    print(
                        f"{variant_key}/{loss}: best test epoch={best_overall.epoch} "
                        "but checkpoint is missing"
                    )
                    continue
                best = choose_best(exportable_candidates, args.metric)
                if best_overall.path != best.path:
                    reason = "better_test_checkpoint_missing"
                    manifest["warnings"].append(
                        {
                            "variant": variant_key,
                            "loss": loss,
                            "reason": reason,
                            "missing_best_epoch": best_overall.epoch,
                            "missing_best_per": best_overall.per,
                            "missing_best_mpd_f1": best_overall.mpd_f1,
                            "exported_epoch": best.epoch,
                            "exported_per": best.per,
                            "exported_mpd_f1": best.mpd_f1,
                        }
                    )
                    warning_rows.append(
                        ",".join(
                            [
                                variant_key,
                                variant["mode"],
                                loss,
                                reason,
                                str(best_overall.epoch),
                                f"{best_overall.per:.10g}",
                                f"{best_overall.mpd_f1:.10g}",
                                str(best.epoch),
                                f"{best.per:.10g}",
                                f"{best.mpd_f1:.10g}",
                            ]
                        )
                    )
                    print(
                        f"{variant_key}/{loss}: warning, better test epoch="
                        f"{best_overall.epoch} checkpoint missing; exporting epoch={best.epoch}"
                    )
            else:
                best = choose_best(find_checkpoints(save_dir), args.metric)
            dst_dir = output_root / variant_key / loss

            methods = materialize_checkpoint(best.path, dst_dir, args.link_mode)
            materialize_file(label_encoder, dst_dir / "label_encoder.txt", args.link_mode)
            materialize_file(
                repo_root / variant["hparams_dir"] / f"phnmonossl_{loss}.yaml",
                dst_dir / "source_hparams.yaml",
                args.link_mode,
            )
            (dst_dir / "inference.yaml").write_text(
                inference_yaml(
                    model_dir=dst_dir,
                    loss=loss,
                    variant_key=variant_key,
                    variant=variant,
                    output_neurons=output_neurons,
                    best=best,
                    repo_root=repo_root,
                    selected_by=f"{args.selection_source}:{args.metric}",
                ),
                encoding="utf-8",
            )

            entry = {
                "display": loss.upper(),
                "ctc_loss_type": loss,
                "best_epoch": best.epoch,
                "best_per": best.per,
                "best_mpd_f1": best.mpd_f1,
                "source_checkpoint": str(best.path),
                "pretrained_dir": str(dst_dir),
                "materialization": methods,
            }
            variant_manifest["losses"][loss] = entry
            rows.append(
                ",".join(
                    [
                        variant_key,
                        variant["mode"],
                        loss,
                        str(best.epoch),
                        f"{best.per:.10g}",
                        f"{best.mpd_f1:.10g}",
                        str(best.path),
                        str(dst_dir),
                    ]
                )
            )
            print(
                f"{variant_key}/{loss}: epoch={best.epoch} "
                f"PER={best.per:.4f} mpd_f1={best.mpd_f1:.4f} -> {dst_dir}"
            )

        manifest["models"][variant_key] = variant_manifest

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_root / "best_models.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (output_root / "missing_models.csv").write_text(
        "\n".join(missing_rows) + "\n", encoding="utf-8"
    )
    (output_root / "warnings.csv").write_text(
        "\n".join(warning_rows) + "\n", encoding="utf-8"
    )
    print(f"Wrote {output_root / 'manifest.json'}")
    print(f"Wrote {output_root / 'best_models.csv'}")
    print(f"Wrote {output_root / 'missing_models.csv'}")
    print(f"Wrote {output_root / 'warnings.csv'}")


if __name__ == "__main__":
    main()
