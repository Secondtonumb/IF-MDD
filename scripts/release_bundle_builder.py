"""Shared staging logic for standalone L2-ARCTIC inference bundles."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path


LABEL_SHA256 = "98cee9707ab67c3e29ee337debf4ba319cbc61c3777024db6b8f3494f0df5bfe"


@dataclass(frozen=True)
class BundleSpec:
    name: str
    bundle_type: str
    checkpoint: Path
    hparams: Path
    output: Path
    root: Path
    wavlm: Path
    llama: Path | None
    required_recoverables: tuple[str, ...]
    model_signature: dict
    defaults: dict


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Required source file not found: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_runtime_file(root: Path, output: Path, relative: str) -> None:
    copy_file(root / relative, output / "runtime" / relative)


def rewrite_hparams(spec: BundleSpec) -> str:
    text = spec.hparams.read_text(encoding="utf-8")
    replacements = {
        "lab_enc_file: utils/label_encoder.txt": (
            "lab_enc_file: !ref <bundle_root>/label_encoder.txt"
        ),
        "perceived_ssl_source: !ref <perceived_ssl_model>": (
            "perceived_ssl_source: !ref <bundle_root>/wavlm"
        ),
        "load_pretrained_ssl_weights: true": (
            "load_pretrained_ssl_weights: false"
        ),
    }
    if spec.llama is not None:
        replacements.update(
            {
                "llm_source: meta-llama/Llama-3.2-1B-Instruct": (
                    "llm_source: !ref <bundle_root>/llama"
                ),
                "load_pretrained_llm_weights: true": (
                    "load_pretrained_llm_weights: false"
                ),
            }
        )
    for original, replacement in replacements.items():
        count = text.count(original)
        if count != 1:
            raise ValueError(
                f"Expected exactly one hparams line {original!r}; found {count}"
            )
        text = text.replace(original, replacement)
    return text


def write_custom_interface(output: Path) -> None:
    source = '''#!/usr/bin/env python3
"""Bundle-local single-audio inference interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torchaudio
import yaml
from hyperpyyaml import load_hyperpyyaml
from speechbrain.dataio.encoder import CTCTextEncoder

_BUNDLE = Path(__file__).resolve().parent
_RUNTIME = _BUNDLE / "runtime"
sys.path.insert(0, str(_RUNTIME))

from utils.release_bundle import (
    install_strict_inference_checkpointer,
    resolve_inference_bundle,
    validate_label_encoder,
    validate_model_signature,
)


class _AudioBatch:
    def __init__(self, wavs, wav_lens, ids):
        self.id = list(ids)
        self.sig = (wavs, wav_lens)

    def to(self, device):
        self.sig = tuple(item.to(device) for item in self.sig)
        return self


def _override_yaml(overrides, device, bundle_type):
    values = {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must use KEY=VALUE syntax: {item}")
        key, raw_value = item.split("=", 1)
        values[key] = yaml.safe_load(raw_value)
    values["bundle_root"] = str(_BUNDLE)
    if bundle_type == "crottc-if":
        values["loss_device"] = str(device)
    if str(device).startswith("cpu"):
        values["precision"] = "fp32"
        values["auto_mix_prec"] = False
    return yaml.safe_dump(values, sort_keys=False)


def _jsonable(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


class BundleModel:
    def __init__(self, brain, label_encoder, device, bundle_type):
        self.brain = brain
        self.label_encoder = label_encoder
        self.device = torch.device(device)
        self.bundle_type = bundle_type

    @torch.inference_mode()
    def transcribe_file(self, audio):
        wav, sample_rate = torchaudio.load(str(audio))
        wav = wav.float()
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sample_rate != 16000:
            wav = torchaudio.functional.resample(wav, sample_rate, 16000)
        batch = _AudioBatch(
            wav.to(self.device),
            torch.ones(1, device=self.device),
            [str(audio)],
        )
        result = self.brain.inference_batch(batch)
        if self.bundle_type == "crottc-if":
            hyps_value = result.get("hyps")
            hyps = _jsonable(hyps_value) if hyps_value is not None else []
            result["sequence_tokens"] = hyps
            result["sequence_text"] = [
                " ".join(self.label_encoder.decode_ndim(sequence))
                for sequence in hyps
            ]
            p_ctc = result.get("p_ctc_feat")
            if p_ctc is not None:
                result["logits_shape"] = list(p_ctc.shape)
                ctc_tokens = torch.argmax(p_ctc, dim=-1)
                result["ctc_argmax_tokens"] = _jsonable(ctc_tokens)
                del result["p_ctc_feat"]
        return _jsonable(result)


def from_pretrained(bundle_dir, device=None, overrides=(), hparams_file=None):
    bundle = resolve_inference_bundle(bundle_dir)
    if bundle.manifest is None or bundle.hyperparams is None:
        raise ValueError("custom_interface requires a manifest bundle root")
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    selected_hparams = (
        (bundle.root / hparams_file).resolve()
        if hparams_file is not None
        else bundle.hyperparams
    )
    try:
        selected_hparams.relative_to(bundle.root)
    except ValueError as exc:
        raise ValueError("hparams_file must stay inside the bundle") from exc
    with selected_hparams.open(encoding="utf-8") as stream:
        hparams = load_hyperpyyaml(
            stream,
            _override_yaml(overrides, device, bundle.manifest["bundle_type"]),
        )
    validate_model_signature(hparams, bundle.manifest)
    bundle_type = bundle.manifest["bundle_type"]
    if bundle_type == "crottc-if":
        from models.Trans_IFMDD_ConPCO_ver2 import Trans_IFMDD_ConPCO_ver2
        brain_class = Trans_IFMDD_ConPCO_ver2
    elif bundle_type == "mdd-llm-llama3.2":
        from models.SSL_LLM_origin_ver2 import SSL_LLM_origin_ver2
        brain_class = SSL_LLM_origin_ver2
    else:
        raise ValueError(f"Unsupported bundle_type: {bundle_type}")
    brain = brain_class(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts={"device": device},
        checkpointer=hparams["checkpointer"],
    )
    label_encoder = CTCTextEncoder()
    label_encoder.load(bundle.label_encoder)
    label_encoder.expect_len(hparams["output_neurons"])
    validate_label_encoder(bundle.label_encoder)
    brain.label_encoder = label_encoder
    install_strict_inference_checkpointer(brain, bundle)
    brain.modules.eval()
    if hasattr(brain, "_ensure_initialized"):
        brain._ensure_initialized()
    return BundleModel(brain, label_encoder, device, bundle_type)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hparams-file", default="hyperparams.yaml")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--device")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    model = from_pretrained(
        _BUNDLE,
        device=args.device,
        overrides=args.override,
        hparams_file=args.hparams_file,
    )
    print(json.dumps(model.transcribe_file(args.audio), indent=2))


if __name__ == "__main__":
    main()
'''
    destination = output / "custom_interface.py"
    destination.write_text(source, encoding="utf-8")
    destination.chmod(0o755)


def build_bundle(spec: BundleSpec) -> Path:
    checkpoint = spec.checkpoint.expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint}")
    output = spec.output.expanduser().resolve()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    label_encoder = spec.root / "utils" / "label_encoder.txt"
    if sha256_file(label_encoder) != LABEL_SHA256:
        raise ValueError("Repository label encoder does not match release SHA256")
    copy_file(label_encoder, output / "label_encoder.txt")
    (output / "hyperparams.yaml").write_text(
        rewrite_hparams(spec),
        encoding="utf-8",
    )
    copy_file(spec.root / "requirements.txt", output / "requirements.txt")

    checkpoint_output = output / "checkpoint"
    for recoverable in spec.required_recoverables:
        copy_file(
            checkpoint / f"{recoverable}.ckpt",
            checkpoint_output / f"{recoverable}.ckpt",
        )
    copy_file(checkpoint / "CKPT.yaml", checkpoint_output / "CKPT.yaml")

    wavlm_files = ("config.json", "preprocessor_config.json")
    for name in wavlm_files:
        copy_file(spec.wavlm / name, output / "wavlm" / name)
    if spec.llama is not None:
        for name in (
            "config.json",
            "generation_config.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ):
            copy_file(spec.llama / name, output / "llama" / name)

    common_runtime = (
        "models/projector.py",
        "trainer/AutoSSLoader.py",
        "utils/release_bundle.py",
        "mpd_eval_v4.py",
    )
    for relative in common_runtime:
        copy_runtime_file(spec.root, output, relative)
    if spec.bundle_type == "crottc-if":
        for relative in (
            "models/Trans_IFMDD_ConPCO_ver2.py",
            "trainer/ConPCO_TransASR.py",
            "utils/layers/utils.py",
            "utils/losses/ConPCO_norm.py",
            "utils/plot/plot_attn.py",
        ):
            copy_runtime_file(spec.root, output, relative)
    else:
        for relative in (
            "models/SSL_LLM_origin_ver2.py",
            "trainer/AutoLLMLoader.py",
        ):
            copy_runtime_file(spec.root, output, relative)
    for package in ("models", "trainer", "utils", "utils/layers", "utils/losses", "utils/plot"):
        package_dir = output / "runtime" / package
        if package_dir.is_dir():
            (package_dir / "__init__.py").touch()

    manifest = {
        "format_version": 1,
        "model_name": spec.name,
        "bundle_type": spec.bundle_type,
        "checkpoint": "checkpoint",
        "hyperparams": "hyperparams.yaml",
        "label_encoder": "label_encoder.txt",
        "label_encoder_sha256": LABEL_SHA256,
        "required_recoverables": list(spec.required_recoverables),
        "model_signature": spec.model_signature,
        "defaults": spec.defaults,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    write_custom_interface(output)
    return output
