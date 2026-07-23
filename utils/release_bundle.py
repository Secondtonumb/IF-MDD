"""Deterministic loading helpers for public IF-MDD inference bundles."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


LABEL_ENCODER_SHA256 = (
    "98cee9707ab67c3e29ee337debf4ba319cbc61c3777024db6b8f3494f0df5bfe"
)
_TOKEN_LINE = re.compile(r"^'(.*)' => ([0-9]+)$")


@dataclass(frozen=True)
class InferenceBundle:
    """Resolved external checkpoint and its deterministic metadata."""

    root: Path
    checkpoint: Path
    manifest: Mapping[str, Any] | None
    hyperparams: Path | None
    label_encoder: Path | None
    required_recoverables: tuple[str, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_token_to_id(path: Path) -> dict[str, int]:
    """Read the token section of a SpeechBrain TextEncoder file."""

    mapping: dict[str, int] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line == "================":
            break
        match = _TOKEN_LINE.match(line)
        if match:
            token, token_id = match.groups()
            mapping[token] = int(token_id)
    if not mapping:
        raise ValueError(f"No token-to-ID entries found in label encoder: {path}")
    ids = sorted(mapping.values())
    if ids != list(range(len(ids))):
        raise ValueError(
            f"Label encoder IDs must be contiguous from zero: {path} has {ids}"
        )
    return mapping


def validate_label_encoder(
    configured_path: str | Path,
    *,
    bundle_path: str | Path | None = None,
    expected_sha256: str = LABEL_ENCODER_SHA256,
) -> dict[str, int]:
    """Fail before checkpoint loading if the complete vocabulary order differs."""

    configured = Path(configured_path).expanduser().resolve()
    if not configured.is_file():
        raise FileNotFoundError(f"Label encoder not found: {configured}")
    actual_sha = sha256_file(configured)
    if actual_sha != expected_sha256:
        raise ValueError(
            "L2-ARCTIC label encoder SHA256 mismatch: "
            f"expected {expected_sha256}, got {actual_sha} ({configured}). "
            "A 44-way output shape is not sufficient; token IDs must match."
        )
    configured_mapping = read_token_to_id(configured)

    if bundle_path is not None:
        bundled = Path(bundle_path).expanduser().resolve()
        if not bundled.is_file():
            raise FileNotFoundError(f"Bundle label encoder not found: {bundled}")
        bundle_sha = sha256_file(bundled)
        if bundle_sha != expected_sha256:
            raise ValueError(
                "Bundle label encoder SHA256 mismatch: "
                f"expected {expected_sha256}, got {bundle_sha} ({bundled})"
            )
        bundle_mapping = read_token_to_id(bundled)
        if bundle_mapping != configured_mapping:
            raise ValueError(
                "Bundle and recipe label encoders use different token-to-ID maps"
            )
    return configured_mapping


def _safe_child(root: Path, relative: str, field: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"manifest {field!r} escapes bundle root: {relative}") from exc
    return candidate


def _checkpoint_candidates(root: Path) -> list[Path]:
    candidates: set[Path] = set()
    if root.is_dir() and (root / "model.ckpt").is_file():
        candidates.add(root.resolve())
    for model_file in root.rglob("model.ckpt"):
        checkpoint = model_file.parent.resolve()
        if checkpoint.name.startswith("CKPT+") or (checkpoint / "CKPT.yaml").is_file():
            candidates.add(checkpoint)
        elif checkpoint.name == "checkpoint":
            candidates.add(checkpoint)
    return sorted(candidates)


def resolve_inference_bundle(path_like: str | Path) -> InferenceBundle:
    """Resolve a manifest root or an exact SpeechBrain checkpoint directory."""

    supplied = Path(path_like).expanduser().resolve()
    if not supplied.is_dir():
        raise FileNotFoundError(f"Inference checkpoint path not found: {supplied}")

    manifest_path = supplied / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format_version") != 1:
            raise ValueError(
                f"Unsupported manifest format_version: {manifest.get('format_version')!r}"
            )
        checkpoint_value = manifest.get("checkpoint")
        label_value = manifest.get("label_encoder")
        hyperparams_value = manifest.get("hyperparams")
        if not all(isinstance(item, str) and item for item in (
            checkpoint_value,
            label_value,
            hyperparams_value,
        )):
            raise ValueError(
                "manifest.json must define checkpoint, hyperparams, and label_encoder"
            )
        checkpoint = _safe_child(supplied, checkpoint_value, "checkpoint")
        label_encoder = _safe_child(supplied, label_value, "label_encoder")
        hyperparams = _safe_child(supplied, hyperparams_value, "hyperparams")
        required = tuple(manifest.get("required_recoverables", ("model",)))
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"Manifest checkpoint not found: {checkpoint}")
        if not hyperparams.is_file():
            raise FileNotFoundError(f"Manifest hyperparams not found: {hyperparams}")
        return InferenceBundle(
            root=supplied,
            checkpoint=checkpoint,
            manifest=manifest,
            hyperparams=hyperparams,
            label_encoder=label_encoder,
            required_recoverables=required,
        )

    candidates = _checkpoint_candidates(supplied)
    if len(candidates) != 1:
        rendered = "\n".join(f"  - {item}" for item in candidates) or "  (none)"
        raise ValueError(
            "A bundle without manifest.json must contain exactly one inference "
            f"checkpoint; found {len(candidates)}:\n{rendered}"
        )
    checkpoint = candidates[0]
    required = ("model", "perceived_ssl") if (checkpoint / "perceived_ssl.ckpt").is_file() else ("model",)
    return InferenceBundle(
        root=supplied,
        checkpoint=checkpoint,
        manifest=None,
        hyperparams=None,
        label_encoder=None,
        required_recoverables=required,
    )


def validate_model_signature(
    hparams: Mapping[str, Any], manifest: Mapping[str, Any] | None
) -> None:
    """Validate architecture-defining scalar fields before loading weights."""

    if manifest is None:
        return
    signature = manifest.get("model_signature", {})
    if not isinstance(signature, dict):
        raise ValueError("manifest model_signature must be an object")
    mismatches = []
    for key, expected in signature.items():
        actual = hparams.get(key)
        if actual != expected:
            mismatches.append(f"{key}: bundle={expected!r}, recipe={actual!r}")
    if mismatches:
        raise ValueError("Bundle model signature mismatch:\n  " + "\n  ".join(mismatches))


def apply_bundle_defaults(
    hparams: Mapping[str, Any],
    manifest: Mapping[str, Any] | None,
    *,
    cli_override_keys: Sequence[str] = (),
) -> None:
    """Apply non-architectural bundle defaults unless the CLI set the key."""

    if manifest is None:
        return
    defaults = manifest.get("defaults", {})
    if not isinstance(defaults, dict):
        raise ValueError("manifest defaults must be an object")
    protected = set(cli_override_keys)
    for key, value in defaults.items():
        if key not in protected:
            hparams[key] = value


def install_strict_inference_checkpointer(brain: Any, bundle: InferenceBundle) -> None:
    """Strictly load all inference recoverables and replace the local checkpointer."""

    existing = brain.checkpointer
    if existing is None:
        raise ValueError("The recipe did not construct a SpeechBrain checkpointer")
    available = existing.recoverables
    required = bundle.required_recoverables
    missing_recoverables = [name for name in required if name not in available]
    if missing_recoverables:
        raise ValueError(
            f"Recipe lacks required recoverables: {', '.join(missing_recoverables)}"
        )

    selected = {name: available[name] for name in required}
    for name, recoverable in selected.items():
        state_path = bundle.checkpoint / f"{name}.ckpt"
        if not state_path.is_file():
            raise FileNotFoundError(
                f"Complete inference checkpoint is missing {state_path.name}: "
                f"{bundle.checkpoint}"
            )
        if not isinstance(recoverable, torch.nn.Module):
            raise TypeError(
                f"Strict inference recoverable {name!r} is not a torch.nn.Module"
            )
        state = torch.load(state_path, map_location=brain.device, weights_only=False)
        try:
            recoverable.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Strict load failed for {name!r} from {state_path}: {exc}"
            ) from exc

    class _LoadedInferenceCheckpointer:
        def __init__(self, checkpoint):
            self.checkpoint = checkpoint

        def recover_if_possible(self, *args, **kwargs):
            return self.checkpoint

    brain.checkpointer = _LoadedInferenceCheckpointer(bundle.checkpoint)
    brain._release_inference_checkpoint = bundle.checkpoint
