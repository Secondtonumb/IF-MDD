"""Metric helpers for context-phone CTC experiments."""

from __future__ import annotations

from utils.context_phone_codec import decode_context_tokens, normalize_context_mode


def _normalize_decoded(decoded):
    """Normalize SpeechBrain decode_ndim output to list[list[str]]."""
    if decoded is None:
        return []
    if isinstance(decoded, str):
        return [decoded.split()]
    if isinstance(decoded, tuple):
        decoded = list(decoded)
    if isinstance(decoded, list):
        if not decoded:
            return []
        if all(isinstance(item, str) for item in decoded):
            return [decoded]
        normalized = []
        for item in decoded:
            if isinstance(item, str):
                normalized.append(item.split())
            elif isinstance(item, tuple):
                normalized.append([str(token) for token in item])
            elif isinstance(item, list):
                normalized.append([str(token) for token in item])
            else:
                normalized.append([str(item)])
        return normalized
    return [[str(decoded)]]


def make_context_phone_ind2lab(base_ind2lab, mode: str | None):
    """Wrap an ind2lab callable and project context-phone tokens to phones.

    The returned callable has the same role as SpeechBrain's usual
    ``label_encoder.decode_ndim`` argument for ErrorRateStats/MpdStats.
    """
    mode = normalize_context_mode(mode)
    if mode in {"", "none", "null", "mono", "uniphone"}:
        return base_ind2lab
    if mode not in {"diphone", "triphone", "between_word_triphone", "word_position_uniphone"}:
        raise ValueError(f"Unsupported context_phone_mode: {mode}")

    def context_ind2lab(indices):
        decoded = _normalize_decoded(base_ind2lab(indices))
        return [decode_context_tokens(tokens, mode) for tokens in decoded]

    return context_ind2lab


def decode_ids_to_phone_string(base_ind2lab, indices, mode: str | None) -> str:
    """Decode one token-id sequence to a space-separated phone string."""
    ind2lab = make_context_phone_ind2lab(base_ind2lab, mode)
    decoded = _normalize_decoded(ind2lab([indices]))
    if not decoded:
        return ""
    return " ".join(str(token) for token in decoded[0])
