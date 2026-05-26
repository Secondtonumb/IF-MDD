"""Utilities for diphone/triphone CTC label experiments."""

from __future__ import annotations


BOUNDARY_CONTEXT = "<blank>"
SEP = "~"
SPECIAL_LABELS = {"<blank>", "<bos>", "<eos>"}
BWT_MODE = "between_word_triphone"
WORD_POSITION_UNIPHONE_MODE = "word_position_uniphone"
WORD_POSITION_NAMES = {
    "b": "beginning",
    "i": "interval",
    "e": "end",
    "s": "single",
}
WORD_POSITION_LABEL_SUFFIXES = {
    "b": "B",
    "i": "I",
    "e": "E",
    "s": "S",
}


def normalize_context_mode(mode: str | None) -> str:
    """Normalize context-phone mode aliases."""
    mode = (mode or "mono").strip().lower().replace("-", "_")
    aliases = {
        "bwt": BWT_MODE,
        "bwt_triphone": BWT_MODE,
        "between_word": BWT_MODE,
        "between_word_tri": BWT_MODE,
        "between_word_triphone": BWT_MODE,
        "between_word_uniphone": WORD_POSITION_UNIPHONE_MODE,
        "wpu": WORD_POSITION_UNIPHONE_MODE,
        "word_position": WORD_POSITION_UNIPHONE_MODE,
        "word_position_uniphone": WORD_POSITION_UNIPHONE_MODE,
        "uniphone_word_position": WORD_POSITION_UNIPHONE_MODE,
        "uniphone_state": WORD_POSITION_UNIPHONE_MODE,
    }
    return aliases.get(mode, mode)


def make_context_tokens(
    phones: list[str],
    mode: str,
    word_ids: list[int | None] | None = None,
) -> list[str]:
    """Convert a phone sequence into same-length context-dependent labels.

    diphone:
        p_{i-1}~p_i; the center phone for decoding is the right phone.
        The first left context is <blank>.

    triphone:
        p_{i-1}~p_i~p_{i+1}; the center phone for decoding is the middle phone.
        Boundary contexts are <blank>.

    between_word_triphone:
        Same left/center/right phone contexts as triphone, plus Hwang-style
        word-position suffixes on the center phone model: @b for word-beginning,
        @e for word-ending, and @s for single-phone words.

    word_position_uniphone:
        The center phone only, with a word-position state suffix:
        _B, _I, _E, or _S.
    """
    mode = normalize_context_mode(mode)
    phones = [str(phone).strip() for phone in phones if str(phone).strip()]
    if mode == "mono":
        return phones
    if mode == "diphone":
        return [
            f"{phones[idx - 1] if idx > 0 else BOUNDARY_CONTEXT}{SEP}{phone}"
            for idx, phone in enumerate(phones)
        ]
    if mode == "triphone":
        return [
            f"{phones[idx - 1] if idx > 0 else BOUNDARY_CONTEXT}{SEP}{phone}{SEP}"
            f"{phones[idx + 1] if idx + 1 < len(phones) else BOUNDARY_CONTEXT}"
            for idx, phone in enumerate(phones)
        ]
    if mode == BWT_MODE:
        return make_between_word_triphone_tokens(phones, word_ids)
    if mode == WORD_POSITION_UNIPHONE_MODE:
        return make_word_position_uniphone_tokens(phones, word_ids)
    raise ValueError(f"Unsupported context-phone mode: {mode}")


def word_position_code(word_ids: list[int | None] | None, idx: int) -> str:
    """Return compact word-position code for a phone index."""
    if not word_ids or idx >= len(word_ids):
        return ""
    word_id = word_ids[idx]
    if word_id is None or word_id < 0:
        return ""

    start = idx
    while start > 0 and word_ids[start - 1] == word_id:
        start -= 1
    end = idx
    while end + 1 < len(word_ids) and word_ids[end + 1] == word_id:
        end += 1

    if start == end:
        return "s"
    if idx == start:
        return "b"
    if idx == end:
        return "e"
    return "i"


def word_position_suffix(word_ids: list[int | None] | None, idx: int) -> str:
    """Return Hwang-style BWT suffix for a phone index.

    For BWT triphones, only begin/end/single are marked. Word-internal phones
    stay unmarked, matching the previous implementation.
    """
    code = word_position_code(word_ids, idx)
    return "" if code == "i" else code


def word_position_name(word_ids: list[int | None] | None, idx: int) -> str:
    """Return full word-position state name for a phone index."""
    return WORD_POSITION_NAMES.get(word_position_code(word_ids, idx), "")


def word_position_label_suffix(word_ids: list[int | None] | None, idx: int) -> str:
    """Return compact label suffix for word-position uniphone tokens."""
    return WORD_POSITION_LABEL_SUFFIXES.get(word_position_code(word_ids, idx), "")


def make_between_word_triphone_tokens(
    phones: list[str],
    word_ids: list[int | None] | None = None,
) -> list[str]:
    """Create between-word triphone labels with word-position suffixes."""
    labels = []
    for idx, phone in enumerate(phones):
        left = phones[idx - 1] if idx > 0 else BOUNDARY_CONTEXT
        right = phones[idx + 1] if idx + 1 < len(phones) else BOUNDARY_CONTEXT
        suffix = word_position_suffix(word_ids, idx)
        suffix_text = f"@{suffix}" if suffix else ""
        labels.append(f"{left}{SEP}{phone}{SEP}{right}{suffix_text}")
    return labels


def make_word_position_uniphone_tokens(
    phones: list[str],
    word_ids: list[int | None] | None = None,
) -> list[str]:
    """Create uniphone labels suffixed with word-position states."""
    labels = []
    for idx, phone in enumerate(phones):
        suffix = word_position_label_suffix(word_ids, idx)
        labels.append(f"{phone}_{suffix}" if suffix else phone)
    return labels


def center_phone(token: str, mode: str) -> str:
    """Map a context-dependent token back to its center monophone."""
    mode = normalize_context_mode(mode)
    if token in SPECIAL_LABELS:
        return token
    parts = str(token).split(SEP)
    if mode == "mono":
        return token
    if mode == "diphone":
        return parts[1] if len(parts) == 2 else token
    if mode in {"triphone", BWT_MODE}:
        return parts[1] if len(parts) == 3 else token
    if mode == WORD_POSITION_UNIPHONE_MODE:
        token = str(token)
        if "@" in token:
            return token.split("@", 1)[0]
        if token.endswith(("_B", "_I", "_E", "_S")):
            return token[:-2]
        return token
    raise ValueError(f"Unsupported context-phone mode: {mode}")


def decode_context_tokens(tokens: list[str], mode: str, drop_special: bool = True) -> list[str]:
    """Collapse context-dependent labels to center-phone labels."""
    phones = [center_phone(token, mode) for token in tokens]
    if drop_special:
        phones = [phone for phone in phones if phone not in SPECIAL_LABELS]
    return phones
