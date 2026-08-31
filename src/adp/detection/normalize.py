from __future__ import annotations

import unicodedata

import regex as re

_PUNCT_RE = re.compile(r"[\p{P}\p{S}]+", re.UNICODE)
_WS_RE = re.compile(r"\s+", re.UNICODE)

_FOLD_MAP = str.maketrans(
    {
        "á": "a",
        "é": "e",
        "í": "i",
        "ó": "o",
        "ö": "o",
        "ő": "o",
        "ú": "u",
        "ü": "u",
        "ű": "u",
    }
)


def normalize(text: str) -> str:
    stage = unicodedata.normalize("NFKC", text)
    stage = stage.casefold()
    stage = _PUNCT_RE.sub(" ", stage)
    stage = _WS_RE.sub(" ", stage)
    return stage.strip()


def fold_diacritics(text: str) -> str:
    return text.translate(_FOLD_MAP)


def normalize_for_matching(text: str) -> str:
    return fold_diacritics(normalize(text))


class NormalizationMap:
    """Normalized text plus a per-character offset map back into the original string."""

    def __init__(self, original: str) -> None:
        self.original = original
        nfkc_chars: list[str] = []
        nfkc_offsets: list[int] = []
        for index, char in enumerate(original):
            expanded = unicodedata.normalize("NFKC", char)
            if not expanded:
                continue
            for piece in expanded:
                nfkc_chars.append(piece)
                nfkc_offsets.append(index)

        folded_chars: list[str] = []
        folded_offsets: list[int] = []
        for char, offset in zip(nfkc_chars, nfkc_offsets, strict=True):
            lowered = char.casefold()
            if not lowered:
                continue
            for piece in lowered:
                folded_chars.append(piece)
                folded_offsets.append(offset)

        out_chars: list[str] = []
        out_offsets: list[int] = []
        pending_space = False
        for char, offset in zip(folded_chars, folded_offsets, strict=True):
            replaced = _PUNCT_RE.sub(" ", char)
            replaced = replaced if replaced else " " if char.isspace() else char
            if replaced.isspace() or replaced == "":
                pending_space = bool(out_chars)
                continue
            if pending_space:
                out_chars.append(" ")
                out_offsets.append(offset)
                pending_space = False
            out_chars.append(fold_diacritics(replaced))
            out_offsets.append(offset)

        self.text = "".join(out_chars)
        self.offsets = out_offsets

    def original_span(self, start: int, end: int) -> tuple[int, int]:
        if not self.offsets:
            return (0, 0)
        clamped_start = max(0, min(start, len(self.offsets) - 1))
        clamped_end = max(0, min(end - 1, len(self.offsets) - 1))
        origin_start = self.offsets[clamped_start]
        origin_end = self.offsets[clamped_end] + 1
        if origin_end <= origin_start:
            origin_end = origin_start + 1
        return (origin_start, min(origin_end, len(self.original)))

    def original_text(self, start: int, end: int) -> str:
        origin_start, origin_end = self.original_span(start, end)
        return self.original[origin_start:origin_end]
