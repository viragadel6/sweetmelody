from __future__ import annotations

from pathlib import Path

import ahocorasick
import yaml
from pydantic import BaseModel, ConfigDict, Field

from adp.detection.normalize import NormalizationMap, normalize_for_matching
from adp.schemas import DetectionKind, DetectionSpan


class LexiconEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    phrase: str
    weight: float = Field(default=1.0, ge=0.0, le=5.0)


class Lexicon(BaseModel):
    model_config = ConfigDict(extra="forbid")

    language: str
    entries: list[LexiconEntry]


def load_lexicon(path: str | Path) -> Lexicon:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"lexicon file must contain a mapping: {path}")
    lexicon = Lexicon.model_validate(raw)
    if not lexicon.entries:
        raise ValueError(f"lexicon file contains no entries: {path}")
    return lexicon


def load_lexicons(lexicon_dir: str | Path, languages: tuple[str, ...]) -> dict[str, Lexicon]:
    directory = Path(lexicon_dir)
    return {lang: load_lexicon(directory / f"{lang}.yaml") for lang in languages}


class LexiconAutomaton:
    def __init__(self, lexicons: dict[str, Lexicon]) -> None:
        self._patterns: dict[str, tuple[str, float, str]] = {}
        automaton = ahocorasick.Automaton()
        for language, lexicon in lexicons.items():
            for entry in lexicon.entries:
                key = normalize_for_matching(entry.phrase)
                if not key:
                    continue
                existing = self._patterns.get(key)
                if existing is not None and existing[1] >= entry.weight:
                    continue
                self._patterns[key] = (entry.phrase, entry.weight, language)
        for key, payload in self._patterns.items():
            automaton.add_word(key, (key, payload[0], payload[1], payload[2]))
        if self._patterns:
            automaton.make_automaton()
            self._ready = True
        else:
            self._ready = False
        self._automaton = automaton

    @property
    def pattern_count(self) -> int:
        return len(self._patterns)

    def patterns(self) -> list[tuple[str, str, float, str]]:
        return [
            (key, payload[0], payload[1], payload[2]) for key, payload in self._patterns.items()
        ]

    def find(self, text: str) -> list[DetectionSpan]:
        if not self._ready:
            return []
        mapping = NormalizationMap(text)
        haystack = mapping.text
        if not haystack:
            return []
        padded = f" {haystack} "
        raw_hits: list[tuple[int, int, str, float, str]] = []
        for end_index, (key, phrase, weight, language) in self._automaton.iter(padded):
            start_index = end_index - len(key) + 1
            before = padded[start_index - 1] if start_index > 0 else " "
            after = padded[end_index + 1] if end_index + 1 < len(padded) else " "
            if before.isalnum() or after.isalnum():
                continue
            raw_hits.append((start_index - 1, end_index, phrase, weight, language))
        raw_hits.sort(key=lambda item: (item[0], -(item[1] - item[0])))
        spans: list[DetectionSpan] = []
        occupied: list[tuple[int, int]] = []
        for start, end, phrase, weight, language in raw_hits:
            if any(start >= low and end <= high for low, high in occupied):
                continue
            occupied.append((start, end))
            origin_start, origin_end = mapping.original_span(start, end + 1)
            spans.append(
                DetectionSpan(
                    kind=DetectionKind.LEXICAL,
                    start=origin_start,
                    end=origin_end,
                    text=mapping.original[origin_start:origin_end],
                    pattern=phrase,
                    weight=weight,
                    language=language,
                    note="aho-corasick lexicon match",
                )
            )
        spans.sort(key=lambda span: span.start)
        return spans


def naive_find(text: str, patterns: list[tuple[str, str, float, str]]) -> list[tuple[int, str]]:
    haystack = f" {normalize_for_matching(text)} "
    found: list[tuple[int, str]] = []
    for key, phrase, _weight, _language in patterns:
        if not key:
            continue
        start = haystack.find(key)
        while start >= 0:
            before = haystack[start - 1] if start > 0 else " "
            after_index = start + len(key)
            after = haystack[after_index] if after_index < len(haystack) else " "
            if not before.isalnum() and not after.isalnum():
                found.append((start - 1, phrase))
            start = haystack.find(key, start + 1)
    found.sort()
    return found
