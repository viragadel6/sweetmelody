from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Protocol

import regex as re

from adp.detection.normalize import normalize_for_matching
from adp.schemas import DetectionKind, DetectionSpan

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+|\n+", re.UNICODE)


def split_sentences(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    cursor = 0
    for chunk in _SENT_SPLIT_RE.split(text):
        if chunk is None:
            continue
        index = text.find(chunk, cursor)
        if index < 0:
            index = cursor
        stripped = chunk.strip()
        if stripped:
            offset = index + chunk.find(stripped)
            spans.append((offset, offset + len(stripped), stripped))
        cursor = index + len(chunk)
    if not spans and text.strip():
        stripped = text.strip()
        offset = text.find(stripped)
        spans.append((offset, offset + len(stripped), stripped))
    return spans


def load_seeds(seeds_dir: str | Path, languages: tuple[str, ...]) -> dict[str, list[str]]:
    directory = Path(seeds_dir)
    seeds: dict[str, list[str]] = {}
    for language in languages:
        path = directory / f"defensive_seeds.{language}.txt"
        lines = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        if len(lines) < 24:
            raise ValueError(f"seed corpus {path} must contain at least 24 sentences")
        seeds[language] = lines
    return seeds


class EmbeddingBackend(Protocol):
    def encode(self, texts: list[str]) -> list[list[float]]: ...

    @property
    def name(self) -> str: ...


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    length = min(len(left), len(right))
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for index in range(length):
        dot += left[index] * right[index]
        left_norm += left[index] * left[index]
        right_norm += right[index] * right[index]
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (math.sqrt(left_norm) * math.sqrt(right_norm))))


class CharNgramBackend:
    """Deterministic in-process embedding backend based on hashed character n-grams."""

    def __init__(self, ngram_size: int = 4, dimensions: int = 1024) -> None:
        self._n = max(2, ngram_size)
        self._dim = dimensions

    @property
    def name(self) -> str:
        return f"char-ngram-{self._n}-{self._dim}"

    def _vector(self, text: str) -> list[float]:
        normalized = normalize_for_matching(text)
        padded = f" {normalized} "
        counter: Counter[int] = Counter()
        if len(padded) < self._n:
            grams = [padded]
        else:
            grams = [padded[i : i + self._n] for i in range(len(padded) - self._n + 1)]
        for gram in grams:
            counter[hash(gram) % self._dim] += 1
        vector = [0.0] * self._dim
        for index, count in counter.items():
            vector[index] = 1.0 + math.log(float(count))
        norm = math.sqrt(sum(value * value for value in vector))
        if norm > 0.0:
            vector = [value / norm for value in vector]
        return vector

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]


class SentenceTransformerBackend:
    def __init__(self, model_name: str, batch_size: int) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self._batch_size = batch_size
        self._model_name = model_name

    @property
    def name(self) -> str:
        return self._model_name

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [[float(value) for value in row] for row in vectors]


def build_backend(model_name: str, batch_size: int, ngram_size: int) -> EmbeddingBackend:
    try:
        return SentenceTransformerBackend(model_name, batch_size)
    except Exception:
        return CharNgramBackend(ngram_size)


class SemanticDetector:
    def __init__(
        self,
        seeds: dict[str, list[str]],
        backend: EmbeddingBackend,
        low: float,
        high: float,
    ) -> None:
        self._backend = backend
        self._low = low
        self._high = high
        self._seed_texts: list[tuple[str, str]] = []
        for language, sentences in seeds.items():
            for sentence in sentences:
                self._seed_texts.append((language, sentence))
        self._seed_vectors = backend.encode([text for _lang, text in self._seed_texts])

    @property
    def backend_name(self) -> str:
        return self._backend.name

    @property
    def seed_count(self) -> int:
        return len(self._seed_texts)

    def score(self, text: str) -> tuple[float, list[DetectionSpan]]:
        sentences = split_sentences(text)
        if not sentences:
            return 0.0, []
        vectors = self._backend.encode([sentence for _start, _end, sentence in sentences])
        spans: list[DetectionSpan] = []
        max_similarity = 0.0
        for (start, end, sentence), vector in zip(sentences, vectors, strict=True):
            best = 0.0
            best_seed = ""
            best_lang = ""
            for (language, seed_text), seed_vector in zip(
                self._seed_texts, self._seed_vectors, strict=True
            ):
                similarity = _cosine(vector, seed_vector)
                if similarity > best:
                    best = similarity
                    best_seed = seed_text
                    best_lang = language
            max_similarity = max(max_similarity, best)
            if best >= self._low:
                spans.append(
                    DetectionSpan(
                        kind=DetectionKind.SEMANTIC,
                        start=start,
                        end=end,
                        text=sentence,
                        pattern=best_seed,
                        similarity=round(best, 6),
                        language=best_lang,
                        note=f"cosine similarity {best:.3f} >= low threshold {self._low:.2f}",
                    )
                )
        return round(max_similarity, 6), spans

    def normalized_component(self, max_similarity: float) -> float:
        span = self._high - self._low
        if span <= 0.0:
            return 1.0 if max_similarity >= self._high else 0.0
        value = (max_similarity - self._low) / span
        return max(0.0, min(1.0, value))
