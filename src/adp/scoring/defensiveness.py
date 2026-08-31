from __future__ import annotations

from adp.config import AppConfig
from adp.detection.detector import DefensiveDetector
from adp.modes import EpistemicMode
from adp.schemas import DefensivenessReport


class DefensivenessScorer:
    """Thin scoring facade over the detector; keeps the metric formula in one place."""

    def __init__(self, config: AppConfig, detector: DefensiveDetector) -> None:
        self._config = config
        self._detector = detector

    @property
    def threshold(self) -> float:
        return self._config.thresholds.defensiveness_max

    def score(self, text: str, mode: EpistemicMode) -> DefensivenessReport:
        return self._detector.analyze(text, mode)

    def combine(
        self, lexical: float, semantic: float, structural: float
    ) -> float:
        weights = self._config.weights.defensiveness
        return round(
            weights.lexical * lexical
            + weights.semantic * semantic
            + weights.structural * structural,
            6,
        )

    def lexical_from_weight_sum(self, weight_sum: float) -> float:
        saturation = self._config.thresholds.lexicon_saturation
        if saturation <= 0.0:
            return 0.0
        return round(min(1.0, weight_sum / saturation), 6)

    def semantic_from_similarity(self, similarity: float) -> float:
        low = self._config.thresholds.semantic_low
        high = self._config.thresholds.semantic_high
        span = high - low
        if span <= 0.0:
            return 1.0 if similarity >= high else 0.0
        return round(max(0.0, min(1.0, (similarity - low) / span)), 6)

    def failing(self, report: DefensivenessReport, mode: EpistemicMode) -> bool:
        return self._detector.should_regenerate(report, mode)

    def violations(self, report: DefensivenessReport, mode: EpistemicMode) -> list[str]:
        return self._detector.violations(report, mode)
