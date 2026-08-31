from __future__ import annotations

from adp.config import AppConfig
from adp.detection.automaton import LexiconAutomaton, load_lexicons
from adp.detection.normalize import normalize_for_matching
from adp.detection.semantic import SemanticDetector, build_backend, load_seeds
from adp.detection.structural import StructuralDetector
from adp.modes import EpistemicMode, allows_verification_vocabulary
from adp.schemas import DefensivenessReport, DetectionSpan


class DefensiveDetector:
    def __init__(
        self,
        config: AppConfig,
        automaton: LexiconAutomaton | None = None,
        semantic: SemanticDetector | None = None,
        structural: StructuralDetector | None = None,
    ) -> None:
        self._config = config
        self._automaton = automaton or LexiconAutomaton(
            load_lexicons(config.paths.lexicon_dir, config.languages)
        )
        if semantic is not None:
            self._semantic = semantic
        else:
            backend = build_backend(
                config.semantic.model_name,
                config.semantic.batch_size,
                config.semantic.char_ngram_size,
            )
            self._semantic = SemanticDetector(
                load_seeds(config.paths.seeds_dir, config.languages),
                backend,
                config.thresholds.semantic_low,
                config.thresholds.semantic_high,
            )
        self._structural = structural or StructuralDetector()

    @property
    def lexicon_pattern_count(self) -> int:
        return self._automaton.pattern_count

    @property
    def semantic_backend(self) -> str:
        return self._semantic.backend_name

    def lexical_component(self, spans: list[DetectionSpan]) -> tuple[float, float]:
        weight_sum = sum(span.weight for span in spans)
        saturation = self._config.thresholds.lexicon_saturation
        component = min(1.0, weight_sum / saturation) if saturation > 0 else 0.0
        return round(component, 6), round(weight_sum, 6)

    def analyze(self, text: str, mode: EpistemicMode) -> DefensivenessReport:
        lexical_spans = self._automaton.find(text)
        semantic_max, semantic_spans = self._semantic.score(text)
        normalized = normalize_for_matching(text)
        structural_flag, structural_reasons = self._structural.evaluate(normalized, mode)

        lexical_component, weight_sum = self.lexical_component(lexical_spans)
        semantic_component = round(self._semantic.normalized_component(semantic_max), 6)

        weights = self._config.weights.defensiveness
        score = (
            weights.lexical * lexical_component
            + weights.semantic * semantic_component
            + weights.structural * float(structural_flag)
        )

        spans = sorted([*lexical_spans, *semantic_spans], key=lambda span: (span.start, span.end))
        return DefensivenessReport(
            lexical_component=lexical_component,
            semantic_component=semantic_component,
            structural_component=float(structural_flag),
            lexical_weight_sum=weight_sum,
            semantic_max=semantic_max,
            structural_flag=structural_flag,
            score=round(score, 6),
            spans=spans,
            structural_reasons=structural_reasons,
        )

    def should_regenerate(self, report: DefensivenessReport, mode: EpistemicMode) -> bool:
        if allows_verification_vocabulary(mode):
            return False
        return report.score > self._config.thresholds.defensiveness_max

    def violations(self, report: DefensivenessReport, mode: EpistemicMode) -> list[str]:
        if not self.should_regenerate(report, mode):
            return []
        details: list[str] = [
            f"I-1 violated: defensiveness D={report.score:.3f} exceeds "
            f"{self._config.thresholds.defensiveness_max:.2f} in mode {mode}"
        ]
        for span in report.spans[:8]:
            details.append(
                f"banned span [{span.kind}] '{span.text[:120]}' matched '{span.pattern[:120]}'"
            )
        details.extend(f"I-3/I-1 structural: {reason}" for reason in report.structural_reasons)
        return details
