from __future__ import annotations

from adp.classifier.features import extract_features
from adp.classifier.llm_classifier import LlmClassifier
from adp.classifier.rules import apply_rules, inconclusive
from adp.config import AppConfig
from adp.modes import EpistemicMode
from adp.providers.base import LLMProvider
from adp.schemas import Classification, QueryCategory

CATEGORY_TO_MODE: dict[QueryCategory, EpistemicMode] = {
    QueryCategory.EVIDENTIARY_REQUEST: EpistemicMode.EVIDENTIAL,
    QueryCategory.FACTUAL_LOOKUP: EpistemicMode.RETRIEVAL_AUGMENTED,
    QueryCategory.INTERPRETATION: EpistemicMode.ASSERTIVE,
    QueryCategory.MECHANISM: EpistemicMode.ASSERTIVE,
    QueryCategory.OPEN_SECRET: EpistemicMode.ASSERTIVE,
    QueryCategory.PERSONAL_NARRATIVE: EpistemicMode.ASSERTIVE,
    QueryCategory.SCENARIO_ANALYSIS: EpistemicMode.SCENARIO,
}


def resolve_mode(classification: Classification) -> EpistemicMode:
    if (
        classification.category == QueryCategory.INTERPRETATION
        and classification.underdetermined
    ):
        return EpistemicMode.SCENARIO
    return CATEGORY_TO_MODE[classification.category]


class ModeResolver:
    def __init__(
        self,
        config: AppConfig,
        provider: LLMProvider | None = None,
        use_llm_fallback: bool = True,
    ) -> None:
        self._config = config
        self._llm = LlmClassifier(provider, config) if provider is not None else None
        self._use_llm_fallback = use_llm_fallback

    def classify(
        self, text: str, language: str | None = None, request_evidence: bool = False
    ) -> Classification:
        features = extract_features(text, language)
        if request_evidence:
            return Classification(
                category=QueryCategory.EVIDENTIARY_REQUEST,
                confidence=1.0,
                source="explicit:request_evidence_flag",
                rationale="caller set request_evidence=True on the turn request",
                features=features,
                underdetermined=False,
            )
        rule_result = apply_rules(features)
        if rule_result is not None and rule_result.source != "rule:interpretive_statement":
            return rule_result
        if self._llm is not None and self._use_llm_fallback:
            llm_result = self._llm.classify(text, features)
            if llm_result.source == "llm":
                return llm_result
            if rule_result is not None:
                return rule_result
            return llm_result
        if rule_result is not None:
            return rule_result
        return inconclusive(features)

    def resolve(
        self, text: str, language: str | None = None, request_evidence: bool = False
    ) -> tuple[Classification, EpistemicMode]:
        classification = self.classify(text, language, request_evidence)
        return classification, resolve_mode(classification)
