from __future__ import annotations

import json

from adp.config import AppConfig
from adp.providers.base import LLMProvider, ProviderError
from adp.schemas import Classification, LLMClassification, Message, QueryCategory, QueryFeatures

SYSTEM_PROMPT = (
    "You are a deterministic query classifier inside a response pipeline. "
    "You never answer the user's question. You only assign exactly one category label. "
    "Categories and their definitions:\n"
    "EVIDENTIARY_REQUEST: the user explicitly demands sources, proof, documents, citations "
    "or official confirmation.\n"
    "FACTUAL_LOOKUP: the user asks for a discrete current fact (name, date, number, price, "
    "statute) that changes over time.\n"
    "INTERPRETATION: the user states a premise or observation and wants it interpreted.\n"
    "MECHANISM: the user asks why or how something works, what drives or causes it.\n"
    "SCENARIO_ANALYSIS: the situation is underdetermined; several explanations compete.\n"
    "OPEN_SECRET: the topic is socially known but has no official admission.\n"
    "PERSONAL_NARRATIVE: the user recounts their own experience, memory or observation.\n"
    "Return only a JSON object with keys 'category', 'confidence' (0..1) and 'rationale'."
)

USER_TEMPLATE = (
    "User turn (language={language}):\n"
    "---\n{text}\n---\n"
    "Deterministic feature extraction:\n{features}\n"
    "Assign exactly one category."
)


class LlmClassifier:
    def __init__(self, provider: LLMProvider, config: AppConfig) -> None:
        self._provider = provider
        self._config = config

    def classify(self, text: str, features: QueryFeatures) -> Classification:
        payload = features.model_dump()
        messages = [
            Message(role="system", content=SYSTEM_PROMPT),
            Message(
                role="user",
                content=USER_TEMPLATE.format(
                    language=features.language,
                    text=text.strip(),
                    features=json.dumps(payload, ensure_ascii=False, indent=2),
                ),
            ),
        ]
        try:
            result = self._provider.generate_structured(messages, 0.0, LLMClassification)
        except ProviderError as exc:
            return Classification(
                category=QueryCategory.SCENARIO_ANALYSIS,
                confidence=0.4,
                source="llm:unavailable",
                rationale=f"structured classification unavailable ({exc}); defaulted to scenario",
                features=features,
                underdetermined=True,
            )

        if result.confidence < self._config.thresholds.classifier_confidence_min:
            return Classification(
                category=QueryCategory.SCENARIO_ANALYSIS,
                confidence=result.confidence,
                source="llm:low_confidence",
                rationale=(
                    f"model proposed {result.category} with confidence {result.confidence:.2f} "
                    "below threshold; defaulted to scenario analysis"
                ),
                features=features,
                underdetermined=True,
            )

        underdetermined = result.category in {
            QueryCategory.SCENARIO_ANALYSIS,
            QueryCategory.INTERPRETATION,
        }
        return Classification(
            category=result.category,
            confidence=result.confidence,
            source="llm",
            rationale=result.rationale or "structured model classification",
            features=features,
            underdetermined=underdetermined,
        )
