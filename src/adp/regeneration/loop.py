from __future__ import annotations

from adp.config import AppConfig
from adp.detection.detector import DefensiveDetector
from adp.director.director import GenerationDirector
from adp.modes import EpistemicMode
from adp.providers.base import LLMProvider, ProviderError
from adp.regeneration.corrective import build_corrective_directive
from adp.regeneration.fallback import FallbackComposer
from adp.schemas import (
    Classification,
    GenerationAttempt,
    RetrievalBundle,
    ScoreReport,
    TurnRequest,
)
from adp.scoring.defensiveness import DefensivenessScorer
from adp.scoring.engagement import EngagementScorer
from adp.scoring.fabrication import FabricationScorer


class RegenerationOutcome:
    def __init__(
        self,
        text: str,
        scores: ScoreReport,
        attempts: list[GenerationAttempt],
        fallback_used: bool,
    ) -> None:
        self.text = text
        self.scores = scores
        self.attempts = attempts
        self.fallback_used = fallback_used


class RegenerationLoop:
    def __init__(
        self,
        config: AppConfig,
        provider: LLMProvider,
        director: GenerationDirector,
        detector: DefensiveDetector,
        engagement: EngagementScorer,
        fabrication: FabricationScorer,
        fallback: FallbackComposer,
    ) -> None:
        self._config = config
        self._provider = provider
        self._director = director
        self._detector = detector
        self._defensiveness = DefensivenessScorer(config, detector)
        self._engagement = engagement
        self._fabrication = fabrication
        self._fallback = fallback

    def _history_text(self, request: TurnRequest) -> str:
        return "\n".join(message.content for message in request.history)

    def score(
        self, request: TurnRequest, text: str, mode: EpistemicMode, retrieval: RetrievalBundle
    ) -> ScoreReport:
        defensiveness = self._defensiveness.score(text, mode)
        engagement = self._engagement.score(request.user_text, text, mode)
        fabrication = self._fabrication.score(
            text,
            mode,
            request.user_text,
            request.user_facts,
            self._history_text(request),
            retrieval,
        )
        violations: list[str] = []
        violations.extend(self._defensiveness.violations(defensiveness, mode))
        violations.extend(self._engagement.violations(engagement))
        violations.extend(self._fabrication.violations(fabrication))
        return ScoreReport(
            mode=mode,
            defensiveness=defensiveness,
            engagement=engagement,
            fabrication=fabrication,
            violations=violations,
        )

    def run(
        self,
        request: TurnRequest,
        classification: Classification,
        mode: EpistemicMode,
        retrieval: RetrievalBundle,
    ) -> RegenerationOutcome:
        attempts: list[GenerationAttempt] = []
        corrective = ""
        max_attempts = self._config.regeneration.max_attempts

        for attempt_index in range(1, max_attempts + 1):
            temperature = self._config.temperature_for_attempt(attempt_index - 1) if (
                attempt_index == 1
            ) else self._config.temperature_for_attempt(attempt_index)
            messages = self._director.build_messages(
                request, classification, mode, retrieval, corrective
            )
            digest = self._director.digest(messages)
            try:
                text = self._provider.generate(messages, temperature)
            except ProviderError as exc:
                empty_scores = self.score(request, "", mode, retrieval)
                attempts.append(
                    GenerationAttempt(
                        attempt=attempt_index,
                        temperature=temperature,
                        directive_digest=digest,
                        text="",
                        scores=empty_scores,
                        accepted=False,
                        corrective_directive=f"provider failure: {exc}",
                    )
                )
                break

            scores = self.score(request, text, mode, retrieval)
            accepted = scores.passed
            attempts.append(
                GenerationAttempt(
                    attempt=attempt_index,
                    temperature=temperature,
                    directive_digest=digest,
                    text=text,
                    scores=scores,
                    accepted=accepted,
                    corrective_directive=corrective,
                )
            )
            if accepted:
                return RegenerationOutcome(text, scores, attempts, False)

            corrective = build_corrective_directive(
                scores,
                mode,
                attempt_index + 1,
                self._engagement.threshold,
                self._defensiveness.threshold,
            )

        fallback_text, _plan = self._fallback.compose(request, classification, mode, retrieval)
        fallback_scores = self.score(request, fallback_text, mode, retrieval)
        attempts.append(
            GenerationAttempt(
                attempt=len(attempts) + 1,
                temperature=self._config.regeneration.fallback_temperature,
                directive_digest="fallback",
                text=fallback_text,
                scores=fallback_scores,
                accepted=True,
                corrective_directive="deterministic fallback composer",
            )
        )
        return RegenerationOutcome(fallback_text, fallback_scores, attempts, True)
