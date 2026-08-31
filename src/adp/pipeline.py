from __future__ import annotations

import time

from adp.audit.log import AuditLog, TraceHook
from adp.classifier.resolver import ModeResolver, resolve_mode
from adp.config import AppConfig, load_config
from adp.detection.detector import DefensiveDetector
from adp.director.director import GenerationDirector
from adp.memory.store import MemoryStore
from adp.modes import EpistemicMode, triggers_retrieval
from adp.providers.base import LLMProvider
from adp.providers.local_server import build_provider
from adp.regeneration.fallback import FallbackComposer
from adp.regeneration.loop import RegenerationLoop
from adp.retrieval.search import RetrievalOrchestrator
from adp.schemas import AuditRecord, RetrievalBundle, TurnRequest, TurnResult
from adp.scoring.engagement import EngagementScorer
from adp.scoring.fabrication import FabricationScorer


class Pipeline:
    def __init__(
        self,
        config: AppConfig,
        provider: LLMProvider,
        retrieval: RetrievalOrchestrator | None = None,
        memory: MemoryStore | None = None,
        audit: AuditLog | None = None,
        detector: DefensiveDetector | None = None,
        use_llm_classifier: bool = True,
        trace_hook: TraceHook | None = None,
    ) -> None:
        self._config = config
        self._provider = provider
        self._retrieval = retrieval if retrieval is not None else RetrievalOrchestrator(config)
        self._memory = memory if memory is not None else MemoryStore(config, provider)
        self._audit = audit if audit is not None else AuditLog(config, trace_hook)
        self._detector = detector if detector is not None else DefensiveDetector(config)
        self._resolver = ModeResolver(config, provider, use_llm_classifier)
        self._director = GenerationDirector(config)
        self._engagement = EngagementScorer(config)
        self._fabrication = FabricationScorer(config)
        self._fallback = FallbackComposer(config, provider, self._detector)
        self._loop = RegenerationLoop(
            config,
            provider,
            self._director,
            self._detector,
            self._engagement,
            self._fabrication,
            self._fallback,
        )

    @property
    def config(self) -> AppConfig:
        return self._config

    @property
    def audit_log(self) -> AuditLog:
        return self._audit

    @property
    def memory(self) -> MemoryStore:
        return self._memory

    def close(self) -> None:
        self._retrieval.close()
        self._memory.close()

    def _retrieve(self, request: TurnRequest, mode: EpistemicMode) -> RetrievalBundle:
        if not triggers_retrieval(mode):
            return RetrievalBundle(
                query=request.user_text,
                executed=False,
                degraded=False,
                error=None,
                documents=[],
                elapsed_s=0.0,
            )
        return self._retrieval.retrieve(request.user_text)

    def process(self, request: TurnRequest) -> TurnResult:
        started = time.monotonic()
        self._memory.ensure_session(request.session_id)

        history = request.history or self._memory.history(
            request.session_id, self._config.memory.max_context_turns
        )
        turn_id = self._memory.append_turn(request.session_id, "user", request.user_text)

        stored_facts, discrepancies = self._memory.extract_and_store_facts(
            request.session_id, request.user_text, turn_id
        )
        active_facts = self._memory.active_facts(
            request.session_id, self._config.memory.max_facts_in_context
        )
        fact_sentences = [fact.as_sentence() for fact in active_facts]
        fact_sentences.extend(discrepancy.analytic_note() for discrepancy in discrepancies)
        merged_facts = list(dict.fromkeys([*request.user_facts, *fact_sentences]))

        effective_request = request.model_copy(
            update={"history": history, "user_facts": merged_facts}
        )

        classification = self._resolver.classify(
            effective_request.user_text,
            effective_request.language,
            effective_request.request_evidence,
        )
        mode = resolve_mode(classification)
        self._audit.trace(
            "pipeline.classified",
            {
                "session_id": request.session_id,
                "turn_id": turn_id,
                "category": str(classification.category),
                "mode": str(mode),
                "source": classification.source,
            },
        )

        retrieval = self._retrieve(effective_request, mode)
        self._audit.trace(
            "pipeline.retrieved",
            {
                "session_id": request.session_id,
                "turn_id": turn_id,
                "executed": retrieval.executed,
                "documents": len(retrieval.documents),
                "error": retrieval.error or "",
            },
        )

        outcome = self._loop.run(effective_request, classification, mode, retrieval)

        self._memory.append_turn(
            request.session_id,
            "assistant",
            outcome.text,
            mode=str(mode),
            category=str(classification.category),
        )

        elapsed = round(time.monotonic() - started, 6)
        decision = "fallback_composer" if outcome.fallback_used else "accepted_generation"
        record = AuditRecord(
            session_id=request.session_id,
            turn_id=turn_id,
            created_at=request.created_at,
            user_text=request.user_text,
            language=classification.features.language,
            category=classification.category,
            mode=mode,
            classifier_source=classification.source,
            classifier_confidence=classification.confidence,
            retrieval_executed=retrieval.executed,
            retrieval_degraded=retrieval.degraded,
            retrieval_doc_count=len(retrieval.documents),
            retrieval_error=retrieval.error,
            attempts=outcome.attempts,
            fallback_used=outcome.fallback_used,
            final_scores=outcome.scores,
            final_decision=decision,
            elapsed_s=elapsed,
            extra={
                "stored_facts": [fact.as_sentence() for fact in stored_facts],
                "discrepancies": [
                    discrepancy.model_dump() for discrepancy in discrepancies
                ],
                "classifier_rationale": classification.rationale,
                "detector_lexicon_patterns": self._detector.lexicon_pattern_count,
                "semantic_backend": self._detector.semantic_backend,
                "retrieval_provenance": retrieval.provenance(),
            },
        )
        self._audit.write(record)

        return TurnResult(
            session_id=request.session_id,
            turn_id=turn_id,
            response=outcome.text,
            mode=mode,
            category=classification.category,
            scores=outcome.scores,
            retrieval=retrieval,
            attempts_used=len(outcome.attempts),
            fallback_used=outcome.fallback_used,
            audit=record,
        )

    def process_turn(self, session_id: str, user_text: str) -> TurnResult:
        return self.process(TurnRequest(session_id=session_id, user_text=user_text))


_DEFAULT_PIPELINE: Pipeline | None = None


def build_default_pipeline(config: AppConfig | None = None) -> Pipeline:
    resolved = config or load_config()
    provider = build_provider(resolved.provider)
    return Pipeline(resolved, provider)


def get_default_pipeline() -> Pipeline:
    global _DEFAULT_PIPELINE
    if _DEFAULT_PIPELINE is None:
        _DEFAULT_PIPELINE = build_default_pipeline()
    return _DEFAULT_PIPELINE


def set_default_pipeline(pipeline: Pipeline | None) -> None:
    global _DEFAULT_PIPELINE
    _DEFAULT_PIPELINE = pipeline


def process_turn(session_id: str, user_text: str) -> TurnResult:
    return get_default_pipeline().process_turn(session_id, user_text)
