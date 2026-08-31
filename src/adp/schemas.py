from __future__ import annotations

import time
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from adp.modes import EpistemicMode


class QueryCategory(StrEnum):
    EVIDENTIARY_REQUEST = "EVIDENTIARY_REQUEST"
    FACTUAL_LOOKUP = "FACTUAL_LOOKUP"
    INTERPRETATION = "INTERPRETATION"
    MECHANISM = "MECHANISM"
    SCENARIO_ANALYSIS = "SCENARIO_ANALYSIS"
    OPEN_SECRET = "OPEN_SECRET"
    PERSONAL_NARRATIVE = "PERSONAL_NARRATIVE"


class DetectionKind(StrEnum):
    LEXICAL = "LEXICAL"
    SEMANTIC = "SEMANTIC"
    STRUCTURAL = "STRUCTURAL"


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str
    content: str


class TurnRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    user_text: str
    language: str | None = None
    history: list[Message] = Field(default_factory=list)
    user_facts: list[str] = Field(default_factory=list)
    request_evidence: bool = False
    created_at: float = Field(default_factory=time.time)


class QueryFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    language: str
    token_count: int
    interrogative_type: str
    request_mood: bool
    first_person_experiential: bool
    recency_marker: bool
    source_demand_hits: list[str] = Field(default_factory=list)
    mechanism_frame: bool
    scenario_frame: bool
    open_secret_frame: bool
    wh_fact_pattern: bool
    imperative_verbs: list[str] = Field(default_factory=list)
    named_entity_candidates: list[str] = Field(default_factory=list)


class Classification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: QueryCategory
    confidence: float = Field(ge=0.0, le=1.0)
    source: str
    rationale: str
    features: QueryFeatures
    underdetermined: bool = False


class LLMClassification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: QueryCategory
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = ""


class RetrievalDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str
    title: str
    snippet: str
    content: str = ""
    rank: int = 0
    score: float = 0.0
    fetched: bool = False
    from_cache: bool = False


class RetrievalBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    executed: bool = False
    degraded: bool = False
    error: str | None = None
    documents: list[RetrievalDocument] = Field(default_factory=list)
    elapsed_s: float = 0.0

    @property
    def has_documents(self) -> bool:
        return len(self.documents) > 0

    def provenance(self) -> list[str]:
        return [f"[{index + 1}] {doc.title} — {doc.url}" for index, doc in enumerate(self.documents)]


class DetectionSpan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: DetectionKind
    start: int
    end: int
    text: str
    pattern: str
    weight: float = 0.0
    similarity: float = 0.0
    language: str = ""
    note: str = ""


class DefensivenessReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lexical_component: float = 0.0
    semantic_component: float = 0.0
    structural_component: float = 0.0
    lexical_weight_sum: float = 0.0
    semantic_max: float = 0.0
    structural_flag: int = 0
    score: float = 0.0
    spans: list[DetectionSpan] = Field(default_factory=list)
    structural_reasons: list[str] = Field(default_factory=list)


class EngagementReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer_first: float = 0.0
    continuity: float = 0.0
    scenario: float = 0.0
    mechanism: float = 0.0
    weights: dict[str, float] = Field(default_factory=dict)
    score: float = 0.0
    hypothesis_count: int = 0
    mechanism_components: dict[str, bool] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)


class FabricationClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entity: str
    entity_type: str
    asserted_act: str
    sentence: str
    traceable: bool
    trace_source: str = ""


class FabricationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: int = 0
    claims: list[FabricationClaim] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class ScoreReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: EpistemicMode
    defensiveness: DefensivenessReport
    engagement: EngagementReport
    fabrication: FabricationReport
    violations: list[str] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.violations


class GenerationAttempt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attempt: int
    temperature: float
    directive_digest: str
    text: str
    scores: ScoreReport
    accepted: bool
    corrective_directive: str = ""


class FallbackHypothesis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    indicators: list[str] = Field(default_factory=list)
    falsifiers: list[str] = Field(default_factory=list)


class FallbackMechanism(BaseModel):
    model_config = ConfigDict(extra="forbid")

    incentives: list[str] = Field(default_factory=list)
    actors: list[str] = Field(default_factory=list)
    patterns: list[str] = Field(default_factory=list)
    consequences: list[str] = Field(default_factory=list)
    conditions: list[str] = Field(default_factory=list)


class FallbackPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    direct_answer: str
    hypotheses: list[FallbackHypothesis] = Field(default_factory=list)
    mechanism: FallbackMechanism = Field(default_factory=FallbackMechanism)


class ExtractedFact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject: str
    predicate: str
    object: str


class ExtractedFactList(BaseModel):
    model_config = ConfigDict(extra="forbid")

    facts: list[ExtractedFact] = Field(default_factory=list)


class AuditRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    turn_id: int
    created_at: float
    user_text: str
    language: str
    category: QueryCategory
    mode: EpistemicMode
    classifier_source: str
    classifier_confidence: float
    retrieval_executed: bool
    retrieval_degraded: bool
    retrieval_doc_count: int
    retrieval_error: str | None
    attempts: list[GenerationAttempt] = Field(default_factory=list)
    fallback_used: bool = False
    final_scores: ScoreReport | None = None
    final_decision: str = ""
    elapsed_s: float = 0.0
    extra: dict[str, Any] = Field(default_factory=dict)


class TurnResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    turn_id: int
    response: str
    mode: EpistemicMode
    category: QueryCategory
    scores: ScoreReport
    retrieval: RetrievalBundle
    attempts_used: int
    fallback_used: bool
    audit: AuditRecord
