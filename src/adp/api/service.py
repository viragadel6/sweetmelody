from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from adp import __version__
from adp.pipeline import Pipeline, get_default_pipeline, set_default_pipeline
from adp.schemas import Message, TurnRequest, TurnResult


class TurnPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1, max_length=200)
    user_text: str = Field(min_length=1)
    language: str | None = None
    request_evidence: bool = False
    history: list[Message] = Field(default_factory=list)
    user_facts: list[str] = Field(default_factory=list)


class HealthPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    version: str
    lexicon_patterns: int
    semantic_backend: str
    languages: list[str]


def create_app(pipeline: Pipeline | None = None) -> FastAPI:
    app = FastAPI(title="ADP", version=__version__)
    if pipeline is not None:
        set_default_pipeline(pipeline)

    def resolve() -> Pipeline:
        return pipeline if pipeline is not None else get_default_pipeline()

    @app.get("/health", response_model=HealthPayload)
    def health() -> HealthPayload:
        current = resolve()
        detector = current._detector  # noqa: SLF001
        return HealthPayload(
            status="ok",
            version=__version__,
            lexicon_patterns=detector.lexicon_pattern_count,
            semantic_backend=detector.semantic_backend,
            languages=list(current.config.languages),
        )

    @app.post("/v1/turns", response_model=TurnResult)
    def create_turn(payload: TurnPayload) -> TurnResult:
        current = resolve()
        request = TurnRequest(
            session_id=payload.session_id,
            user_text=payload.user_text,
            language=payload.language,
            history=payload.history,
            user_facts=payload.user_facts,
            request_evidence=payload.request_evidence,
        )
        try:
            return current.process(request)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"pipeline failure: {exc}") from exc

    @app.get("/v1/sessions/{session_id}/audit")
    def session_audit(session_id: str) -> dict[str, Any]:
        current = resolve()
        records = current.audit_log.read_raw(session_id)
        if not records:
            raise HTTPException(status_code=404, detail="no audit records for session")
        return {"session_id": session_id, "records": records}

    return app


app = create_app()
