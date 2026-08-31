from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class SessionRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    created_at: float
    updated_at: float
    turn_count: int = 0


class TurnRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    turn_id: int
    session_id: str
    created_at: float
    role: str
    content: str
    mode: str = ""
    category: str = ""


class FactRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fact_id: int
    session_id: str
    subject: str
    predicate: str
    object: str
    source_turn_id: int
    superseded_by: int | None = None
    created_at: float = 0.0

    def as_sentence(self) -> str:
        return f"{self.subject} {self.predicate} {self.object}".strip()


class Discrepancy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject: str
    predicate: str
    previous_object: str
    new_object: str
    previous_turn_id: int
    new_turn_id: int

    def analytic_note(self) -> str:
        return (
            f"Earlier in this session you described {self.subject} {self.predicate} "
            f"'{self.previous_object}' (turn {self.previous_turn_id}); now it is "
            f"'{self.new_object}' (turn {self.new_turn_id}). Both readings are on the table, so "
            "the analysis below treats the change itself as a signal rather than as an error."
        )


class MemorySnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    turns: list[TurnRow] = Field(default_factory=list)
    facts: list[FactRow] = Field(default_factory=list)
    discrepancies: list[Discrepancy] = Field(default_factory=list)
