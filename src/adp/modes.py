from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class EpistemicMode(StrEnum):
    ASSERTIVE = "ASSERTIVE"
    SCENARIO = "SCENARIO"
    EVIDENTIAL = "EVIDENTIAL"
    RETRIEVAL_AUGMENTED = "RETRIEVAL_AUGMENTED"


class ModeCapabilities(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: EpistemicMode
    allows_verification_vocabulary: bool
    requires_hypotheses: bool
    requires_mechanism: bool
    requires_citations: bool
    triggers_retrieval: bool
    min_hypotheses: int


_MATRIX: dict[EpistemicMode, ModeCapabilities] = {
    EpistemicMode.ASSERTIVE: ModeCapabilities(
        mode=EpistemicMode.ASSERTIVE,
        allows_verification_vocabulary=False,
        requires_hypotheses=False,
        requires_mechanism=True,
        requires_citations=False,
        triggers_retrieval=False,
        min_hypotheses=0,
    ),
    EpistemicMode.SCENARIO: ModeCapabilities(
        mode=EpistemicMode.SCENARIO,
        allows_verification_vocabulary=False,
        requires_hypotheses=True,
        requires_mechanism=True,
        requires_citations=False,
        triggers_retrieval=False,
        min_hypotheses=2,
    ),
    EpistemicMode.EVIDENTIAL: ModeCapabilities(
        mode=EpistemicMode.EVIDENTIAL,
        allows_verification_vocabulary=True,
        requires_hypotheses=False,
        requires_mechanism=False,
        requires_citations=True,
        triggers_retrieval=True,
        min_hypotheses=0,
    ),
    EpistemicMode.RETRIEVAL_AUGMENTED: ModeCapabilities(
        mode=EpistemicMode.RETRIEVAL_AUGMENTED,
        allows_verification_vocabulary=False,
        requires_hypotheses=False,
        requires_mechanism=False,
        requires_citations=False,
        triggers_retrieval=True,
        min_hypotheses=0,
    ),
}


def capabilities(mode: EpistemicMode) -> ModeCapabilities:
    return _MATRIX[mode]


def allows_verification_vocabulary(mode: EpistemicMode) -> bool:
    return _MATRIX[mode].allows_verification_vocabulary


def requires_hypotheses(mode: EpistemicMode) -> bool:
    return _MATRIX[mode].requires_hypotheses


def requires_mechanism(mode: EpistemicMode) -> bool:
    return _MATRIX[mode].requires_mechanism


def requires_citations(mode: EpistemicMode) -> bool:
    return _MATRIX[mode].requires_citations


def triggers_retrieval(mode: EpistemicMode) -> bool:
    return _MATRIX[mode].triggers_retrieval


def all_modes() -> tuple[EpistemicMode, ...]:
    return tuple(_MATRIX.keys())
