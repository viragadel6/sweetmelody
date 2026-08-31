from __future__ import annotations

from adp.schemas import Classification, QueryCategory, QueryFeatures

RULE_EVIDENTIARY = "rule:source_demand_in_request_mood"
RULE_FACTUAL = "rule:recency_plus_wh_fact"
RULE_PERSONAL = "rule:first_person_experiential"
RULE_OPEN_SECRET = "rule:open_secret_frame"
RULE_MECHANISM = "rule:mechanism_frame"
RULE_SCENARIO = "rule:scenario_frame"
RULE_INTERPRETATION = "rule:interpretive_statement"
RULE_INCONCLUSIVE = "rule:inconclusive"


def apply_rules(features: QueryFeatures) -> Classification | None:
    if features.source_demand_hits and (
        features.request_mood or features.interrogative_type == "direct_question"
    ):
        return Classification(
            category=QueryCategory.EVIDENTIARY_REQUEST,
            confidence=0.97,
            source=RULE_EVIDENTIARY,
            rationale=(
                "source-demand lexicon hits "
                f"{features.source_demand_hits} appear in request mood; this rule overrides all "
                "other precedence levels"
            ),
            features=features,
            underdetermined=False,
        )

    if features.recency_marker and features.wh_fact_pattern:
        return Classification(
            category=QueryCategory.FACTUAL_LOOKUP,
            confidence=0.9,
            source=RULE_FACTUAL,
            rationale="recency marker combined with a wh-fact pattern indicates a factual lookup",
            features=features,
            underdetermined=False,
        )

    if features.first_person_experiential and not features.source_demand_hits:
        return Classification(
            category=QueryCategory.PERSONAL_NARRATIVE,
            confidence=0.88,
            source=RULE_PERSONAL,
            rationale="first-person experiential markers indicate a personal account",
            features=features,
            underdetermined=False,
        )

    if features.open_secret_frame:
        return Classification(
            category=QueryCategory.OPEN_SECRET,
            confidence=0.86,
            source=RULE_OPEN_SECRET,
            rationale="open-secret framing present without an explicit evidence demand",
            features=features,
            underdetermined=False,
        )

    if features.mechanism_frame and not features.scenario_frame:
        return Classification(
            category=QueryCategory.MECHANISM,
            confidence=0.85,
            source=RULE_MECHANISM,
            rationale="'why/how does this work' framing indicates a mechanism question",
            features=features,
            underdetermined=False,
        )

    if features.scenario_frame:
        return Classification(
            category=QueryCategory.SCENARIO_ANALYSIS,
            confidence=0.85,
            source=RULE_SCENARIO,
            rationale=(
                "'what could explain / which is more likely' framing indicates scenario analysis"
            ),
            features=features,
            underdetermined=True,
        )

    if features.interrogative_type == "statement" and features.token_count >= 8:
        return Classification(
            category=QueryCategory.INTERPRETATION,
            confidence=0.6,
            source=RULE_INTERPRETATION,
            rationale="declarative multi-clause input inviting interpretation of a stated premise",
            features=features,
            underdetermined=True,
        )

    return None


def inconclusive(features: QueryFeatures) -> Classification:
    return Classification(
        category=QueryCategory.SCENARIO_ANALYSIS,
        confidence=0.4,
        source=RULE_INCONCLUSIVE,
        rationale="deterministic rules did not fire; scenario analysis is the safe default",
        features=features,
        underdetermined=True,
    )
