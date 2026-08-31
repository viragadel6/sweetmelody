from __future__ import annotations

import math
from collections import Counter

import regex as re

from adp.config import AppConfig
from adp.detection.normalize import normalize_for_matching
from adp.detection.semantic import split_sentences
from adp.detection.structural import count_hypotheses
from adp.modes import EpistemicMode, requires_hypotheses, requires_mechanism
from adp.schemas import EngagementReport

_TOKEN_RE = re.compile(r"\p{L}[\p{L}\p{M}\-']*|\d+", re.UNICODE)

DISCLAIMER_OPENERS: tuple[str, ...] = (
    "i cannot",
    "i can not",
    "i am unable",
    "i am not able",
    "i must note",
    "i should note",
    "i have to note",
    "it is important to note",
    "it should be noted",
    "please note",
    "as an ai",
    "i am just",
    "based on the available",
    "based on available",
    "there is no reliable",
    "there is no official",
    "i do not have access",
    "i must be careful",
    "unfortunately",
    "disclaimer",
    "nem tudom",
    "nem tudok",
    "nem allithatom",
    "nem lehet",
    "meg kell jegyeznem",
    "fontos megjegyezni",
    "megjegyzem",
    "sajnos",
    "mint mesterseges",
    "csak egy nyelvi modell",
    "a rendelkezesre allo",
    "nincs hivatalos",
    "nincs forras",
    "nincs hozzaferesem",
    "ovatosnak kell",
)

PROVENANCE_OPENERS: tuple[str, ...] = (
    "according to the sources",
    "the sources say",
    "search results show",
    "i searched",
    "based on my search",
    "a talalatok szerint",
    "a keresesi eredmenyek",
    "a forrasok szerint",
    "rakerestem",
)

ANSWER_MARKERS: tuple[str, ...] = (
    "the reason",
    "the main driver",
    "the most likely",
    "most likely",
    "the leading hypothesis",
    "the answer is",
    "the short answer",
    "this happens because",
    "it works by",
    "what is going on",
    "a fő ok",
    "a legvaloszinubb",
    "a valasz",
    "roviden",
    "ez azert",
    "a lenyeg",
    "a vezeto hipotezis",
)

MECHANISM_COMPONENTS: dict[str, tuple[str, ...]] = {
    "incentives": (
        "incentive",
        "incentives",
        "interest",
        "interests",
        "profit",
        "cost",
        "benefit",
        "motivation",
        "erdek",
        "erdekek",
        "haszon",
        "koltseg",
        "profit",
        "motivacio",
    ),
    "actors": (
        "actor",
        "actors",
        "player",
        "players",
        "stakeholder",
        "stakeholders",
        "institution",
        "institutions",
        "party",
        "parties",
        "szereplo",
        "szereplok",
        "erintett",
        "erintettek",
        "intezmeny",
        "intezmenyek",
        "fel",
        "felek",
    ),
    "patterns": (
        "pattern",
        "patterns",
        "typical",
        "recurring",
        "repeatedly",
        "routine",
        "regularity",
        "minta",
        "mintazat",
        "tipikus",
        "visszatero",
        "rendszeresen",
        "rutin",
    ),
    "consequences": (
        "consequence",
        "consequences",
        "result",
        "results",
        "outcome",
        "outcomes",
        "effect",
        "effects",
        "impact",
        "kovetkezmeny",
        "kovetkezmenyek",
        "eredmeny",
        "hatas",
        "hatasok",
        "kimenet",
    ),
    "conditions": (
        "condition",
        "conditions",
        "if ",
        "when ",
        "provided that",
        "depends on",
        "plausible",
        "more likely if",
        "feltetel",
        "feltetelek",
        "ha ",
        "amikor ",
        "amennyiben",
        "attol fugg",
        "valoszinubb ha",
    ),
}


def tokens(text: str) -> list[str]:
    return [match.group(0).casefold() for match in _TOKEN_RE.finditer(text)]


def _bag(text: str) -> Counter[str]:
    return Counter(tokens(normalize_for_matching(text)))


def cosine_similarity_bags(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    shared = set(left) & set(right)
    dot = sum(left[key] * right[key] for key in shared)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


class EngagementScorer:
    def __init__(self, config: AppConfig) -> None:
        self._config = config

    @property
    def threshold(self) -> float:
        return self._config.thresholds.engagement_min

    def _weights(self, mode: EpistemicMode) -> dict[str, float]:
        base = self._config.weights.engagement
        weights = {
            "answer_first": base.answer_first,
            "continuity": base.continuity,
            "scenario": base.scenario,
            "mechanism": base.mechanism,
        }
        redistribute = 0.0
        if not requires_hypotheses(mode):
            redistribute += weights["scenario"]
            weights["scenario"] = 0.0
        if not requires_mechanism(mode):
            redistribute += weights["mechanism"]
            weights["mechanism"] = 0.0
        if redistribute > 0.0:
            anchor_total = weights["answer_first"] + weights["continuity"]
            if anchor_total <= 0.0:
                weights["answer_first"] = 1.0
            else:
                weights["answer_first"] += redistribute * (weights["answer_first"] / anchor_total)
                weights["continuity"] += redistribute * (weights["continuity"] / anchor_total)
        total = sum(weights.values())
        if total > 0.0:
            weights = {key: value / total for key, value in weights.items()}
        return {key: round(value, 6) for key, value in weights.items()}

    def answer_first_score(self, text: str) -> tuple[float, list[str]]:
        reasons: list[str] = []
        head_tokens = tokens(text)[:40]
        if not head_tokens:
            return 0.0, ["I-6 violated: response is empty"]
        window = normalize_for_matching(" ".join(head_tokens))
        sentences = split_sentences(text)
        first_sentence = normalize_for_matching(sentences[0][2]) if sentences else window

        for opener in DISCLAIMER_OPENERS:
            if first_sentence.startswith(opener) or window.startswith(opener):
                reasons.append(
                    f"I-6 violated: first sentence opens with the disclaimer '{opener}'"
                )
                return 0.0, reasons
        for opener in PROVENANCE_OPENERS:
            if first_sentence.startswith(opener):
                reasons.append(
                    f"I-6 violated: first sentence opens with a provenance statement '{opener}'"
                )
                return 0.0, reasons

        score = 0.5
        if any(marker in window for marker in ANSWER_MARKERS):
            score = 1.0
        elif len(tokens(sentences[0][2] if sentences else text)) >= 8:
            score = 0.85
        else:
            reasons.append(
                "I-6 partially satisfied: opening sentence is short and lacks an explicit "
                "answer marker"
            )
        contains_hedge = any(
            hedge in first_sentence
            for hedge in ("cannot", "unable", "no source", "nem tudom", "nem lehet")
        )
        if contains_hedge:
            reasons.append("I-6 violated: hedge inside the first sentence")
            return 0.0, reasons
        return round(score, 6), reasons

    def continuity_score(self, query: str, response: str) -> tuple[float, list[str]]:
        reasons: list[str] = []
        query_bag = _bag(query)
        response_bag = _bag(response)
        if not query_bag:
            return 1.0, reasons
        similarity = cosine_similarity_bags(query_bag, response_bag)
        content_terms = [term for term in query_bag if len(term) > 3]
        if content_terms:
            covered = sum(1 for term in content_terms if term in response_bag)
            coverage = covered / float(len(content_terms))
        else:
            coverage = 1.0
        raw = (0.5 * similarity) + (0.5 * coverage)
        mapped = min(1.0, raw / 0.6) if raw < 0.6 else 1.0
        if mapped < self._config.thresholds.continuity_min:
            reasons.append(
                "I-4 violated: response does not reference or extend the user's premise "
                f"(continuity {mapped:.3f} < {self._config.thresholds.continuity_min:.2f})"
            )
        return round(mapped, 6), reasons

    def scenario_score(self, text: str) -> tuple[float, int, list[str]]:
        normalized = normalize_for_matching(text)
        count = count_hypotheses(text) or count_hypotheses(normalized)
        if count >= 2:
            return 1.0, count, []
        if count == 1:
            return 0.5, count, ["I-5 partially satisfied: only one hypothesis enumerated"]
        return 0.0, count, ["I-5 violated: no hypotheses enumerated"]

    def mechanism_score(self, text: str) -> tuple[float, dict[str, bool], list[str]]:
        normalized = normalize_for_matching(text)
        present: dict[str, bool] = {}
        for component, markers in MECHANISM_COMPONENTS.items():
            present[component] = any(marker in normalized for marker in markers)
        found = sum(1 for value in present.values() if value)
        fraction = found / float(len(MECHANISM_COMPONENTS))
        reasons: list[str] = []
        missing = [key for key, value in present.items() if not value]
        if missing:
            reasons.append(f"I-5 incomplete: missing mechanism components {missing}")
        return round(fraction, 6), present, reasons

    def score(self, query: str, response: str, mode: EpistemicMode) -> EngagementReport:
        weights = self._weights(mode)
        reasons: list[str] = []

        answer_first, answer_reasons = self.answer_first_score(response)
        reasons.extend(answer_reasons)

        continuity, continuity_reasons = self.continuity_score(query, response)
        reasons.extend(continuity_reasons)

        if weights["scenario"] > 0.0:
            scenario, hypothesis_count, scenario_reasons = self.scenario_score(response)
            reasons.extend(scenario_reasons)
        else:
            scenario, hypothesis_count, _ = self.scenario_score(response)
            scenario = 0.0

        if weights["mechanism"] > 0.0:
            mechanism, components, mechanism_reasons = self.mechanism_score(response)
            reasons.extend(mechanism_reasons)
        else:
            mechanism, components, _ = self.mechanism_score(response)
            mechanism = 0.0

        score = (
            weights["answer_first"] * answer_first
            + weights["continuity"] * continuity
            + weights["scenario"] * scenario
            + weights["mechanism"] * mechanism
        )
        return EngagementReport(
            answer_first=answer_first,
            continuity=continuity,
            scenario=scenario,
            mechanism=mechanism,
            weights=weights,
            score=round(score, 6),
            hypothesis_count=hypothesis_count,
            mechanism_components=components,
            reasons=reasons,
        )

    def failing(self, report: EngagementReport) -> bool:
        return report.score < self.threshold

    def violations(self, report: EngagementReport) -> list[str]:
        if not self.failing(report):
            return []
        details = [
            f"engagement E={report.score:.3f} below threshold {self.threshold:.2f} "
            f"(a1={report.answer_first:.2f} a2={report.continuity:.2f} "
            f"a3={report.scenario:.2f} a4={report.mechanism:.2f})"
        ]
        details.extend(report.reasons)
        return details
