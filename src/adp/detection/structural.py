from __future__ import annotations

import regex as re

from adp.modes import EpistemicMode, allows_verification_vocabulary

HEDGE_MARKERS: tuple[str, ...] = (
    "cannot",
    "can not",
    "unable to",
    "not able to",
    "no source",
    "no reliable",
    "no evidence",
    "unverified",
    "unverifiable",
    "not verified",
    "speculate",
    "speculation",
    "responsibly",
    "careful",
    "caution",
    "nem tudom",
    "nem lehet",
    "nem allithatom",
    "nincs forras",
    "nincs bizonyitek",
    "nincs hivatalos",
    "spekulal",
    "ovatos",
    "felelosen",
    "megerositeni",
    "igazolhato",
)

MECHANISM_KEYWORDS: tuple[str, ...] = (
    "incentive",
    "incentives",
    "interest",
    "interests",
    "actor",
    "actors",
    "stakeholder",
    "mechanism",
    "pattern",
    "patterns",
    "consequence",
    "consequences",
    "condition",
    "conditions",
    "structure",
    "dynamic",
    "feedback",
    "erdek",
    "erdekek",
    "szereplo",
    "szereplok",
    "mechanizmus",
    "minta",
    "mintazat",
    "kovetkezmeny",
    "kovetkezmenyek",
    "feltetel",
    "feltetelek",
    "szerkezet",
    "dinamika",
)

CAUSAL_CONNECTIVES: tuple[str, ...] = (
    "because",
    "therefore",
    "hence",
    "thus",
    "so that",
    "as a result",
    "which leads to",
    "leads to",
    "causes",
    "drives",
    "due to",
    "consequently",
    "mert",
    "ezert",
    "tehat",
    "igy",
    "kovetkezteben",
    "ennek eredmenyekent",
    "ami ahhoz vezet",
    "vezet",
    "okozza",
    "miatt",
    "ennelfogva",
)

SCENARIO_MARKERS: tuple[str, ...] = (
    "hypothesis",
    "hypotheses",
    "scenario",
    "scenarios",
    "variant",
    "variants",
    "reading",
    "possibility",
    "alternative",
    "h1",
    "h2",
    "h3",
    "hipotezis",
    "forgatokonyv",
    "valtozat",
    "olvasat",
    "lehetoseg",
    "alternativa",
)

PROOF_DEMAND_PATTERNS: tuple[str, ...] = (
    "can you provide a source",
    "could you provide a source",
    "what is your source",
    "what's your source",
    "do you have a source",
    "do you have proof",
    "do you have evidence",
    "please provide evidence",
    "please provide proof",
    "please cite",
    "prove it",
    "prove this",
    "you need to prove",
    "you would need to show",
    "i would need a source",
    "i would need evidence",
    "i need a source",
    "unless you can prove",
    "van erre forrasod",
    "van forrasod",
    "mi a forrasod",
    "tudsz erre forrast",
    "van bizonyitekod",
    "kerlek igazold",
    "bizonyitsd be",
    "igazolnod kell",
    "forrast kellene",
    "amig nem igazolod",
)


def _count_hits(haystack: str, needles: tuple[str, ...]) -> int:
    return sum(1 for needle in needles if needle in haystack)


def _hit_list(haystack: str, needles: tuple[str, ...]) -> list[str]:
    return [needle for needle in needles if needle in haystack]


_HYPOTHESIS_RE = re.compile(
    r"(?:^|\n)\s*(?:h\s?\d+|hipot[eé]zis\s?\d+|hypothesis\s?\d+|scenario\s?\d+|"
    r"forgat[oó]k[oö]nyv\s?\d+|variant\s?\d+|v[aá]ltozat\s?\d+)\b",
    re.IGNORECASE | re.UNICODE,
)


def count_hypotheses(normalized_text: str) -> int:
    labels = {match.group(0).strip().casefold() for match in _HYPOTHESIS_RE.finditer(normalized_text)}
    if labels:
        return len(labels)
    enumerated = re.findall(
        r"(?:^|\n)\s*(?:\d+[.)]|[-*•])\s+\S", normalized_text, re.UNICODE
    )
    if len(enumerated) >= 2 and _count_hits(normalized_text, SCENARIO_MARKERS) >= 1:
        return len(enumerated)
    marker_hits = _count_hits(normalized_text, SCENARIO_MARKERS)
    if marker_hits >= 2:
        return 2
    if marker_hits == 1:
        return 1
    return 0


class StructuralDetector:
    def __init__(self) -> None:
        self._hedges = HEDGE_MARKERS
        self._mechanism = MECHANISM_KEYWORDS
        self._causal = CAUSAL_CONNECTIVES

    def evaluate(self, normalized_text: str, mode: EpistemicMode) -> tuple[int, list[str]]:
        reasons: list[str] = []
        hedge_hits = _hit_list(normalized_text, self._hedges)
        mechanism_hits = _hit_list(normalized_text, self._mechanism)
        causal_hits = _hit_list(normalized_text, self._causal)
        hypotheses = count_hypotheses(normalized_text)

        analytic_present = bool(mechanism_hits) or bool(causal_hits) or hypotheses >= 2
        if hedge_hits and not analytic_present:
            reasons.append(
                "hedge markers "
                f"{hedge_hits[:5]} co-occur with the absence of analytic markers "
                "(no scenario enumeration, no mechanism keywords, no causal connectives)"
            )

        if not allows_verification_vocabulary(mode):
            proof_hits = _hit_list(normalized_text, PROOF_DEMAND_PATTERNS)
            if proof_hits:
                reasons.append(
                    f"response demands justification or proof from the user ({proof_hits[:5]}) "
                    f"while mode {mode} does not permit verification vocabulary"
                )

        return (1 if reasons else 0), reasons

    @staticmethod
    def analytic_inventory(normalized_text: str) -> dict[str, int]:
        return {
            "hedges": _count_hits(normalized_text, HEDGE_MARKERS),
            "mechanism_keywords": _count_hits(normalized_text, MECHANISM_KEYWORDS),
            "causal_connectives": _count_hits(normalized_text, CAUSAL_CONNECTIVES),
            "scenario_markers": _count_hits(normalized_text, SCENARIO_MARKERS),
            "hypotheses": count_hypotheses(normalized_text),
        }
