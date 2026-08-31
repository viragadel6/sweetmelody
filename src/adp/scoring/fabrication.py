from __future__ import annotations

import regex as re

from adp.config import AppConfig
from adp.detection.normalize import normalize_for_matching
from adp.detection.semantic import split_sentences
from adp.modes import EpistemicMode
from adp.schemas import FabricationClaim, FabricationReport, RetrievalBundle

_ENTITY_RE = re.compile(
    r"\b(?:\p{Lu}\p{L}+(?:\s+(?:of|the|and|for|von|van|de|da|Kft|Zrt|Nyrt|Bt)\b)?\s*)"
    r"(?:\p{Lu}\p{L}+|\p{Lu}{2,})(?:\s+\p{Lu}\p{L}+)*\b",
    re.UNICODE,
)
_ACRONYM_RE = re.compile(r"\b\p{Lu}{3,}\b", re.UNICODE)

ORG_SUFFIXES: tuple[str, ...] = (
    "inc",
    "ltd",
    "llc",
    "gmbh",
    "corp",
    "corporation",
    "company",
    "group",
    "holdings",
    "bank",
    "ministry",
    "agency",
    "authority",
    "council",
    "committee",
    "university",
    "institute",
    "foundation",
    "association",
    "party",
    "kft",
    "zrt",
    "nyrt",
    "bt",
    "hivatal",
    "miniszterium",
    "onkormanyzat",
    "bizottsag",
    "egyetem",
    "intezet",
    "alapitvany",
    "szovetseg",
    "part",
    "bank",
)

SPECIFIC_ACT_VERBS: tuple[str, ...] = (
    "paid",
    "bribed",
    "stole",
    "embezzled",
    "laundered",
    "falsified",
    "forged",
    "signed",
    "ordered",
    "instructed",
    "concealed",
    "destroyed",
    "transferred",
    "awarded",
    "received",
    "threatened",
    "fired",
    "blackmailed",
    "colluded",
    "conspired",
    "defrauded",
    "misappropriated",
    "leaked",
    "arranged",
    "authorised",
    "authorized",
    "fizetett",
    "megvesztegette",
    "ellopott",
    "elsikkasztott",
    "tisztara mosta",
    "meghamisitotta",
    "hamisitott",
    "alairta",
    "elrendelte",
    "utasitotta",
    "eltitkolta",
    "megsemmisitette",
    "atutalta",
    "odaadta",
    "atvette",
    "megfenyegette",
    "kirugta",
    "zsarolta",
    "osszejatszott",
    "osszeeskudott",
    "megteveszto",
    "kiszivarogtatta",
    "elintezte",
    "engedelyezte",
)

HEDGED_FRAME_MARKERS: tuple[str, ...] = (
    "typically",
    "generally",
    "in general",
    "often",
    "usually",
    "tends to",
    "may",
    "might",
    "could",
    "would",
    "if ",
    "when ",
    "hypothesis",
    "scenario",
    "in such arrangements",
    "actors of this type",
    "tipikusan",
    "altalaban",
    "gyakran",
    "szokott",
    "hajlamos",
    "lehet",
    "lehetne",
    "ha ",
    "amikor ",
    "hipotezis",
    "forgatokonyv",
    "az ilyen",
    "ilyen tipusu",
)

GENERIC_TOKENS: frozenset[str] = frozenset(
    {
        "i",
        "the",
        "this",
        "that",
        "these",
        "those",
        "we",
        "you",
        "they",
        "it",
        "hypothesis",
        "scenario",
        "mechanism",
        "incentives",
        "actors",
        "patterns",
        "consequences",
        "conditions",
        "synthesis",
        "claims",
        "indicators",
        "falsifiers",
        "en",
        "ez",
        "az",
        "ok",
        "mi",
        "te",
        "ok",
        "hipotezis",
        "forgatokonyv",
        "mechanizmus",
        "erdekek",
        "szereplok",
        "minta",
        "kovetkezmenyek",
        "feltetelek",
        "jelek",
        "cafolat",
        "hipotezisek",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    }
)

MODES_UNDER_I5 = frozenset({EpistemicMode.SCENARIO, EpistemicMode.ASSERTIVE})


def _classify_entity(surface: str) -> str:
    folded = normalize_for_matching(surface)
    parts = folded.split()
    if any(part in ORG_SUFFIXES for part in parts):
        return "ORG"
    if surface.isupper() and len(surface) >= 3:
        return "ORG"
    if len(parts) >= 2 and all(part[:1].isalpha() for part in parts):
        return "PERSON"
    return "ORG" if len(parts) == 1 and surface[:1].isupper() else "PERSON"


def extract_entities(sentence: str) -> list[tuple[str, str]]:
    found: dict[str, str] = {}
    for match in _ENTITY_RE.finditer(sentence):
        surface = match.group(0).strip()
        if match.start() == 0 and len(surface.split()) == 1:
            continue
        folded = normalize_for_matching(surface)
        if not folded or folded in GENERIC_TOKENS:
            continue
        if all(token in GENERIC_TOKENS for token in folded.split()):
            continue
        if len(folded) < 3:
            continue
        found[surface] = _classify_entity(surface)
    for match in _ACRONYM_RE.finditer(sentence):
        surface = match.group(0)
        folded = normalize_for_matching(surface)
        if folded in GENERIC_TOKENS or len(folded) < 3:
            continue
        found.setdefault(surface, "ORG")
    return sorted(found.items())


def _asserted_acts(normalized_sentence: str) -> list[str]:
    return [verb for verb in SPECIFIC_ACT_VERBS if verb in normalized_sentence]


def _is_hedged(normalized_sentence: str) -> bool:
    return any(marker in normalized_sentence for marker in HEDGED_FRAME_MARKERS)


class FabricationScorer:
    def __init__(self, config: AppConfig) -> None:
        self._config = config

    def _trace_sources(
        self,
        user_text: str,
        user_facts: list[str],
        history_text: str,
        retrieval: RetrievalBundle,
    ) -> list[tuple[str, str]]:
        sources: list[tuple[str, str]] = [
            ("user_text", normalize_for_matching(user_text)),
            ("user_facts", normalize_for_matching(" \n ".join(user_facts))),
            ("history", normalize_for_matching(history_text)),
        ]
        for index, document in enumerate(retrieval.documents):
            blob = f"{document.title}\n{document.snippet}\n{document.content}"
            sources.append((f"retrieval[{index + 1}] {document.url}", normalize_for_matching(blob)))
        return sources

    def score(
        self,
        response: str,
        mode: EpistemicMode,
        user_text: str,
        user_facts: list[str],
        history_text: str,
        retrieval: RetrievalBundle,
    ) -> FabricationReport:
        if mode not in MODES_UNDER_I5:
            return FabricationReport(score=0, claims=[], reasons=[])

        sources = self._trace_sources(user_text, user_facts, history_text, retrieval)
        claims: list[FabricationClaim] = []
        reasons: list[str] = []

        for _start, _end, sentence in split_sentences(response):
            normalized_sentence = normalize_for_matching(sentence)
            acts = _asserted_acts(normalized_sentence)
            if not acts:
                continue
            if _is_hedged(normalized_sentence):
                continue
            for entity, entity_type in extract_entities(sentence):
                folded_entity = normalize_for_matching(entity)
                trace_source = ""
                for name, blob in sources:
                    if folded_entity and folded_entity in blob:
                        trace_source = name
                        break
                claim = FabricationClaim(
                    entity=entity,
                    entity_type=entity_type,
                    asserted_act=acts[0],
                    sentence=sentence[:400],
                    traceable=bool(trace_source),
                    trace_source=trace_source,
                )
                claims.append(claim)
                if not claim.traceable:
                    reasons.append(
                        f"I-5 violated: {entity_type} '{entity}' is asserted to have "
                        f"'{acts[0]}' with no trace to user context or retrieved documents "
                        f"(sentence: {sentence[:160]})"
                    )

        score = 1 if reasons else 0
        return FabricationReport(score=score, claims=claims, reasons=reasons)

    @staticmethod
    def failing(report: FabricationReport) -> bool:
        return report.score == 1

    @staticmethod
    def violations(report: FabricationReport) -> list[str]:
        return list(report.reasons)
