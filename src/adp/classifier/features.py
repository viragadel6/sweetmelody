from __future__ import annotations

import regex as re

from adp.schemas import QueryFeatures

SOURCE_DEMAND_LEXICON: dict[str, tuple[str, ...]] = {
    "en": (
        "source",
        "sources",
        "proof",
        "evidence",
        "official confirmation",
        "document",
        "documents",
        "documentation",
        "citation",
        "citations",
        "cite",
        "reference",
        "references",
        "verify",
        "verification",
        "confirm officially",
        "link",
        "links",
        "study",
        "studies",
        "report",
        "court record",
        "statute",
    ),
    "hu": (
        "forras",
        "forrast",
        "forrasok",
        "forrasokat",
        "forrasod",
        "bizonyitek",
        "bizonyitekot",
        "bizonyitekok",
        "hivatalos megerosites",
        "hivatalos megerositest",
        "dokumentum",
        "dokumentumot",
        "dokumentumok",
        "idezet",
        "idezetet",
        "hivatkozas",
        "hivatkozast",
        "igazolas",
        "igazolast",
        "linket",
        "link",
        "tanulmany",
        "tanulmanyt",
        "jelentes",
        "jogszabaly",
    ),
}

REQUEST_MOOD_MARKERS: dict[str, tuple[str, ...]] = {
    "en": (
        "give me",
        "show me",
        "provide",
        "send me",
        "can you provide",
        "could you provide",
        "please cite",
        "please show",
        "i need",
        "i want",
        "list the",
        "where can i find",
        "do you have",
        "what is your source",
    ),
    "hu": (
        "adj",
        "adjon",
        "mutass",
        "mutasson",
        "kuldj",
        "kuldjon",
        "sorolj",
        "soroljon",
        "kerem",
        "kerlek",
        "szeretnek",
        "szeretném",
        "hol talalom",
        "van forrasod",
        "tudsz mutatni",
        "mi a forrasod",
        "igazold",
    ),
}

FIRST_PERSON_EXPERIENTIAL: dict[str, tuple[str, ...]] = {
    "en": (
        "i saw",
        "i heard",
        "i noticed",
        "i remember",
        "i felt",
        "i experienced",
        "i was there",
        "happened to me",
        "my boss",
        "my landlord",
        "my colleague",
        "my neighbour",
        "my neighbor",
        "my doctor",
        "when i was",
        "i went",
        "i received",
        "i got a letter",
        "i live",
        "i work",
    ),
    "hu": (
        "lattam",
        "hallottam",
        "eszrevettem",
        "emlekszem",
        "ereztem",
        "atéltem",
        "ott voltam",
        "velem tortent",
        "a fonokom",
        "a fonoköm",
        "a fonököm",
        "a foberlom",
        "a szomszedom",
        "a kollegam",
        "az orvosom",
        "amikor en",
        "kaptam egy levelet",
        "en lakom",
        "en dolgozom",
        "nekem azt mondtak",
    ),
}

RECENCY_MARKERS: dict[str, tuple[str, ...]] = {
    "en": (
        "today",
        "yesterday",
        "this week",
        "this month",
        "this year",
        "right now",
        "currently",
        "latest",
        "most recent",
        "recently",
        "just now",
        "last week",
        "last month",
        "2024",
        "2025",
        "2026",
    ),
    "hu": (
        "ma",
        "tegnap",
        "ezen a heten",
        "ebben a honapban",
        "iden",
        "most",
        "jelenleg",
        "legutobbi",
        "legfrissebb",
        "nemreg",
        "az elmult heten",
        "az elmult honapban",
        "2024",
        "2025",
        "2026",
    ),
}

MECHANISM_FRAMES: dict[str, tuple[str, ...]] = {
    "en": (
        "how does this work",
        "how does it work",
        "why does this work",
        "how come",
        "why is it that",
        "what makes",
        "how is it possible",
        "what is the mechanism",
        "how do they",
        "why do they",
        "why is this",
        "why are",
        "how can",
        "what drives",
        "what causes",
    ),
    "hu": (
        "hogyan mukodik",
        "hogy mukodik",
        "miert mukodik",
        "hogy lehet az",
        "hogyan lehetseges",
        "mi a mechanizmus",
        "miert van az",
        "miert ennyire",
        "miert olyan",
        "mitol",
        "mi okozza",
        "hogyan tudjak",
        "miert teszik",
        "miert draga",
        "miert nem",
    ),
}

SCENARIO_FRAMES: dict[str, tuple[str, ...]] = {
    "en": (
        "what could explain",
        "what might explain",
        "which is more likely",
        "what are the possibilities",
        "what could be going on",
        "what do you think is happening",
        "could it be that",
        "is it possible that",
        "what are the chances",
        "which scenario",
        "what would you guess",
        "what is your read",
    ),
    "hu": (
        "mi magyarazhatja",
        "mi lehet a magyarazat",
        "melyik a valoszinubb",
        "milyen lehetosegek",
        "mi tortenhet",
        "szerinted mi tortenik",
        "lehet hogy",
        "elkepzelheto hogy",
        "mekkora az eselye",
        "melyik forgatokonyv",
        "mit tippelnel",
        "mi a te olvasatod",
    ),
}

OPEN_SECRET_FRAMES: dict[str, tuple[str, ...]] = {
    "en": (
        "everybody knows",
        "everyone knows",
        "open secret",
        "no one talks about",
        "nobody admits",
        "never officially admitted",
        "unofficially",
        "off the record",
        "it is common knowledge",
        "widely known but",
        "nobody says it out loud",
    ),
    "hu": (
        "mindenki tudja",
        "kozismert",
        "nyilt titok",
        "senki nem beszel rola",
        "senki nem ismeri be",
        "soha nem ismertek be",
        "nem hivatalosan",
        "bizalmasan",
        "koztudott",
        "tudott dolog de",
        "senki nem mondja ki",
    ),
}

WH_FACT_PATTERNS: dict[str, tuple[str, ...]] = {
    "en": (
        "who is",
        "who was",
        "when did",
        "when was",
        "when is",
        "where is",
        "where was",
        "how much",
        "how many",
        "what is the price",
        "what year",
        "what date",
        "what is the current",
    ),
    "hu": (
        "ki a",
        "ki az",
        "ki volt",
        "mikor volt",
        "mikor lesz",
        "mikor tortent",
        "hol van",
        "hol volt",
        "mennyi",
        "mennyibe",
        "hany",
        "milyen aron",
        "melyik evben",
        "mi a jelenlegi",
    ),
}

IMPERATIVE_VERBS: dict[str, tuple[str, ...]] = {
    "en": (
        "give",
        "show",
        "list",
        "cite",
        "provide",
        "send",
        "find",
        "prove",
        "verify",
        "confirm",
    ),
    "hu": (
        "adj",
        "mutass",
        "sorolj",
        "idezz",
        "kuldj",
        "keress",
        "igazold",
        "bizonyitsd",
        "erositsd",
        "nezd",
    ),
}

HU_DIACRITICS = "áéíóöőúüűÁÉÍÓÖŐÚÜŰ"
HU_FUNCTION_WORDS = (
    "hogy",
    "nem",
    "van",
    "mert",
    "csak",
    "meg",
    "azt",
    "ezt",
    "egy",
    "mit",
    "miert",
    "miért",
    "amit",
    "ami",
    "lehet",
    "kell",
    "vagy",
)
EN_FUNCTION_WORDS = (
    "the",
    "and",
    "that",
    "this",
    "with",
    "what",
    "why",
    "how",
    "does",
    "is",
    "are",
    "of",
    "to",
    "for",
    "it",
)

_WORD_RE = re.compile(r"\p{L}[\p{L}\p{M}\-']*", re.UNICODE)
_ENTITY_RE = re.compile(r"\b\p{Lu}\p{L}{2,}(?:\s+\p{Lu}\p{L}{2,})*\b", re.UNICODE)

_FOLD_MAP = str.maketrans(
    {
        "á": "a",
        "é": "e",
        "í": "i",
        "ó": "o",
        "ö": "o",
        "ő": "o",
        "ú": "u",
        "ü": "u",
        "ű": "u",
        "Á": "a",
        "É": "e",
        "Í": "i",
        "Ó": "o",
        "Ö": "o",
        "Ő": "o",
        "Ú": "u",
        "Ü": "u",
        "Ű": "u",
    }
)


def fold(text: str) -> str:
    return text.casefold().translate(_FOLD_MAP)


def tokenize(text: str) -> list[str]:
    return [match.group(0) for match in _WORD_RE.finditer(text)]


def detect_language(text: str, allowed: tuple[str, ...] = ("en", "hu")) -> str:
    lowered = text.casefold()
    hu_score = sum(1 for char in text if char in HU_DIACRITICS) * 2.0
    tokens = set(fold(token) for token in tokenize(lowered))
    hu_score += sum(3.0 for word in HU_FUNCTION_WORDS if fold(word) in tokens)
    en_score = sum(3.0 for word in EN_FUNCTION_WORDS if word in tokens)
    if hu_score > en_score and "hu" in allowed:
        return "hu"
    if "en" in allowed:
        return "en"
    return allowed[0]


def _hits(folded: str, phrases: tuple[str, ...]) -> list[str]:
    return [phrase for phrase in phrases if fold(phrase) in folded]


def _any_hit(folded: str, phrases: tuple[str, ...]) -> bool:
    return any(fold(phrase) in folded for phrase in phrases)


def _imperative_hits(tokens: list[str], verbs: tuple[str, ...]) -> list[str]:
    folded_tokens = [fold(token) for token in tokens]
    if not folded_tokens:
        return []
    head = set(folded_tokens[:4])
    return [verb for verb in verbs if fold(verb) in head]


def extract_features(text: str, language: str | None = None) -> QueryFeatures:
    lang = (language or detect_language(text)).lower()
    if lang not in SOURCE_DEMAND_LEXICON:
        lang = "en"
    folded = fold(text)
    tokens = tokenize(text)

    source_hits = _hits(folded, SOURCE_DEMAND_LEXICON[lang])
    imperatives = _imperative_hits(tokens, IMPERATIVE_VERBS[lang])
    request_mood = bool(imperatives) or _any_hit(folded, REQUEST_MOOD_MARKERS[lang])

    if "?" in text:
        interrogative_type = "direct_question"
    elif request_mood:
        interrogative_type = "request"
    else:
        interrogative_type = "statement"

    entities = [
        match.group(0)
        for match in _ENTITY_RE.finditer(text)
        if match.start() > 0 and len(match.group(0)) > 2
    ]

    return QueryFeatures(
        language=lang,
        token_count=len(tokens),
        interrogative_type=interrogative_type,
        request_mood=request_mood,
        first_person_experiential=_any_hit(folded, FIRST_PERSON_EXPERIENTIAL[lang]),
        recency_marker=_any_hit(folded, RECENCY_MARKERS[lang]),
        source_demand_hits=source_hits,
        mechanism_frame=_any_hit(folded, MECHANISM_FRAMES[lang]),
        scenario_frame=_any_hit(folded, SCENARIO_FRAMES[lang]),
        open_secret_frame=_any_hit(folded, OPEN_SECRET_FRAMES[lang]),
        wh_fact_pattern=_any_hit(folded, WH_FACT_PATTERNS[lang]),
        imperative_verbs=imperatives,
        named_entity_candidates=entities[:12],
    )
