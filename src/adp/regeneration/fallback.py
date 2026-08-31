from __future__ import annotations

import json
from pathlib import Path

from adp.config import AppConfig
from adp.detection.detector import DefensiveDetector
from adp.director import prompts
from adp.modes import EpistemicMode
from adp.providers.base import LLMProvider, ProviderError
from adp.schemas import (
    Classification,
    FallbackHypothesis,
    FallbackMechanism,
    FallbackPlan,
    Message,
    RetrievalBundle,
    TurnRequest,
)

PLAN_SYSTEM = (
    "You are the deterministic fallback planner of a response pipeline. You do not write prose. "
    "You output exactly one JSON object with this shape:\n"
    '{"direct_answer": str, "hypotheses": [{"label": str, "indicators": [str], '
    '"falsifiers": [str]}], "mechanism": {"incentives": [str], "actors": [str], '
    '"patterns": [str], "consequences": [str], "conditions": [str]}}\n'
    "Rules:\n"
    "- direct_answer is one or two full sentences stating the substantive answer or the leading "
    "hypothesis. It must not contain any disclaimer, hedge, provenance clause or statement about "
    "your own limitations.\n"
    "- Provide at least two hypotheses, each with at least two indicators and at least one "
    "falsifier.\n"
    "- Every mechanism list must contain at least one concrete item.\n"
    "- Do not assert specific unverified acts by identifiable persons or organisations; speak in "
    "terms of actor types, incentives and structural patterns unless the specific fact is present "
    "in the supplied context.\n"
    "- Never write 'I cannot', 'no reliable source', 'cannot be verified', or their Hungarian "
    "equivalents.\n"
)

MECHANISM_LABELS = {
    "en": {
        "hypothesis": "H",
        "indicators": "Indicators",
        "falsifiers": "Falsifiers",
    },
    "hu": {
        "hypothesis": "H",
        "indicators": "Jelek",
        "falsifiers": "Cáfoló megfigyelések",
    },
}

GENERIC_ANSWER = {
    "en": (
        "The pattern you describe most plausibly runs on the incentive structure below, and the "
        "leading reading is the one whose indicators you can already observe."
    ),
    "hu": (
        "Az általad leírt jelenség legvalószínűbben az alábbi érdekszerkezet szerint működik, és "
        "az a vezető olvasat, amelynek jelei már most megfigyelhetők."
    ),
}

GENERIC_MECHANISM = {
    "en": FallbackMechanism(
        incentives=[
            "the parties who control the decision retain the gains while the costs are "
            "distributed to those without leverage"
        ],
        actors=[
            "decision-holders with discretionary power, intermediaries who route the flow, and "
            "the affected parties with the least exit option"
        ],
        patterns=[
            "the arrangement repeats whenever discretion is high, oversight is thin and the "
            "affected side cannot verify the process"
        ],
        consequences=[
            "the observable outcome drifts steadily in favour of the discretionary side while "
            "the stated rationale stays constant"
        ],
        conditions=[
            "the reading strengthens when the same actor types reappear across cycles and when "
            "the timing of decisions tracks the incentive rather than the stated criterion"
        ],
    ),
    "hu": FallbackMechanism(
        incentives=[
            "a döntést kontrolláló felek megtartják a hasznot, a költség pedig azokra terhelődik, "
            "akiknek nincs alkupozíciójuk"
        ],
        actors=[
            "diszkrecionális jogkörű döntéshozók, a folyamatot irányító közvetítők, és a "
            "legkevesebb kilépési lehetőséggel bíró érintettek"
        ],
        patterns=[
            "a szerkezet mindig újratermelődik, ahol nagy a mérlegelési szabadság, gyenge a "
            "kontroll, és az érintett fél nem tudja ellenőrizni a folyamatot"
        ],
        consequences=[
            "a megfigyelhető kimenet folyamatosan a diszkrecionális oldal felé csúszik, miközben "
            "a hivatalos indoklás változatlan marad"
        ],
        conditions=[
            "az olvasat erősödik, ha ugyanazok a szereplőtípusok térnek vissza ciklusonként, és "
            "ha a döntések időzítése az érdeket követi, nem a kimondott kritériumot"
        ],
    ),
}

GENERIC_HYPOTHESES = {
    "en": [
        FallbackHypothesis(
            label="Structural incentive drift",
            indicators=[
                "the same actor types reappear in each cycle",
                "the timing of decisions tracks the incentive rather than the stated criterion",
            ],
            falsifiers=[
                "decisions diverge from the incentive when the incentive is strongest",
            ],
        ),
        FallbackHypothesis(
            label="Coordination without explicit agreement",
            indicators=[
                "parallel behaviour appears without any traceable instruction",
                "outcomes converge although the parties have no formal link",
            ],
            falsifiers=[
                "behaviour diverges as soon as one party's cost structure changes",
            ],
        ),
    ],
    "hu": [
        FallbackHypothesis(
            label="Szerkezeti érdekelcsúszás",
            indicators=[
                "ciklusonként ugyanazok a szereplőtípusok jelennek meg",
                "a döntések időzítése az érdeket követi, nem a kimondott kritériumot",
            ],
            falsifiers=[
                "a döntések épp akkor térnek el az érdektől, amikor az érdek a legerősebb",
            ],
        ),
        FallbackHypothesis(
            label="Koordináció kimondott megállapodás nélkül",
            indicators=[
                "párhuzamos viselkedés jelenik meg bármilyen visszakövethető utasítás nélkül",
                "a kimenetek egybeesnek, holott a felek között nincs formális kapcsolat",
            ],
            falsifiers=[
                "a viselkedés szétválik, amint az egyik fél költségszerkezete megváltozik",
            ],
        ),
    ],
}


def _load_template(templates_dir: Path, name: str) -> str:
    return (templates_dir / name).read_text(encoding="utf-8")


def _bullets(items: list[str]) -> str:
    cleaned = [item.strip() for item in items if item and item.strip()]
    if not cleaned:
        return "—"
    return "; ".join(cleaned)


def _premise_echo(user_text: str, limit: int = 220) -> str:
    collapsed = " ".join(user_text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1].rstrip() + "…"


class FallbackComposer:
    def __init__(
        self,
        config: AppConfig,
        provider: LLMProvider,
        detector: DefensiveDetector,
    ) -> None:
        self._config = config
        self._provider = provider
        self._detector = detector
        self._templates = {
            "mechanism": _load_template(config.paths.templates_dir, "mechanism.md.tmpl"),
            "scenario": _load_template(config.paths.templates_dir, "scenario.md.tmpl"),
            "evidential": _load_template(config.paths.templates_dir, "evidential.md.tmpl"),
        }

    def _plan_messages(
        self,
        request: TurnRequest,
        classification: Classification,
        mode: EpistemicMode,
        retrieval: RetrievalBundle,
    ) -> list[Message]:
        language = classification.features.language
        context_blocks = [
            f"Language: {language}",
            f"Resolved mode: {mode}",
            f"Resolved category: {classification.category}",
            prompts.continuation_block(
                request.user_text, request.user_facts[: self._config.memory.max_facts_in_context]
            ),
            prompts.retrieval_policy_block(retrieval, mode),
            prompts.fabrication_block(mode),
            "Emit only the JSON object.",
        ]
        return [
            Message(role="system", content=PLAN_SYSTEM),
            Message(role="user", content="\n\n".join(context_blocks)),
        ]

    def _default_plan(self, language: str) -> FallbackPlan:
        lang = language if language in GENERIC_ANSWER else "en"
        return FallbackPlan(
            direct_answer=GENERIC_ANSWER[lang],
            hypotheses=[hypothesis.model_copy(deep=True) for hypothesis in GENERIC_HYPOTHESES[lang]],
            mechanism=GENERIC_MECHANISM[lang].model_copy(deep=True),
        )

    def _sanitize_plan(self, plan: FallbackPlan, language: str) -> FallbackPlan:
        default = self._default_plan(language)
        answer = plan.direct_answer.strip()
        if not answer:
            answer = default.direct_answer
        else:
            report = self._detector.analyze(answer, EpistemicMode.SCENARIO)
            if report.score > self._config.thresholds.defensiveness_max:
                answer = default.direct_answer

        hypotheses: list[FallbackHypothesis] = []
        for hypothesis in plan.hypotheses:
            label = hypothesis.label.strip()
            indicators = [item.strip() for item in hypothesis.indicators if item.strip()]
            falsifiers = [item.strip() for item in hypothesis.falsifiers if item.strip()]
            if not label or not indicators:
                continue
            if not falsifiers:
                falsifiers = default.hypotheses[0].falsifiers
            hypotheses.append(
                FallbackHypothesis(label=label, indicators=indicators, falsifiers=falsifiers)
            )
        while len(hypotheses) < 2:
            hypotheses.append(default.hypotheses[len(hypotheses) % len(default.hypotheses)])

        mechanism = FallbackMechanism(
            incentives=[item.strip() for item in plan.mechanism.incentives if item.strip()]
            or default.mechanism.incentives,
            actors=[item.strip() for item in plan.mechanism.actors if item.strip()]
            or default.mechanism.actors,
            patterns=[item.strip() for item in plan.mechanism.patterns if item.strip()]
            or default.mechanism.patterns,
            consequences=[item.strip() for item in plan.mechanism.consequences if item.strip()]
            or default.mechanism.consequences,
            conditions=[item.strip() for item in plan.mechanism.conditions if item.strip()]
            or default.mechanism.conditions,
        )
        return FallbackPlan(direct_answer=answer, hypotheses=hypotheses, mechanism=mechanism)

    def plan(
        self,
        request: TurnRequest,
        classification: Classification,
        mode: EpistemicMode,
        retrieval: RetrievalBundle,
    ) -> FallbackPlan:
        language = classification.features.language
        try:
            raw_plan = self._provider.generate_structured(
                self._plan_messages(request, classification, mode, retrieval),
                self._config.regeneration.fallback_temperature,
                FallbackPlan,
            )
        except ProviderError:
            return self._default_plan(language)
        return self._sanitize_plan(raw_plan, language)

    def _render_hypotheses(self, plan: FallbackPlan, language: str) -> str:
        labels = MECHANISM_LABELS.get(language, MECHANISM_LABELS["en"])
        blocks: list[str] = []
        for index, hypothesis in enumerate(plan.hypotheses, start=1):
            blocks.append(
                f"{labels['hypothesis']}{index}: {hypothesis.label}\n"
                f"  - {labels['indicators']}: {_bullets(hypothesis.indicators)}\n"
                f"  - {labels['falsifiers']}: {_bullets(hypothesis.falsifiers)}"
            )
        return "\n\n".join(blocks)

    def _render_sources(self, retrieval: RetrievalBundle, plan: FallbackPlan) -> str:
        if retrieval.has_documents:
            lines: list[str] = []
            for index, document in enumerate(retrieval.documents, start=1):
                excerpt = (document.content or document.snippet)[:400].replace("\n", " ").strip()
                lines.append(f"[{index}] {document.title} — {document.url}\n    {excerpt}")
            return "\n".join(lines)
        return (
            "No document is currently attached to these claims; the specific missing datum is the "
            "documentary record itself, while the analysis above rests on the mechanism and the "
            f"conversation context. Leading reading: {plan.direct_answer}"
        )

    def render(
        self,
        plan: FallbackPlan,
        request: TurnRequest,
        mode: EpistemicMode,
        retrieval: RetrievalBundle,
        language: str,
    ) -> str:
        mechanism = plan.mechanism
        common = {
            "direct_answer": plan.direct_answer.strip(),
            "incentives": _bullets(mechanism.incentives),
            "actors": _bullets(mechanism.actors),
            "patterns": _bullets(mechanism.patterns),
            "consequences": _bullets(mechanism.consequences),
            "conditions": _bullets(mechanism.conditions),
            "premise_echo": _premise_echo(request.user_text),
        }
        if mode == EpistemicMode.EVIDENTIAL:
            template = self._templates["evidential"]
            return template.format(**common, source_block=self._render_sources(retrieval, plan))
        if mode == EpistemicMode.SCENARIO:
            template = self._templates["scenario"]
            return template.format(
                **common, hypothesis_block=self._render_hypotheses(plan, language)
            )
        if mode == EpistemicMode.RETRIEVAL_AUGMENTED:
            template = self._templates["mechanism"]
            base = template.format(**common)
            if retrieval.has_documents:
                return base + "\n\nSources consulted:\n" + self._render_sources(retrieval, plan)
            return base
        template = self._templates["mechanism"]
        return template.format(**common)

    def compose(
        self,
        request: TurnRequest,
        classification: Classification,
        mode: EpistemicMode,
        retrieval: RetrievalBundle,
    ) -> tuple[str, FallbackPlan]:
        language = classification.features.language
        plan = self.plan(request, classification, mode, retrieval)
        text = self.render(plan, request, mode, retrieval, language)
        report = self._detector.analyze(text, mode)
        if self._detector.should_regenerate(report, mode):
            safe_plan = self._default_plan(language)
            text = self.render(safe_plan, request, mode, retrieval, language)
            final_report = self._detector.analyze(text, mode)
            if self._detector.should_regenerate(final_report, mode):
                text = self.render(
                    safe_plan, request, EpistemicMode.SCENARIO, RetrievalBundle(query=""), language
                )
            return text, safe_plan
        return text, plan

    def plan_json(self, plan: FallbackPlan) -> str:
        return json.dumps(plan.model_dump(), ensure_ascii=False, indent=2)
