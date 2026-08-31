from __future__ import annotations

from adp.modes import EpistemicMode, requires_hypotheses, requires_mechanism
from adp.schemas import ScoreReport

HEADER = "--- corrective directive (attempt {attempt}) ---"


def _missing_mechanism(report: ScoreReport) -> list[str]:
    return [
        component
        for component, present in report.engagement.mechanism_components.items()
        if not present
    ]


def build_corrective_directive(
    report: ScoreReport,
    mode: EpistemicMode,
    attempt: int,
    engagement_threshold: float,
    defensiveness_threshold: float,
) -> str:
    lines = [HEADER.format(attempt=attempt)]
    lines.append(
        "The previous draft was rejected by deterministic post-checks. Rewrite it completely. "
        "Do not apologise, do not mention this directive, do not explain the rewrite."
    )

    lines.append("Violated invariants and failing metrics:")
    for violation in report.violations:
        lines.append(f"- {violation}")

    if report.defensiveness.spans:
        lines.append("Remove every one of these spans and every paraphrase of them:")
        for span in report.defensiveness.spans[:12]:
            lines.append(f'- "{span.text.strip()[:160]}"')

    if report.defensiveness.score > defensiveness_threshold:
        lines.append(
            f"Defensiveness D={report.defensiveness.score:.3f} must drop to at most "
            f"{defensiveness_threshold:.2f}. Replace hedging about your own limits with actual "
            "analysis: name the incentive, the actor type, the pattern and the consequence."
        )

    if report.defensiveness.structural_reasons:
        lines.append(
            "Structural failure: never demand sources, proof or justification from the user, and "
            "never hedge without accompanying analytic content."
        )

    if report.engagement.score < engagement_threshold:
        lines.append(
            f"Engagement E={report.engagement.score:.3f} must reach at least "
            f"{engagement_threshold:.2f}."
        )
        if report.engagement.answer_first < 1.0:
            lines.append(
                "- Requirement I-6: rewrite the first sentence so that it states the substantive "
                "answer or the leading hypothesis outright, with no disclaimer, hedge or "
                "provenance clause."
            )
        if report.engagement.continuity < 1.0:
            lines.append(
                "- Requirement I-4: quote or paraphrase the user's premise explicitly in the "
                "opening paragraph and extend it with your own reasoning."
            )
        if requires_hypotheses(mode) and report.engagement.hypothesis_count < 2:
            lines.append(
                "- Enumerate at least two hypotheses labelled 'H1:' and 'H2:', each with "
                "supporting indicators and falsifiers."
            )
        if requires_mechanism(mode):
            missing = _missing_mechanism(report)
            if missing:
                lines.append(
                    "- Add explicitly labelled mechanism components that are currently missing: "
                    f"{', '.join(missing)}."
                )

    if report.fabrication.score == 1:
        lines.append(
            "Requirement I-5: the following specific assertions about identifiable entities are "
            "not traceable to the user's context or the retrieval results. Restate them at the "
            "level of actor types, incentives and structural patterns, or drop them:"
        )
        for claim in report.fabrication.claims:
            if not claim.traceable:
                lines.append(
                    f'- "{claim.entity}" asserted to have "{claim.asserted_act}" '
                    f'in: "{claim.sentence[:160]}"'
                )

    lines.append(
        "Keep the full analytic depth: the rewrite must be at least as long and at least as "
        "specific as the rejected draft, minus the rejected formulations."
    )
    return "\n".join(lines)
