from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from adp.config import AppConfig, load_config
from adp.detection.detector import DefensiveDetector
from adp.memory.store import MemoryStore
from adp.modes import EpistemicMode
from adp.pipeline import Pipeline
from adp.providers.base import LLMProvider, ReplayProvider
from adp.providers.local_server import build_provider
from adp.retrieval.search import (
    NullSearchProvider,
    RetrievalOrchestrator,
    StaticSearchProvider,
)
from adp.schemas import RetrievalBundle, TurnRequest
from adp.scoring.engagement import EngagementScorer
from adp.scoring.fabrication import FabricationScorer

app = typer.Typer(add_completion=False, help="ADP — analytic-engagement response pipeline")
console = Console()


def _load(config_path: Path | None) -> AppConfig:
    return load_config(config_path)


def _provider(config: AppConfig, replay: Path | None) -> LLMProvider:
    if replay is not None:
        return ReplayProvider.from_file(replay, strict=False)
    return build_provider(config.provider)


def _retrieval(config: AppConfig, search_fixture: Path | None) -> RetrievalOrchestrator:
    if search_fixture is not None:
        provider = StaticSearchProvider.from_payload(
            search_fixture.read_text(encoding="utf-8")
        )
        return RetrievalOrchestrator(config, provider=provider)
    if not config.search.enabled:
        return RetrievalOrchestrator(config, provider=NullSearchProvider())
    return RetrievalOrchestrator(config)


@app.command()
def chat(
    session: str = typer.Option(..., "--session", "-s", help="Session identifier"),
    config_path: Path | None = typer.Option(None, "--config", "-c", help="Config YAML path"),
    replay: Path | None = typer.Option(None, "--replay", help="Replay script JSON for the LLM"),
    search_fixture: Path | None = typer.Option(
        None, "--search-fixture", help="Static search results JSON"
    ),
    text: str | None = typer.Option(None, "--text", "-t", help="Single turn, then exit"),
    show_scores: bool = typer.Option(True, "--scores/--no-scores"),
) -> None:
    config = _load(config_path)
    provider = _provider(config, replay)
    pipeline = Pipeline(
        config,
        provider,
        retrieval=_retrieval(config, search_fixture),
        memory=MemoryStore(config, provider),
        use_llm_classifier=replay is None,
    )
    try:
        if text is not None:
            _run_turn(pipeline, session, text, show_scores)
            return
        console.print("[bold]ADP chat[/bold] — empty line or Ctrl-D exits.")
        while True:
            try:
                line = console.input("[cyan]you>[/cyan] ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print()
                break
            if not line:
                break
            _run_turn(pipeline, session, line, show_scores)
    finally:
        pipeline.close()


def _run_turn(pipeline: Pipeline, session: str, text: str, show_scores: bool) -> None:
    result = pipeline.process(TurnRequest(session_id=session, user_text=text))
    console.print(f"[green]adp>[/green] {result.response}")
    if not show_scores:
        return
    table = Table(title=f"turn {result.turn_id} — {result.mode}", show_lines=False)
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("category", str(result.category))
    table.add_row("D (defensiveness)", f"{result.scores.defensiveness.score:.3f}")
    table.add_row("E (engagement)", f"{result.scores.engagement.score:.3f}")
    table.add_row("F (fabrication)", str(result.scores.fabrication.score))
    table.add_row("attempts", str(result.attempts_used))
    table.add_row("fallback", "yes" if result.fallback_used else "no")
    table.add_row("retrieval docs", str(len(result.retrieval.documents)))
    console.print(table)
    if result.scores.violations:
        for violation in result.scores.violations:
            console.print(f"[yellow]violation:[/yellow] {violation}")


@app.command()
def analyze(
    text: str | None = typer.Option(None, "--text", "-t", help="Text to analyze"),
    file: Path | None = typer.Option(None, "--file", "-f", help="File containing the text"),
    mode: EpistemicMode = typer.Option(EpistemicMode.SCENARIO, "--mode", "-m"),
    query: str = typer.Option("", "--query", "-q", help="Original user query for continuity"),
    lang: str = typer.Option("en", "--lang", "-l"),
    config_path: Path | None = typer.Option(None, "--config", "-c"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    if text is None and file is None:
        console.print("[red]provide --text or --file[/red]")
        raise typer.Exit(code=2)
    payload = text if text is not None else file.read_text(encoding="utf-8")  # type: ignore[union-attr]
    config = _load(config_path)
    detector = DefensiveDetector(config)
    engagement = EngagementScorer(config)
    fabrication = FabricationScorer(config)

    defensiveness_report = detector.analyze(payload, mode)
    engagement_report = engagement.score(query or payload, payload, mode)
    fabrication_report = fabrication.score(
        payload, mode, query, [], "", RetrievalBundle(query=query)
    )

    if as_json:
        console.print_json(
            json.dumps(
                {
                    "language": lang,
                    "mode": str(mode),
                    "defensiveness": defensiveness_report.model_dump(mode="json"),
                    "engagement": engagement_report.model_dump(mode="json"),
                    "fabrication": fabrication_report.model_dump(mode="json"),
                    "regenerate": detector.should_regenerate(defensiveness_report, mode),
                },
                ensure_ascii=False,
            )
        )
        return

    table = Table(title=f"analysis — mode {mode}, language {lang}")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("D", f"{defensiveness_report.score:.3f}")
    table.add_row("  L", f"{defensiveness_report.lexical_component:.3f}")
    table.add_row("  S", f"{defensiveness_report.semantic_component:.3f}")
    table.add_row("  R", str(defensiveness_report.structural_flag))
    table.add_row("E", f"{engagement_report.score:.3f}")
    table.add_row("  a1 answer-first", f"{engagement_report.answer_first:.3f}")
    table.add_row("  a2 continuity", f"{engagement_report.continuity:.3f}")
    table.add_row("  a3 scenario", f"{engagement_report.scenario:.3f}")
    table.add_row("  a4 mechanism", f"{engagement_report.mechanism:.3f}")
    table.add_row("F", str(fabrication_report.score))
    table.add_row("regenerate", "yes" if detector.should_regenerate(defensiveness_report, mode) else "no")
    console.print(table)

    if defensiveness_report.spans:
        spans = Table(title="detected spans")
        spans.add_column("kind")
        spans.add_column("span")
        spans.add_column("pattern")
        spans.add_column("w/sim", justify="right")
        for span in defensiveness_report.spans:
            spans.add_column if False else None
            metric = f"{span.weight:.2f}" if span.weight else f"{span.similarity:.3f}"
            spans.add_row(str(span.kind), span.text[:60], span.pattern[:60], metric)
        console.print(spans)
    for reason in defensiveness_report.structural_reasons:
        console.print(f"[yellow]structural:[/yellow] {reason}")
    for reason in engagement_report.reasons:
        console.print(f"[yellow]engagement:[/yellow] {reason}")
    for reason in fabrication_report.reasons:
        console.print(f"[red]fabrication:[/red] {reason}")


def main() -> None:
    try:
        app()
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
