from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DEFAULT_CONFIG_PATH = Path("config/default.yaml")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ProviderConfig(StrictModel):
    kind: Literal["http_chat", "local_server", "replay"] = "http_chat"
    base_url: str = "http://localhost:8080/v1"
    api_key_env: str = "ADP_API_KEY"
    model: str = "gpt-4o-mini"
    timeout_s: float = Field(default=60.0, gt=0.0, le=600.0)
    max_output_tokens: int = Field(default=2048, gt=0, le=32768)
    structured_repair_retries: int = Field(default=2, ge=0, le=5)

    @property
    def api_key(self) -> str:
        return os.environ.get(self.api_key_env, "")


class SearchConfig(StrictModel):
    enabled: bool = True
    kind: Literal["http_search", "null"] = "http_search"
    base_url: str = "http://localhost:8081/search"
    api_key_env: str = "ADP_SEARCH_KEY"
    max_results: int = Field(default=6, ge=1, le=50)
    fetch_top_n: int = Field(default=3, ge=0, le=20)
    user_agent: str = "adp-retrieval/1.0"

    @property
    def api_key(self) -> str:
        return os.environ.get(self.api_key_env, "")

    @model_validator(mode="after")
    def _check_fetch(self) -> SearchConfig:
        if self.fetch_top_n > self.max_results:
            raise ValueError("search.fetch_top_n must not exceed search.max_results")
        return self


class RetrievalConfig(StrictModel):
    timeout_s: float = Field(default=8.0, gt=0.0, le=120.0)
    cache_ttl_s: int = Field(default=86400, ge=0)
    min_extract_chars: int = Field(default=240, ge=0)
    max_extract_chars: int = Field(default=12000, gt=0)

    @model_validator(mode="after")
    def _check_bounds(self) -> RetrievalConfig:
        if self.min_extract_chars >= self.max_extract_chars:
            raise ValueError("retrieval.min_extract_chars must be < retrieval.max_extract_chars")
        return self


class ThresholdConfig(StrictModel):
    defensiveness_max: float = Field(default=0.30, ge=0.0, le=1.0)
    engagement_min: float = Field(default=0.70, ge=0.0, le=1.0)
    semantic_low: float = Field(default=0.70, ge=0.0, le=1.0)
    semantic_high: float = Field(default=0.85, ge=0.0, le=1.0)
    lexicon_saturation: float = Field(default=2.0, gt=0.0)
    classifier_confidence_min: float = Field(default=0.55, ge=0.0, le=1.0)
    continuity_min: float = Field(default=0.25, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_semantic(self) -> ThresholdConfig:
        if self.semantic_low >= self.semantic_high:
            raise ValueError("thresholds.semantic_low must be < thresholds.semantic_high")
        return self


class DefensivenessWeights(StrictModel):
    lexical: float = Field(default=0.45, ge=0.0, le=1.0)
    semantic: float = Field(default=0.35, ge=0.0, le=1.0)
    structural: float = Field(default=0.20, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _sum_to_one(self) -> DefensivenessWeights:
        total = self.lexical + self.semantic + self.structural
        if abs(total - 1.0) > 1e-9:
            raise ValueError("weights.defensiveness components must sum to 1.0")
        return self


class EngagementWeights(StrictModel):
    answer_first: float = Field(default=0.30, ge=0.0, le=1.0)
    continuity: float = Field(default=0.20, ge=0.0, le=1.0)
    scenario: float = Field(default=0.25, ge=0.0, le=1.0)
    mechanism: float = Field(default=0.25, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _sum_to_one(self) -> EngagementWeights:
        total = self.answer_first + self.continuity + self.scenario + self.mechanism
        if abs(total - 1.0) > 1e-9:
            raise ValueError("weights.engagement components must sum to 1.0")
        return self


class WeightConfig(StrictModel):
    defensiveness: DefensivenessWeights = DefensivenessWeights()
    engagement: EngagementWeights = EngagementWeights()


class RegenerationConfig(StrictModel):
    max_attempts: int = Field(default=3, ge=1, le=10)
    temperatures: tuple[float, ...] = (0.7, 0.4, 0.2)
    fallback_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    initial_temperature: float = Field(default=0.7, ge=0.0, le=2.0)

    @field_validator("temperatures", mode="before")
    @classmethod
    def _coerce(cls, value: Any) -> Any:
        if isinstance(value, list):
            return tuple(float(item) for item in value)
        return value

    @model_validator(mode="after")
    def _check_len(self) -> RegenerationConfig:
        if len(self.temperatures) < self.max_attempts:
            raise ValueError(
                "regeneration.temperatures must provide at least max_attempts entries"
            )
        for temp in self.temperatures:
            if temp < 0.0 or temp > 2.0:
                raise ValueError("regeneration.temperatures entries must be within [0, 2]")
        return self


class LatencyConfig(StrictModel):
    classify_budget_s: float = Field(default=6.0, gt=0.0)
    generate_budget_s: float = Field(default=60.0, gt=0.0)
    total_budget_s: float = Field(default=120.0, gt=0.0)

    @model_validator(mode="after")
    def _check_total(self) -> LatencyConfig:
        if self.total_budget_s < self.generate_budget_s:
            raise ValueError("latency.total_budget_s must be >= latency.generate_budget_s")
        return self


class PathConfig(StrictModel):
    lexicon_dir: Path = Path("config/lexicon")
    seeds_dir: Path = Path("config/seeds")
    templates_dir: Path = Path("config/templates")
    memory_db: Path = Path("var/adp_memory.sqlite")
    cache_db: Path = Path("var/adp_cache.sqlite")
    audit_dir: Path = Path("var/audit")


class MemoryConfig(StrictModel):
    extract_facts: bool = True
    max_context_turns: int = Field(default=12, ge=0, le=200)
    max_facts_in_context: int = Field(default=40, ge=0, le=500)


class AuditConfig(StrictModel):
    enabled: bool = True
    tracing: bool = False


class SemanticConfig(StrictModel):
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    batch_size: int = Field(default=32, ge=1, le=512)
    char_ngram_size: int = Field(default=4, ge=2, le=8)


class AppConfig(StrictModel):
    provider: ProviderConfig = ProviderConfig()
    search: SearchConfig = SearchConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    thresholds: ThresholdConfig = ThresholdConfig()
    weights: WeightConfig = WeightConfig()
    regeneration: RegenerationConfig = RegenerationConfig()
    latency: LatencyConfig = LatencyConfig()
    languages: tuple[str, ...] = ("en", "hu")
    paths: PathConfig = PathConfig()
    memory: MemoryConfig = MemoryConfig()
    audit: AuditConfig = AuditConfig()
    semantic: SemanticConfig = SemanticConfig()

    @field_validator("languages", mode="before")
    @classmethod
    def _coerce_languages(cls, value: Any) -> Any:
        if isinstance(value, list):
            return tuple(str(item).lower() for item in value)
        return value

    @model_validator(mode="after")
    def _check_languages(self) -> AppConfig:
        if "en" not in self.languages or "hu" not in self.languages:
            raise ValueError("languages must include at least 'en' and 'hu'")
        for lang in self.languages:
            lexicon = self.paths.lexicon_dir / f"{lang}.yaml"
            seeds = self.paths.seeds_dir / f"defensive_seeds.{lang}.txt"
            if not lexicon.exists():
                raise ValueError(f"missing lexicon file: {lexicon}")
            if not seeds.exists():
                raise ValueError(f"missing seed file: {seeds}")
        for name in ("mechanism.md.tmpl", "scenario.md.tmpl", "evidential.md.tmpl"):
            template = self.paths.templates_dir / name
            if not template.exists():
                raise ValueError(f"missing template file: {template}")
        return self

    def temperature_for_attempt(self, attempt: int) -> float:
        if attempt <= 0:
            return self.regeneration.initial_temperature
        index = min(attempt - 1, len(self.regeneration.temperatures) - 1)
        return self.regeneration.temperatures[index]

    def ensure_directories(self) -> None:
        for path in (self.paths.memory_db, self.paths.cache_db):
            path.parent.mkdir(parents=True, exist_ok=True)
        self.paths.audit_dir.mkdir(parents=True, exist_ok=True)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"configuration root must be a mapping: {path}")
    return raw


def load_config(path: str | Path | None = None) -> AppConfig:
    if path is None:
        env_path = os.environ.get("ADP_CONFIG")
        resolved = Path(env_path) if env_path else DEFAULT_CONFIG_PATH
    else:
        resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"configuration file not found: {resolved}")
    data = _read_yaml(resolved)
    config = AppConfig.model_validate(data)
    config.ensure_directories()
    return config
