"""Configuration.

One YAML file is the source of truth; environment variables override it. No
module anywhere else in the codebase reads a literal tunable - they all take a
config object (or a slice of one) so tests can vary behaviour without patching.

Precedence, highest first:
    1. explicit kwargs passed to ``load_config``
    2. environment variables  (ASKMYDOCS_CHUNKING__CHUNK_TOKENS=512)
    3. .env file
    4. the YAML file
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from .errors import ConfigError

#: Repo root, resolved from this file's location (src/askmydocs/config.py).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_FILE = PROJECT_ROOT / "config" / "default.yaml"


class PathsConfig(BaseModel):
    raw_pdfs: Path
    processed: Path
    indexes: Path
    chunks_file: Path
    manifest_file: Path

    def resolved(self, root: Path = PROJECT_ROOT) -> PathsConfig:
        """Return a copy with every relative path anchored to ``root``.

        Keeps the YAML readable (relative paths) while guaranteeing the code
        never depends on the process working directory.
        """
        return PathsConfig(
            **{
                name: (root / value if not value.is_absolute() else value)
                for name, value in self.model_dump().items()
            }
        )


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: Literal["console", "json"] = "console"

    @field_validator("level")
    @classmethod
    def _upper(cls, v: str) -> str:
        level = v.upper()
        if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"unknown log level: {v}")
        return level


class IngestionConfig(BaseModel):
    scanned_page_char_threshold: int = Field(50, ge=0)
    scanned_doc_page_ratio: float = Field(0.8, ge=0.0, le=1.0)
    furniture_page_ratio: float = Field(0.6, ge=0.0, le=1.0)
    furniture_margin_ratio: float = Field(0.12, ge=0.0, le=0.5)
    furniture_max_chars: int = Field(120, gt=0)
    extract_tables: bool = True
    max_file_mb: int = Field(100, gt=0)


class StructureConfig(BaseModel):
    heading_size_ratio: float = Field(1.15, ge=1.0)
    heading_max_words: int = Field(12, gt=0)
    bold_qualifies_as_heading: bool = True
    numbered_heading_patterns: list[str] = Field(default_factory=list)
    min_headings_per_page: float = Field(0.33, ge=0.0)
    max_heading_depth: int = Field(4, gt=0)

    @field_validator("numbered_heading_patterns")
    @classmethod
    def _compilable(cls, patterns: list[str]) -> list[str]:
        import re

        for pattern in patterns:
            try:
                re.compile(pattern)
            except re.error as exc:  # pragma: no cover - config authoring error
                raise ValueError(f"invalid heading regex {pattern!r}: {exc}") from exc
        return patterns


class ChunkingConfig(BaseModel):
    chunk_tokens: int = Field(450, gt=0)
    chunk_overlap_tokens: int = Field(60, ge=0)
    min_section_tokens: int = Field(80, ge=0)
    keep_tables_atomic: bool = True
    prepend_breadcrumb: bool = True
    separators: list[str] = Field(default_factory=lambda: ["\n\n", "\n", ". ", " "])
    tokenizer_model: str = "BAAI/bge-base-en-v1.5"

    @model_validator(mode="after")
    def _overlap_fits(self) -> ChunkingConfig:
        if self.chunk_overlap_tokens >= self.chunk_tokens:
            raise ValueError("chunk_overlap_tokens must be smaller than chunk_tokens")
        return self


class EmbeddingConfig(BaseModel):
    model: str = "BAAI/bge-base-en-v1.5"
    batch_size: int = Field(32, gt=0)
    normalize: bool = True
    query_prefix: str = ""
    device: str = "auto"


class RetrievalConfig(BaseModel):
    vector_top_n: int = Field(30, gt=0)
    bm25_top_n: int = Field(30, gt=0)
    rrf_k: int = Field(60, gt=0)
    # Relative influence of each retriever in the fusion. Equal by default;
    # exposed so Phase 8 can tune it against the golden set.
    vector_weight: float = Field(1.0, ge=0.0)
    bm25_weight: float = Field(1.0, ge=0.0)
    bm25_k1: float = Field(1.5, gt=0.0)
    bm25_b: float = Field(0.75, ge=0.0, le=1.0)
    rerank_enabled: bool = True
    rerank_top_k: int = Field(6, gt=0)
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_batch_size: int = Field(32, gt=0)
    rerank_max_length: int = Field(512, gt=0)
    #: ms-marco cross-encoders emit raw logits (roughly -11..+11), not
    #: probabilities. None keeps the top_k regardless of absolute score.
    min_rerank_score: float | None = None

    @model_validator(mode="after")
    def _at_least_one_retriever(self) -> RetrievalConfig:
        if self.vector_weight == 0.0 and self.bm25_weight == 0.0:
            raise ValueError("at least one of vector_weight / bm25_weight must be non-zero")
        return self


class GenerationConfig(BaseModel):
    model: str = "llama-3.3-70b-versatile"
    temperature: float = Field(0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(1024, gt=0)
    request_timeout_s: int = Field(60, gt=0)
    max_retries: int = Field(4, ge=0)
    retry_base_delay_s: float = Field(1.0, gt=0.0)
    retry_max_delay_s: float = Field(20.0, gt=0.0)
    #: Ceiling on the context block. Lowest-ranked sources are dropped first.
    max_context_tokens: int = Field(6000, gt=0)
    refusal_text: str = "I don't have enough information to answer that."


class VerificationConfig(BaseModel):
    enabled: bool = True
    entailment_mode: Literal["off", "heuristic", "llm"] = "off"
    flag_uncited_claims: bool = True
    #: Sentences shorter than this are fragments, not claims worth citing.
    #: Three, not four: "Rollbacks are automatic." is a real assertion, and the
    #: colon/question/meta filters already remove most short non-claims.
    min_claim_words: int = Field(3, gt=0)
    #: Identifiers shorter than this are too common to be evidence.
    min_identifier_length: int = Field(4, gt=0)
    #: Ceiling on claims sent to the LLM judge in one batch, to bound tokens.
    entailment_max_claims: int = Field(12, gt=0)


class EvaluationConfig(BaseModel):
    golden_set: Path = Path("eval/golden/golden_set.jsonl")
    baseline_file: Path = Path("eval/baselines/metrics_baseline.json")
    runs_dir: Path = Path("eval/runs")
    k_values: list[int] = Field(default_factory=lambda: [1, 3, 5, 10])
    #: RAGAS adds LLM-judged faithfulness and answer relevance on top of the
    #: custom metrics. Off by default: it costs API calls per item and its
    #: scores are not reproducible, which makes it a poor CI gate.
    use_ragas: bool = False
    ragas_metrics: list[str] = Field(
        default_factory=lambda: ["faithfulness", "answer_relevancy"]
    )
    #: Metrics the CI build may fail on. Each entry is a dotted path into the
    #: evaluation report plus how far it may move in the bad direction.
    gates: list[dict[str, Any]] = Field(default_factory=list)


class AppConfig(BaseSettings):
    """Root configuration object. Passed down; never re-read from disk deeper in."""

    model_config = SettingsConfigDict(
        env_prefix="ASKMYDOCS_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        yaml_file=DEFAULT_CONFIG_FILE,
    )

    paths: PathsConfig
    logging: LoggingConfig = LoggingConfig()
    ingestion: IngestionConfig = IngestionConfig()
    structure: StructureConfig = StructureConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    generation: GenerationConfig = GenerationConfig()
    verification: VerificationConfig = VerificationConfig()
    evaluation: EvaluationConfig = EvaluationConfig()

    # Secret, never in YAML. Accepts the plain GROQ_API_KEY name people expect.
    groq_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("GROQ_API_KEY", "ASKMYDOCS_GROQ_API_KEY"),
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Order == precedence, highest first.
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
        )


def load_config(config_file: str | Path | None = None, **overrides: Any) -> AppConfig:
    """Build an :class:`AppConfig`.

    ``config_file`` wins over the ``ASKMYDOCS_CONFIG_FILE`` env var, which wins
    over ``config/default.yaml``. Paths in the result are already resolved to
    absolute, so callers never worry about the working directory.
    """
    path = config_file or os.getenv("ASKMYDOCS_CONFIG_FILE") or DEFAULT_CONFIG_FILE
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")

    # pydantic-settings reads yaml_file off the settings class, so a per-call
    # file means a per-call subclass. Cheap, and keeps AppConfig immutable.
    class _ScopedConfig(AppConfig):
        model_config = SettingsConfigDict(
            **{**AppConfig.model_config, "yaml_file": path}
        )

    try:
        config = _ScopedConfig(**overrides)
    except Exception as exc:
        raise ConfigError(f"invalid configuration in {path}: {exc}") from exc

    # Anchor relative paths against the repo, not the process working directory.
    config.paths = config.paths.resolved(PROJECT_ROOT)
    return config


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Process-wide config for long-lived callers (API, Streamlit).

    Batch jobs and tests should call :func:`load_config` directly so they can
    point at a different file without fighting the cache.
    """
    return load_config()
