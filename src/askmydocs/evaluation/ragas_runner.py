"""Optional RAGAS metrics.

RAGAS contributes two things the custom metrics do not: ``faithfulness``, an
LLM's judgement of whether the answer follows from the context, and
``answer_relevancy``, whether the answer actually addresses the question rather
than something adjacent.

It is off by default and isolated behind this module for three reasons. It costs
several API calls per item; its scores are not reproducible run to run, which
makes it a poor thing to gate a build on; and it defaults to OpenAI, so using it
here means wiring it to Groq and to the local bge embeddings. None of that
should be able to break the harness, so every failure below degrades to "no
RAGAS scores" and the run continues.
"""

from __future__ import annotations

from typing import Any

from ..config import AppConfig
from ..logging_setup import get_logger

log = get_logger(__name__)


def run_ragas(records: list[dict[str, Any]], config: AppConfig) -> dict[str, Any]:
    """Score generated answers with RAGAS.

    Returns the metrics, or a dict explaining why they are missing. Never raises.
    """
    if not config.evaluation.use_ragas:
        return {"enabled": False, "reason": "use_ragas is false"}

    samples = _to_samples(records)
    if not samples:
        return {"enabled": True, "reason": "no generated answers to score"}

    try:
        return _score(samples, config)
    except ImportError as exc:
        log.warning("ragas_unavailable", error=str(exc))
        return {
            "enabled": True,
            "reason": f'install the eval extra: uv pip install -e ".[eval]" ({exc})',
        }
    except Exception as exc:
        log.exception("ragas_failed")
        return {"enabled": True, "reason": f"{type(exc).__name__}: {exc}"}


def _to_samples(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reshape harness records into what RAGAS expects."""
    samples: list[dict[str, Any]] = []
    for record in records:
        answer = record.get("_answer")
        if answer is None or answer.refused:
            # A refusal has no claims to be faithful to; scoring it would
            # measure the refusal sentence, not the system.
            continue
        samples.append(
            {
                "user_input": record["question"],
                "response": answer.text,
                "retrieved_contexts": [source.text for source in answer.sources],
                "reference": record.get("expected_answer") or "",
            }
        )
    return samples


def _score(samples: list[dict[str, Any]], config: AppConfig) -> dict[str, Any]:
    from datasets import Dataset
    from langchain_groq import ChatGroq
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    from ..indexing.embedder import SentenceTransformerEmbedder

    metrics = _resolve_metrics(config.evaluation.ragas_metrics)
    if not metrics:
        return {"enabled": True, "reason": "no valid metric names configured"}

    llm = LangchainLLMWrapper(
        ChatGroq(
            model=config.generation.model,
            api_key=config.groq_api_key,
            temperature=0.0,
        )
    )
    embeddings = LangchainEmbeddingsWrapper(
        _LocalEmbeddings(SentenceTransformerEmbedder(config.embedding))
    )

    log.info("ragas_started", samples=len(samples), metrics=config.evaluation.ragas_metrics)
    result = evaluate(
        dataset=Dataset.from_list(samples), metrics=metrics, llm=llm, embeddings=embeddings
    )

    scores = {
        name: round(float(value), 4)
        for name, value in dict(result).items()
        if isinstance(value, int | float)
    }
    log.info("ragas_finished", **scores)
    return {"enabled": True, "samples": len(samples), "scores": scores}


def _resolve_metrics(names: list[str]) -> list[Any]:
    from ragas import metrics as ragas_metrics

    resolved = []
    for name in names:
        metric = getattr(ragas_metrics, name, None)
        if metric is None:
            log.warning("unknown_ragas_metric", metric=name)
            continue
        resolved.append(metric)
    return resolved


class _LocalEmbeddings:
    """LangChain embeddings interface over the bge model already loaded here.

    Keeps RAGAS from reaching for OpenAI embeddings, which would mean a second
    provider, a second key, and a different vector space from the one retrieval
    actually uses.
    """

    def __init__(self, embedder: Any) -> None:
        self._embedder = embedder

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._embedder.encode_documents(texts)]

    def embed_query(self, text: str) -> list[float]:
        return list(self._embedder.encode_query(text).tolist())
