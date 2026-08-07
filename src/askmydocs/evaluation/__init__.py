"""Evaluation: does the system actually work, and did this change break it?"""

from .generation_metrics import refusal_breakdown, score_answer
from .golden import GoldenItem, GoldenSet, load_golden_set, write_golden_set
from .harness import run_evaluation
from .retrieval_metrics import (
    aggregate,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    score_ranking,
)

__all__ = [
    "GoldenItem",
    "GoldenSet",
    "aggregate",
    "load_golden_set",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "refusal_breakdown",
    "run_evaluation",
    "score_answer",
    "score_ranking",
    "write_golden_set",
]
