"""Triggering an evaluation run.

The harness itself arrives in Phase 8. The endpoint and its job plumbing exist
now so the contract is fixed and the front end can be built against it; the
runner reports plainly that the harness is not implemented yet rather than
pretending to succeed.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, status

from ...logging_setup import get_logger
from ..deps import StateDep
from ..schemas import EvalRequest, JobResponse

log = get_logger(__name__)

router = APIRouter(tags=["evaluation"])


@router.post("/eval", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
def trigger_eval(
    payload: EvalRequest, state: StateDep, background: BackgroundTasks
) -> JobResponse:
    """Start an evaluation run against the golden set."""
    if state.pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=state.index_error or "no index loaded - nothing to evaluate",
        )

    job = state.jobs.create(
        kind="eval", detail=payload.golden_set or str(state.config.evaluation.golden_set)
    )
    # Not exclusive: evaluation only reads, so it can run alongside queries.
    background.add_task(
        state.jobs.run, job, lambda: _run_eval(state, payload), exclusive=False
    )
    return JobResponse(**job.as_dict())


def _run_eval(state: StateDep, payload: EvalRequest) -> dict[str, Any]:
    try:
        from ...evaluation.harness import run_evaluation
    except ImportError as exc:  # pragma: no cover - until Phase 8 lands
        raise NotImplementedError(
            "the evaluation harness is not implemented yet (Phase 8)"
        ) from exc

    return run_evaluation(
        state.config,
        golden_set=payload.golden_set,
        limit=payload.limit,
        pipeline=state.pipeline,
        answerer=state.answerer,
        verifier=state.verifier,
    )
