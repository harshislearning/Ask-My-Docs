"""Inspecting background jobs."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from ..deps import StateDep
from ..schemas import JobResponse

router = APIRouter(tags=["jobs"])


@router.get("/jobs", response_model=list[JobResponse])
def list_jobs(state: StateDep) -> list[JobResponse]:
    return [JobResponse(**job.as_dict()) for job in state.jobs.list()]


@router.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str, state: StateDep) -> JobResponse:
    job = state.jobs.get(job_id)
    if job is None:
        # Jobs are held in memory, so an id from before a restart is gone
        # rather than merely unknown. Say so.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"no job {job_id} - job history does not survive a restart",
        )
    return JobResponse(**job.as_dict())
