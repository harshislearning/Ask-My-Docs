"""A minimal in-process job registry for slow operations.

Ingestion and evaluation take minutes. Running them inside a request means the
connection is held open past every sensible proxy timeout, and the caller has no
way to see progress or failure - so they run in the background and the caller
polls a job id.

In-process and non-durable on purpose: this is an internal single-instance
service, and a job queue with a broker would be more infrastructure than the
problem deserves. The consequences are stated rather than hidden - jobs are lost
on restart, and running two workers would give each its own registry.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from ..logging_setup import bind_run, clear_run, get_logger

log = get_logger(__name__)

#: Completed jobs kept for inspection before the oldest are discarded.
MAX_JOBS = 50


class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class Job:
    def __init__(self, kind: str, detail: str | None = None) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind
        self.status = JobStatus.PENDING
        self.created_at = datetime.now(UTC)
        self.started_at: datetime | None = None
        self.finished_at: datetime | None = None
        self.detail = detail
        self.error: str | None = None
        self.result: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "detail": self.detail,
            "error": self.error,
            "result": self.result,
        }


class JobRegistry:
    """Tracks background jobs and serialises the ones that must not overlap."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        #: Ingestion rewrites the files queries read from, so only one such job
        #: may run at a time.
        self._exclusive = threading.Lock()

    def create(self, kind: str, detail: str | None = None) -> Job:
        job = Job(kind, detail)
        with self._lock:
            self._jobs[job.id] = job
            self._evict_old()
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def is_busy(self) -> bool:
        return self._exclusive.locked()

    def run(self, job: Job, work: Callable[[], dict[str, Any]], *, exclusive: bool = True) -> None:
        """Execute ``work``, recording the outcome on ``job``.

        Called from a background thread. Never raises: a crashed job must show
        up as a failed job, not as a silent hang.
        """
        lock = self._exclusive if exclusive else _NULL_LOCK
        bind_run(job_id=job.id, job_kind=job.kind)
        try:
            with lock:
                job.status = JobStatus.RUNNING
                job.started_at = datetime.now(UTC)
                log.info("job_started", job_id=job.id, kind=job.kind)

                job.result = work()
                job.status = JobStatus.SUCCEEDED
                log.info("job_succeeded", job_id=job.id, kind=job.kind, result=job.result)
        except Exception as exc:
            job.status = JobStatus.FAILED
            job.error = f"{type(exc).__name__}: {exc}"
            log.exception("job_failed", job_id=job.id, kind=job.kind)
        finally:
            job.finished_at = datetime.now(UTC)
            clear_run()

    def _evict_old(self) -> None:
        if len(self._jobs) <= MAX_JOBS:
            return
        finished = [j for j in self._jobs.values() if j.finished_at is not None]
        for job in sorted(finished, key=lambda j: j.created_at)[: len(self._jobs) - MAX_JOBS]:
            self._jobs.pop(job.id, None)


class _NullLock:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: object) -> None:
        return None


_NULL_LOCK = _NullLock()
