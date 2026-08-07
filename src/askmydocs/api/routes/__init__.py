"""HTTP routes, one module per resource."""

from . import evaluation, health, ingest, jobs, query

__all__ = ["evaluation", "health", "ingest", "jobs", "query"]
