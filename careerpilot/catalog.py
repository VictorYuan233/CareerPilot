"""Load normalized jobs and candidate evidence from a caller's local JSON files."""

import json
from pathlib import Path

from pydantic import TypeAdapter

from .schemas import CandidateEvidence, JobRecord


_DATA = Path(__file__).resolve().parents[1] / "data"


def load_jobs(path: str | Path | None = None) -> list[JobRecord]:
    """Read a job collection; the bundled synthetic catalogue is the demo default."""
    source = Path(path) if path is not None else _DATA / "sample_jobs" / "jobs.json"
    rows = json.loads(source.read_text(encoding="utf-8"))["jobs"]
    return TypeAdapter(list[JobRecord]).validate_python(rows)


def load_evidence(path: str | Path | None = None) -> list[CandidateEvidence]:
    """Read a profile's source-backed evidence, preserving its IDs and fact keys."""
    source = Path(path) if path is not None else _DATA / "sample_profiles" / "synthetic_candidate.json"
    rows = json.loads(source.read_text(encoding="utf-8"))["evidence"]
    return TypeAdapter(list[CandidateEvidence]).validate_python(rows)
