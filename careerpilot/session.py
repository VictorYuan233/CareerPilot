"""Serializable workflow state and bounded runtime accounting."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from .schemas import (
    ApplicationPackage,
    CandidateEvidence,
    Id,
    JobRecord,
    MatchResult,
    ReviewResult,
    UserGoal,
)


class PlanStep(BaseModel):
    """One planned operation and the condition that makes it necessary."""

    action: str
    condition: str | None = None


class ApprovalRecord(BaseModel):
    """An explicit human decision, with package approvals tied to a revision."""

    action: Literal["job_selection", "package"]
    job_id: Id
    package_version: int | None = None
    feedback_ids: list[Id] = Field(default_factory=list)
    valid: bool = True
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TraceEvent(BaseModel):
    """Operational labels only; source text and exception messages stay out of logs."""

    agent: str
    tool: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    duration_ms: float = 0
    model_calls: int = 0
    model_usage: dict[str, int] = Field(default_factory=dict)
    outcome: str
    error_code: str | None = None


class SessionState(BaseModel):
    """A complete checkpoint that can resume with newly supplied agent instances."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    goal: UserGoal | None = None
    plan: list[PlanStep] = Field(default_factory=list)
    status: Literal[
        "running", "awaiting_selection", "awaiting_clarification",
        "awaiting_approval", "completed", "rejected", "blocked", "exhausted",
    ] = "running"
    phase: Literal["discover", "match", "prepare", "review", "deliver"] = "discover"
    evidence: list[CandidateEvidence] = Field(default_factory=list)
    jobs: list[JobRecord] = Field(default_factory=list)
    matches: list[MatchResult] = Field(default_factory=list)
    selected_job_id: str | None = None
    package: ApplicationPackage | None = None
    review: ReviewResult | None = None
    reviewed_package: ApplicationPackage | None = None
    reviewed_evidence: list[CandidateEvidence] | None = None
    reviewed_job: JobRecord | None = None
    approved_version: int | None = None
    approvals: list[ApprovalRecord] = Field(default_factory=list)
    approved_feedback_ids: list[Id] = Field(default_factory=list)
    pending_questions: list[str] = Field(default_factory=list)
    repair_count: int = Field(default=0, ge=0, le=2)
    steps_used: int = Field(default=0, ge=0)
    model_calls_used: int = Field(default=0, ge=0)
    max_steps: int = Field(default=12, ge=1)
    max_model_calls: int = Field(default=0, ge=0)
    max_repairs: int = Field(default=2, ge=0, le=2)
    error_code: str | None = None
    trace: list[TraceEvent] = Field(default_factory=list)


class BudgetExceeded(RuntimeError):
    """Raised before an operation would exceed its session allowance."""


class BudgetRuntime:
    """Agents reserve each model call immediately before making it."""

    def __init__(self, state: SessionState):
        self.state = state

    @property
    def approved_feedback_ids(self) -> tuple[str, ...]:
        """Expose an immutable snapshot of feedback explicitly approved for reuse."""
        return tuple(self.state.approved_feedback_ids)

    def reserve_model_call(self) -> None:
        if self.state.model_calls_used >= self.state.max_model_calls:
            raise BudgetExceeded("model_budget_exhausted")
        self.state.model_calls_used += 1

    def reserve_step(self) -> None:
        if self.state.steps_used >= self.state.max_steps:
            raise BudgetExceeded("step_budget_exhausted")
        self.state.steps_used += 1


def save_session(state: SessionState, path: str | Path) -> Path:
    """Atomically replace a JSON checkpoint after validating its current state."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    validated = SessionState.model_validate(state.model_dump(mode="json"))
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(validated.model_dump_json(indent=2), encoding="utf-8")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def load_session(path: str | Path) -> SessionState:
    """Read a trusted local checkpoint; schema versions are explicitly validated."""
    return SessionState.model_validate_json(Path(path).read_text(encoding="utf-8"))
