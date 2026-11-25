"""Version-one contracts shared by the orchestrator and the three specialist agents.

IDs link records within a session. Evidence provenance belongs to the record,
while references in matches, drafts and reviews only contain the stable IDs.
"""

from datetime import datetime
from typing import Annotated, Literal, Protocol

from pydantic import (
    BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt, StrictStr,
    StringConstraints, model_validator,
)


Id = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,95}$")]
Intent = Literal["search", "match", "prepare", "revise"]
Eligibility = Literal["eligible", "ineligible", "unknown"]
Decision = Literal["pass", "revise", "human_required", "blocked"]


class UnsupportedRevision(ValueError):
    """The preparation adapter could not apply the requested change."""


class Contract(BaseModel):
    """Fail at the interface when a producer uses an unknown field."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, allow_inf_nan=False)


class UserGoal(Contract):
    """Parsed user intent and selection criteria. A preference is never a veto."""

    intent: Intent
    target_roles: list[str] = Field(default_factory=list)
    location: str | None = None
    hard_constraints: list[str] = Field(default_factory=list)
    unresolved_hard_constraints: list[str] = Field(default_factory=list)
    preferences: list[str] = Field(default_factory=list)
    max_required_experience_years: float | None = Field(default=None, ge=0)
    limit: int = Field(default=3, ge=1, le=10)


class CandidateEvidence(Contract):
    """One source-backed statement, including whether it may enter draft text."""

    evidence_id: Id
    source_id: Id
    text: str = Field(min_length=1)
    provenance: str = Field(min_length=1)
    confirmed: bool
    sensitive: bool
    # Confirmation describes the source; the value records a yes/no answer.
    boolean_value: bool | None = None
    fact_key: Id | None = None
    fact_value: StrictStr | StrictInt | StrictFloat | None = None

    @model_validator(mode="after")
    def single_value(self):
        if self.boolean_value is not None and self.fact_value is not None:
            raise ValueError("An evidence record describes one value.")
        return self

    @property
    def value(self) -> bool | str | int | float | None:
        return self.boolean_value if self.boolean_value is not None else self.fact_value


class JobRequirement(Contract):
    """A normalized condition, joined to evidence by a source-defined fact key."""

    fact_key: Id
    label: str = Field(min_length=1)
    operator: Literal["eq", "gte", "lte"] = "eq"
    expected: StrictBool | StrictStr | StrictInt | StrictFloat = True
    category: Literal["qualification", "experience_years"] = "qualification"
    ats_question: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def valid_threshold(self):
        numeric = type(self.expected) in (int, float)
        if self.operator in {"gte", "lte"} and not numeric:
            raise ValueError("A threshold comparison requires a numeric expected value.")
        if self.category == "experience_years" and (not numeric or self.expected < 0):
            raise ValueError("Required years of experience must be non-negative.")
        return self


class JobRecord(Contract):
    """A read-only snapshot. Recheck status before preparing a package."""

    job_id: Id
    title: str = Field(min_length=1)
    company: str = Field(min_length=1)
    location: str
    source_url: str
    description: str
    fetched_at: datetime
    status: Literal["open", "expired", "unavailable"]
    requirements: list[JobRequirement] = Field(default_factory=list)
    requirements_complete: bool = False
    role_aliases: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def distinct_questions(self):
        questions = [req.ats_question for req in self.requirements if req.ats_question is not None]
        if len(set(questions)) != len(questions):
            raise ValueError("Each ATS question must map to one requirement.")
        return self


class MatchResult(Contract):
    """All result IDs refer to the session's job and evidence records."""

    job_id: Id
    eligibility: Eligibility
    matched_requirements: list[str]
    gaps: list[str]
    evidence_ids: list[Id]
    score: float = Field(ge=0, le=100, allow_inf_nan=False)
    reasons: list[str]


class ApplicationPackage(Contract):
    """A draft revision. Keys in claim_evidence are claims in its materials."""

    job_id: Id
    resume_draft: str = Field(min_length=1)
    cover_letter: str = Field(min_length=1)
    ats_answers: dict[str, str | None]
    claim_evidence: dict[str, list[Id]]
    version: int = Field(ge=1)


class ReviewResult(Contract):
    """The verifier controls review decisions; the orchestrator enforces gates."""

    decision: Decision
    error_codes: list[Id] = Field(default_factory=list)
    evidence_refs: list[Id] = Field(default_factory=list)
    repair_instructions: list[str] = Field(default_factory=list)


class Runtime(Protocol):
    """Reserve before each model request, including retries. Exhaustion stops work."""

    @property
    def approved_feedback_ids(self) -> tuple[str, ...]: ...

    def reserve_model_call(self) -> None: ...


class IntentClassifier(Protocol):
    mode: str

    def classify(self, text: str, *, known_job: bool = False) -> UserGoal: ...


class DiscoveryAgent(Protocol):
    def discover(
        self, goal: UserGoal, evidence: list[CandidateEvidence], *, runtime: Runtime
    ) -> list[JobRecord]: ...

    def match(
        self, goal: UserGoal, jobs: list[JobRecord], evidence: list[CandidateEvidence],
        *, runtime: Runtime
    ) -> list[MatchResult]: ...


class PreparationAgent(Protocol):
    def prepare(
        self, goal: UserGoal, job: JobRecord, evidence: list[CandidateEvidence],
        previous: ApplicationPackage | None, review: ReviewResult | None, *, runtime: Runtime
    ) -> ApplicationPackage: ...


class VerificationAgent(Protocol):
    def review(
        self, package: ApplicationPackage, job: JobRecord, evidence: list[CandidateEvidence],
        *, runtime: Runtime
    ) -> ReviewResult: ...
