"""Deterministic routing and human checkpoints around replaceable agents."""

from __future__ import annotations

from time import perf_counter
from typing import Callable, TypeVar

from pydantic import TypeAdapter

from .config import RuntimeConfig
from .intent import UnsupportedIntentError
from .schemas import (
    ApplicationPackage, CandidateEvidence, DiscoveryAgent, Id, IntentClassifier,
    JobRecord, MatchResult, PreparationAgent, ReviewResult, UnsupportedRevision, UserGoal, VerificationAgent,
)
from .session import ApprovalRecord, BudgetExceeded, BudgetRuntime, PlanStep, SessionState, TraceEvent

T = TypeVar("T")


class _Stopped(Exception):
    """Internal signal after the session has recorded a safe terminal state."""


class Orchestrator:
    """Run until a human decision or terminal outcome, preserving a resumable state."""

    def __init__(
        self, discovery: DiscoveryAgent, preparation: PreparationAgent,
        verification: VerificationAgent, classifier: IntentClassifier | None = None, *,
        config: RuntimeConfig | None = None, max_steps=None, max_model_calls=None, max_repairs=None,
    ):
        # Validate limits before a session can make any calls.
        settings = (config or RuntimeConfig()).model_dump()
        overrides = {"max_steps": max_steps, "max_model_calls": max_model_calls, "max_repairs": max_repairs}
        settings.update({key: value for key, value in overrides.items() if value is not None})
        limits = RuntimeConfig.model_validate(settings)
        if classifier is None:
            from .intent import AdkClassifier, ReplayClassifier

            classifier = AdkClassifier(limits) if limits.mode == "live" else ReplayClassifier()
        self.discovery = discovery
        self.preparation = preparation
        self.verification = verification
        self.classifier = classifier
        self.max_steps = limits.max_steps
        self.max_model_calls = limits.max_model_calls
        self.max_repairs = limits.max_repairs

    def start(
        self, text: str, evidence: list[CandidateEvidence], jobs: list[JobRecord] | None = None,
        package: ApplicationPackage | None = None, *, approved_feedback_ids: list[str] | None = None,
    ) -> SessionState:
        """Parse a goal and stop at its first human checkpoint or terminal state."""
        state = SessionState(
            evidence=evidence,
            jobs=jobs or [],
            package=package,
            max_steps=self.max_steps,
            max_model_calls=self.max_model_calls,
            max_repairs=self.max_repairs,
            approved_feedback_ids=list(dict.fromkeys(
                TypeAdapter(list[Id]).validate_python(approved_feedback_ids or [])
            )),
        ).model_copy(deep=True)
        try:
            evidence_ids = [item.evidence_id for item in state.evidence]
            if len(evidence_ids) != len(set(evidence_ids)):
                self._stop(state, "duplicate_evidence_ids")
            state.goal = self._call(state, "orchestrator", "classify", lambda: self._classify(text, state))
            if state.goal.intent not in {"search", "match", "prepare", "revise"}:
                self._stop(state, "unsupported_intent")
            if state.goal.intent == "revise" and state.package is None:
                self._stop(state, "missing_previous_package")
            if state.package is not None and state.goal.intent != "revise":
                self._stop(state, "unexpected_previous_package")
            if state.package is not None:
                if not any(job.job_id == state.package.job_id for job in state.jobs):
                    self._stop(state, "package_job_missing")
                state.selected_job_id = state.package.job_id
            if state.goal.intent == "match" and not state.jobs:
                self._stop(state, "missing_known_job")
            if state.goal.intent == "search":
                # Search the requested market even when a caller supplies earlier results.
                state.jobs = []
            if state.goal.intent == "prepare" and len(state.jobs) == 1:
                state.selected_job_id = state.jobs[0].job_id
            state.plan = self._build_plan(state)
            if state.goal.intent == "revise":
                state.review = ReviewResult(decision="revise", repair_instructions=[text])
                self._schedule_repair(state)
            state.phase = "match" if state.jobs else "discover"
        except _Stopped:
            return state
        return self.advance(state)

    def advance(self, state: SessionState) -> SessionState:
        """Resume running states, leaving human checkpoints untouched."""
        runtime = BudgetRuntime(state)
        try:
            while state.status == "running":
                if state.goal is None:
                    self._stop(state, "missing_goal")
                if state.phase == "discover":
                    state.jobs = self._call(
                        state, "discovery", "discover",
                        lambda: TypeAdapter(list[JobRecord]).validate_python(
                            self.discovery.discover(state.goal, state.evidence, runtime=runtime)
                        ),
                    )
                    state.phase = "match"
                elif state.phase == "match":
                    self._match(state, runtime)
                elif state.phase == "prepare":
                    self._prepare(state, runtime)
                elif state.phase == "review":
                    self._review(state, runtime)
                else:
                    self._stop(state, "invalid_resume_phase")
        except _Stopped:
            pass
        return state

    def select_job(self, state: SessionState, job_id: str) -> SessionState:
        self._require_status(state, "awaiting_selection")
        match = next((item for item in state.matches if item.job_id == job_id), None)
        if match is None or match.eligibility == "ineligible":
            raise ValueError("Select an eligible or unresolved job from the current shortlist.")
        state.selected_job_id = job_id
        state.approvals.append(ApprovalRecord(action="job_selection", job_id=job_id))
        self._human_trace(state, "select_job")
        state.status = "running"
        try:
            self._route_selected(state, match)
        except _Stopped:
            return state
        return self.advance(state)

    def clarify(self, state: SessionState, evidence: list[CandidateEvidence]) -> SessionState:
        self._require_status(state, "awaiting_clarification")
        updates = TypeAdapter(list[CandidateEvidence]).validate_python(evidence)
        if not updates or any(not item.confirmed for item in updates):
            raise ValueError("Clarification requires confirmed evidence.")
        if len({item.evidence_id for item in updates}) != len(updates):
            raise ValueError("Each clarification evidence ID must be unique.")
        merged = {item.evidence_id: item for item in state.evidence}
        merged.update({item.evidence_id: item.model_copy(deep=True) for item in updates})
        state.evidence = list(merged.values())
        state.pending_questions = []
        self._invalidate_approval(state)
        state.phase = "match"
        state.status = "running"
        self._human_trace(state, "clarify")
        return self.advance(state)

    def approve(self, state: SessionState, version: int, *, feedback_ids=None) -> SessionState:
        self._require_status(state, "awaiting_approval")
        consented_ids = TypeAdapter(list[Id]).validate_python(feedback_ids or [])
        try:
            job = self._selected_job(state)
        except _Stopped:
            return state
        if state.package is None or state.package.version != version:
            raise ValueError("Approval must identify the current package version.")
        error_code = None
        if state.review is None or state.review.decision != "pass" or state.package != state.reviewed_package:
            error_code = "package_changed_after_review"
        elif state.reviewed_evidence is None or state.reviewed_job is None:
            error_code = "review_context_missing"
        elif (
            job != state.reviewed_job
            or sorted(state.evidence, key=lambda item: item.evidence_id) != state.reviewed_evidence
        ):
            error_code = "review_context_changed"
        if error_code:
            self._invalidate_approval(state)
            state.status = "blocked"
            state.error_code = error_code
            state.trace.append(TraceEvent(
                agent="orchestrator", tool="terminate", outcome="blocked", error_code=state.error_code,
            ))
            return state
        state.approved_version = version
        state.approved_feedback_ids = list(dict.fromkeys(state.approved_feedback_ids + consented_ids))
        state.approvals.append(ApprovalRecord(
            action="package", job_id=state.package.job_id, package_version=version,
            feedback_ids=list(dict.fromkeys(consented_ids)),
        ))
        state.status = "completed"
        state.phase = "deliver"
        self._human_trace(state, "approve")
        return state

    def reject(self, state: SessionState) -> SessionState:
        if state.status not in {"awaiting_selection", "awaiting_clarification", "awaiting_approval"}:
            raise ValueError("Rejection requires a pending human checkpoint.")
        self._invalidate_approval(state)
        state.status = "rejected"
        state.pending_questions = []
        self._human_trace(state, "reject")
        return state

    def revise(self, state: SessionState, instructions: str) -> SessionState:
        if state.status not in {"awaiting_approval", "completed"} or state.package is None:
            raise ValueError("Revision requires a reviewed application package.")
        if state.review is None or state.review.decision != "pass":
            raise ValueError("Revision requires a completed verification review.")
        if not instructions.strip():
            raise ValueError("Revision instructions must contain text.")
        self._invalidate_approval(state)
        state.review = ReviewResult(decision="revise", repair_instructions=[instructions])
        self._human_trace(state, "revise")
        try:
            self._schedule_repair(state)
        except _Stopped:
            return state
        return self.advance(state)

    def _classify(self, text: str, state: SessionState) -> UserGoal:
        usage = getattr(self.classifier, "last_usage", None)
        if isinstance(usage, dict):
            usage.clear()
        if getattr(self.classifier, "mode", "replay") == "live":
            BudgetRuntime(state).reserve_model_call()
        return UserGoal.model_validate(self.classifier.classify(text, known_job=bool(state.jobs)))

    def _match(self, state: SessionState, runtime: BudgetRuntime) -> None:
        if not state.jobs:
            if state.goal.intent == "search":
                state.status = "completed"
                state.phase = "deliver"
                return
            self._stop(state, "no_jobs_found")
        job_ids = [job.job_id for job in state.jobs]
        if len(job_ids) != len(set(job_ids)):
            self._stop(state, "duplicate_job_ids")
        state.matches = self._call(
            state, "discovery", "match",
            lambda: TypeAdapter(list[MatchResult]).validate_python(
                self.discovery.match(state.goal, state.jobs, state.evidence, runtime=runtime)
            ),
        )
        matched_ids = [item.job_id for item in state.matches]
        if len(matched_ids) != len(set(matched_ids)) or set(matched_ids) != set(job_ids):
            self._stop(state, "invalid_match_job_ids")
        valid_evidence = {item.evidence_id for item in state.evidence if item.confirmed}
        if any(set(item.evidence_ids) - valid_evidence for item in state.matches):
            self._stop(state, "invalid_match_evidence")
        if state.selected_job_id is None and state.goal.intent in {"search", "prepare"}:
            # Discovery supplies ranked decisions. Apply the display/selection cap last.
            state.matches = state.matches[:state.goal.limit]
        if state.goal.intent in {"search", "match"}:
            state.status = "completed"
            state.phase = "deliver"
            return
        if state.selected_job_id is None:
            if not any(item.eligibility != "ineligible" for item in state.matches):
                self._stop(state, "no_eligible_jobs")
            state.status = "awaiting_selection"
            return
        selected = next((item for item in state.matches if item.job_id == state.selected_job_id), None)
        if selected is None:
            self._stop(state, "selected_job_missing")
        self._route_selected(state, selected)

    def _route_selected(self, state: SessionState, match: MatchResult) -> None:
        self._selected_job(state)
        if match.eligibility == "ineligible":
            self._stop(state, "ineligible_job")
        if match.eligibility == "unknown":
            state.status = "awaiting_clarification"
            state.pending_questions = match.gaps or ["Confirm the missing eligibility facts."]
            return
        state.phase = "prepare"

    def _prepare(self, state: SessionState, runtime: BudgetRuntime) -> None:
        job = self._selected_job(state)
        previous = state.package.model_copy(deep=True) if state.package else None
        self._invalidate_approval(state)
        package = self._call(
            state, "preparation", "prepare",
            lambda: ApplicationPackage.model_validate(
                self.preparation.prepare(
                    state.goal, job, state.evidence, previous, state.review, runtime=runtime
                )
            ),
        )
        if package.job_id != job.job_id:
            self._stop(state, "package_job_mismatch")
        if previous is not None and package.version <= previous.version:
            self._stop(state, "package_version_not_advanced")
        state.package = package
        state.phase = "review"

    def _review(self, state: SessionState, runtime: BudgetRuntime) -> None:
        job = self._selected_job(state)
        if state.package is None:
            self._stop(state, "missing_package")
        reviewed_package = state.package.model_copy(deep=True)
        reviewed_job = job.model_copy(deep=True)
        reviewed_evidence = [
            item.model_copy(deep=True)
            for item in sorted(state.evidence, key=lambda item: item.evidence_id)
        ]
        state.review = self._call(
            state, "verification", "review",
            lambda: ReviewResult.model_validate(
                self.verification.review(state.package, job, state.evidence, runtime=runtime)
            ),
        )
        if state.review.decision == "pass":
            state.reviewed_package = reviewed_package
            state.reviewed_job = reviewed_job
            state.reviewed_evidence = reviewed_evidence
            state.status = "awaiting_approval"
        elif state.review.decision == "human_required":
            state.status = "awaiting_clarification"
            state.pending_questions = state.review.repair_instructions or ["Confirm the unresolved facts."]
        elif state.review.decision == "revise":
            self._schedule_repair(state)
        else:
            self._stop(state, "verification_blocked")

    def _schedule_repair(self, state: SessionState) -> None:
        if state.repair_count >= state.max_repairs:
            self._stop(state, "repair_limit_exhausted", exhausted=True)
        state.repair_count += 1
        state.phase = "prepare"
        state.status = "running"

    def _selected_job(self, state: SessionState) -> JobRecord:
        job = next((item for item in state.jobs if item.job_id == state.selected_job_id), None)
        if job is None:
            self._stop(state, "selected_job_missing")
        if job.status != "open":
            self._stop(state, "job_unavailable")
        return job

    def _call(self, state: SessionState, agent: str, tool: str, operation: Callable[[], T]) -> T:
        started = perf_counter()
        before = state.model_calls_used
        outcome = "success"
        code = None
        try:
            BudgetRuntime(state).reserve_step()
            return operation()
        except BudgetExceeded:
            outcome, code = "exhausted", "budget_exhausted"
        except UnsupportedIntentError:
            outcome, code = "blocked", "unsupported_intent"
        except UnsupportedRevision:
            outcome, code = "blocked", "unsupported_revision"
        except Exception:
            outcome, code = "blocked", "agent_call_failed"
        finally:
            usage = {}
            if tool == "classify" and state.model_calls_used > before:
                reported = getattr(self.classifier, "last_usage", {})
                if isinstance(reported, dict):
                    usage = {
                        key: value for key, value in reported.items()
                        if key in {"prompt_token_count", "candidates_token_count", "total_token_count"}
                        and type(value) is int and value >= 0
                    }
            state.trace.append(TraceEvent(
                agent=agent, tool=tool, duration_ms=(perf_counter() - started) * 1000,
                model_calls=state.model_calls_used - before, outcome=outcome, error_code=code,
                model_usage=usage,
            ))
        # Record the failed operation before publishing the terminal decision.
        self._stop(state, code, exhausted=outcome == "exhausted")

    @staticmethod
    def _invalidate_approval(state: SessionState) -> None:
        state.approved_version = None
        state.reviewed_package = None
        state.reviewed_evidence = None
        state.reviewed_job = None
        for approval in state.approvals:
            if approval.action == "package":
                approval.valid = False

    @staticmethod
    def _build_plan(state: SessionState) -> list[PlanStep]:
        """Describe the chosen route once, including bounded conditional branches."""
        plan = [PlanStep(action="classify")]
        if not state.jobs:
            plan.append(PlanStep(action="discover"))
        plan.append(PlanStep(action="match"))
        if state.goal.intent in {"search", "match"}:
            return plan + [PlanStep(action="deliver_shortlist")]
        if state.selected_job_id is None:
            plan.append(PlanStep(action="select_job"))
        plan.extend([
            PlanStep(action="clarify", condition="Eligibility or verification requires confirmed evidence."),
            PlanStep(action="prepare"),
            PlanStep(action="review"),
        ])
        if state.max_repairs:
            plan.append(PlanStep(
                action="repair", condition=f"Review requests revision, at most {state.max_repairs} attempts.",
            ))
        return plan + [PlanStep(action="approve"), PlanStep(action="deliver_package")]

    @staticmethod
    def _require_status(state: SessionState, expected: str) -> None:
        if state.status != expected:
            raise ValueError(f"This action requires session status {expected}.")

    @staticmethod
    def _human_trace(state: SessionState, action: str) -> None:
        state.trace.append(TraceEvent(agent="human", tool=action, outcome="accepted"))

    @staticmethod
    def _stop(state: SessionState, code: str, *, exhausted=False) -> None:
        state.status = "exhausted" if exhausted else "blocked"
        state.error_code = code
        state.approved_version = None
        state.trace.append(TraceEvent(
            agent="orchestrator", tool="terminate", outcome=state.status, error_code=code,
        ))
        raise _Stopped()
