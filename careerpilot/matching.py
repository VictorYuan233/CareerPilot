"""Data-driven, deterministic matching for structured job and evidence snapshots."""

from __future__ import annotations

import math
import re

from .schemas import CandidateEvidence, Eligibility, JobRecord, JobRequirement, MatchResult, Runtime, UserGoal


_QUERY_WORDS = {
    "a", "an", "and", "for", "in", "of", "or", "the", "to", "with",
    "job", "opportunity", "position", "role",
}


def _tokens(text: str) -> set[str]:
    """Normalize punctuation and ordinary plurals without inferring role synonyms."""
    words = re.findall(r"[^\W_]+(?:[+#]+)?", text.casefold())
    normalized: set[str] = set()
    for word in words:
        if len(word) > 4 and word.endswith("ies"):
            word = word[:-3] + "y"
        elif len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        if word == "intern":
            word = "internship"
        normalized.add(word)
    return normalized


def _typed_value(value):
    """Compare explicit facts without treating a boolean as a numeric value."""
    if type(value) is bool:
        return "boolean", value
    if type(value) is int or (type(value) is float and math.isfinite(value)):
        return "number", value
    if isinstance(value, str) and value.strip():
        return "text", " ".join(value.casefold().split())
    return None


def evaluate_requirement(
    requirement: JobRequirement, evidence: list[CandidateEvidence],
) -> tuple[Eligibility, list[CandidateEvidence]]:
    """Evaluate one fact key. Missing, conflicting or incompatible values remain unknown.

    Supporting records include confirmed conflicting facts so a caller can explain
    why clarification is needed. Sensitive values stay inside deterministic checks.
    """
    supporting = [
        item for item in evidence
        if item.fact_key == requirement.fact_key and item.confirmed and item.value is not None
    ]
    if not supporting:
        return "unknown", []
    values = [_typed_value(item.value) for item in supporting]
    expected = _typed_value(requirement.expected)
    if expected is None or any(value is None for value in values) or len(set(values)) != 1:
        return "unknown", supporting
    actual = values[0]
    if actual[0] != expected[0]:
        return "unknown", supporting
    if requirement.operator == "eq":
        satisfied = actual[1] == expected[1]
    elif actual[0] != "number":
        return "unknown", supporting
    elif requirement.operator == "gte":
        satisfied = actual[1] >= expected[1]
    else:
        satisfied = actual[1] <= expected[1]
    return ("eligible" if satisfied else "ineligible"), supporting


def _role_relevance(job: JobRecord, goal: UserGoal) -> float | None:
    requested = [_tokens(role) - _QUERY_WORDS for role in goal.target_roles]
    requested = [words for words in requested if words]
    if not requested:
        return 0.0
    role_options = [_tokens(text) - _QUERY_WORDS for text in [job.title, *job.role_aliases]]
    relevance = [
        len(target) / len(option)
        for target in requested
        for option in role_options
        if option and target <= option
    ]
    return max(relevance) if relevance else None


def _preference_relevance(job: JobRecord, goal: UserGoal) -> float:
    requested = _tokens(" ".join(goal.preferences)) - _QUERY_WORDS
    if not requested:
        return 0.0
    available = _tokens(" ".join([job.title, job.description, *[req.label for req in job.requirements]]))
    return len(requested & available) / len(requested)


class ReplayDiscovery:
    """Read a supplied catalog and rank every relevant role before shortlist capping."""

    def __init__(self, jobs: list[JobRecord] | None = None):
        if jobs is None:
            from .catalog import load_jobs

            jobs = load_jobs()
        self._jobs = [job.model_copy(deep=True) for job in jobs]

    def discover(
        self, goal: UserGoal, evidence: list[CandidateEvidence], *, runtime: Runtime,
    ) -> list[JobRecord]:
        del evidence, runtime
        candidates = [
            job for job in self._jobs
            if job.status == "open"
            and (not goal.location or _tokens(goal.location) <= _tokens(job.location))
            and _role_relevance(job, goal) is not None
        ]
        return [job.model_copy(deep=True) for job in candidates]

    def match(
        self, goal: UserGoal, jobs: list[JobRecord], evidence: list[CandidateEvidence],
        *, runtime: Runtime,
    ) -> list[MatchResult]:
        del runtime
        results: list[MatchResult] = []
        for job in jobs:
            statuses: list[Eligibility] = []
            matched: list[str] = []
            gaps: list[str] = []
            questions: list[str] = []
            supporting_ids: list[str] = []
            for requirement in job.requirements:
                eligibility, supporting = evaluate_requirement(requirement, evidence)
                statuses.append(eligibility)
                supporting_ids.extend(item.evidence_id for item in supporting)
                if eligibility == "eligible":
                    matched.append(requirement.label)
                elif eligibility == "ineligible":
                    gaps.append(f"The confirmed fact does not meet {requirement.label}.")
                else:
                    questions.append(requirement.ats_question or f"Please confirm: {requirement.label}.")
                if (
                    goal.max_required_experience_years is not None
                    and requirement.category == "experience_years"
                    and requirement.operator in {"eq", "gte"}
                    and type(requirement.expected) in {int, float}
                    and requirement.expected > goal.max_required_experience_years
                ):
                    statuses.append("ineligible")
                    gaps.append(
                        f"The role requires {requirement.expected:g} years, exceeding the requested "
                        f"maximum of {goal.max_required_experience_years:g}."
                    )
            if job.status != "open":
                statuses.append("ineligible")
                gaps.append("The listing is not open.")
            if not job.requirements_complete:
                statuses.append("unknown")
                gaps.append("The listing requirements have not been completely structured.")
            if goal.unresolved_hard_constraints:
                statuses.append("unknown")
                gaps.extend(
                    f"Unresolved hard constraint: {constraint}"
                    for constraint in goal.unresolved_hard_constraints
                )
            if "ineligible" in statuses:
                eligibility = "ineligible"
            elif "unknown" in statuses:
                eligibility = "unknown"
                gaps.extend(questions)
            else:
                eligibility = "eligible"
            relevance = _role_relevance(job, goal) or 0.0
            preference = _preference_relevance(job, goal)
            coverage = len(matched) / len(job.requirements) if job.requirements else 0.0
            score = {"eligible": 70, "unknown": 35, "ineligible": 0}[eligibility]
            score += 15 * relevance + 10 * preference + 5 * coverage
            results.append(MatchResult(
                job_id=job.job_id,
                eligibility=eligibility,
                matched_requirements=matched,
                gaps=gaps,
                evidence_ids=list(dict.fromkeys(supporting_ids)),
                score=round(score, 3),
                reasons=[
                    f"Structured requirement assessment is {eligibility}.",
                    f"Ranking uses role relevance {relevance:.3f}, preference coverage "
                    f"{preference:.3f} and supported requirement coverage {coverage:.3f}.",
                ],
            ))
        return sorted(results, key=lambda result: result.score, reverse=True)
