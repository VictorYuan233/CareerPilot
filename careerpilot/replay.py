"""Offline adapters for normalized job and candidate records.

The course notebook uses the default catalog. Callers may supply their own
records or replace each adapter through the same specialist protocols.
"""

from __future__ import annotations

import re

from .catalog import load_evidence, load_jobs
from .matching import ReplayDiscovery, evaluate_requirement
from .schemas import (
    ApplicationPackage, CandidateEvidence, JobRecord, ReviewResult, Runtime,
    UnsupportedRevision, UserGoal,
)

__all__ = ["ReplayDiscovery", "ReplayPreparation", "ReplayVerification", "load_evidence", "load_jobs"]


class ReplayPreparation:
    """Create evidence-linked drafts from arbitrary normalized job records."""

    def prepare(
        self,
        goal: UserGoal,
        job: JobRecord,
        evidence: list[CandidateEvidence],
        previous: ApplicationPackage | None,
        review: ReviewResult | None,
        *,
        runtime: Runtime,
    ) -> ApplicationPackage:
        del runtime
        public_evidence = [item for item in evidence if item.confirmed and not item.sensitive]
        public_evidence = _rank_evidence(public_evidence, goal, job)
        resume_evidence = list(public_evidence)
        cover_evidence = list(public_evidence)

        if previous is not None:
            evidence_by_id = {item.evidence_id: item for item in evidence}
            resume_evidence = _previous_public_evidence(
                previous.resume_draft,
                f"# Resume draft for {_normalise_text(job.title)}",
                {"No confirmed, non-sensitive profile evidence is available yet."},
                evidence_by_id,
            )
            cover_evidence = _previous_public_evidence(
                previous.cover_letter,
                f"# Cover letter draft for {_normalise_text(job.title)} at {_normalise_text(job.company)}",
                {
                    f"Selected role: {_normalise_text(job.title)} at {_normalise_text(job.company)}.",
                    "Source-backed candidate details follow.",
                    "No confirmed, non-sensitive profile evidence is available yet.",
                },
                evidence_by_id,
            )
            instructions = " ".join(review.repair_instructions).strip() if review else ""
            resume_evidence, cover_evidence = _apply_revision(
                instructions, resume_evidence, cover_evidence, public_evidence,
            )

        resume_draft = _render_resume(job, resume_evidence)
        cover_letter = _render_cover_letter(job, cover_evidence)
        ats_answers, ats_refs = _build_ats_answers(job, evidence)
        claim_evidence = _build_claim_evidence(resume_evidence, cover_evidence, ats_refs)
        if previous is not None and (
            resume_draft == previous.resume_draft
            and cover_letter == previous.cover_letter
            and ats_answers == previous.ats_answers
            and claim_evidence == previous.claim_evidence
        ):
            raise UnsupportedRevision("The requested revision would not change the current draft.")

        return ApplicationPackage(
            job_id=job.job_id,
            resume_draft=resume_draft,
            cover_letter=cover_letter,
            ats_answers=ats_answers,
            claim_evidence=claim_evidence,
            version=(previous.version + 1) if previous else 1,
        )


class ReplayVerification:
    """Check the bounded draft templates, factual bullets and ATS answers."""

    def review(
        self,
        package: ApplicationPackage,
        job: JobRecord,
        evidence: list[CandidateEvidence],
        *,
        runtime: Runtime,
    ) -> ReviewResult:
        del runtime
        if package.job_id != job.job_id:
            return ReviewResult(
                decision="blocked",
                error_codes=["package_job_mismatch"],
                evidence_refs=[],
                repair_instructions=["Rebuild the package for the selected job."],
            )

        evidence_by_id = {item.evidence_id: item for item in evidence}
        body_bullets = _verify_draft_bodies(package, job, evidence_by_id)
        if body_bullets is None:
            return _revise_result("unverified_draft_content", "Remove content that does not match the source-linked replay templates.")

        expected_map: dict[str, set[str]] = {}
        for claim, references in body_bullets.items():
            expected_map[claim] = references

        requirements = {
            requirement.ats_question: requirement
            for requirement in job.requirements
            if requirement.ats_question
        }
        expected_answers: dict[str, str | None] = {}
        unresolved: list[str] = []
        failed_requirements: list[str] = []
        requirement_refs: set[str] = set()
        for requirement in job.requirements:
            outcome, supporting = evaluate_requirement(requirement, evidence)
            requirement_refs.update(item.evidence_id for item in supporting)
            if outcome == "unknown":
                unresolved.append(requirement.ats_question or requirement.label)
            elif outcome == "ineligible":
                failed_requirements.append(requirement.label)
            if requirement.ats_question is None:
                continue
            question = requirement.ats_question
            answer = _answer_for_requirement(requirement, outcome, supporting)
            expected_answers[question] = answer
            if outcome != "unknown" and answer is None:
                unresolved.append(question + " has source evidence that cannot be safely answered in the form.")
            elif answer is not None:
                expected_map[f"ATS: {question}"] = {item.evidence_id for item in supporting}

        for question, answer in package.ats_answers.items():
            if question not in requirements or answer != expected_answers.get(question):
                return _revise_result(
                    "unsupported_ats_answer",
                    "Use the answer supported by the source fact for each job question.",
                    requirement_refs,
                )
        missing_questions = set(requirements) - set(package.ats_answers)
        if missing_questions:
            return _revise_result(
                "missing_ats_answer_field",
                "Include an answer field for each question named by the job requirements.",
                requirement_refs,
            )

        if package.claim_evidence != {
            claim: sorted(refs) for claim, refs in expected_map.items()
        }:
            return _revise_result(
                "claim_evidence_mismatch",
                "Make each claim reference exactly the source records used in the draft or ATS answer.",
                requirement_refs,
            )

        if failed_requirements:
            return ReviewResult(
                decision="blocked",
                error_codes=["eligibility_requirement_not_met"],
                evidence_refs=sorted(requirement_refs),
                repair_instructions=["The candidate does not meet: " + ", ".join(failed_requirements)],
            )
        if not job.requirements_complete:
            unresolved.append("The job requirements have not been completely normalized.")
        if unresolved:
            return ReviewResult(
                decision="human_required",
                error_codes=["job_or_candidate_facts_unresolved"],
                evidence_refs=sorted(requirement_refs),
                repair_instructions=["Resolve these job requirements before approval: " + "; ".join(unresolved)],
            )
        if not _confirmed_public_evidence(evidence):
            return ReviewResult(
                decision="human_required",
                error_codes=["no_public_candidate_evidence"],
                evidence_refs=[],
                repair_instructions=["Add at least one confirmed, non-sensitive candidate fact before approving the package."],
            )
        return ReviewResult(
            decision="pass",
            error_codes=[],
            evidence_refs=sorted({ref for refs in expected_map.values() for ref in refs}),
            repair_instructions=[],
        )


_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "is", "it", "of", "on", "or", "the", "this", "to", "with", "my", "me",
    "please", "focus", "letter", "cover", "resume", "cv", "remove", "shorten",
    "shorter", "unsupported", "claim", "claims", "an", "all", "some",
}
_BULLET_PATTERN = re.compile(r"^- (.+) \[([A-Za-z0-9][A-Za-z0-9_.:-]{0,95})\]$")


def _normalise_text(text: str) -> str:
    return " ".join(text.split())


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[^\W_]+(?:[+#]+)?", text.casefold()))


def _evidence_keywords(item: CandidateEvidence) -> set[str]:
    structured = " ".join(
        value for value in (item.fact_key or "", str(item.fact_value) if item.fact_value is not None else "")
        if value
    ).replace("_", " ").replace(".", " ")
    return _tokens(f"{item.text} {structured}")


def _rank_evidence(
    evidence: list[CandidateEvidence], goal: UserGoal, job: JobRecord,
) -> list[CandidateEvidence]:
    context = " ".join([job.title, *job.role_aliases, *(req.label for req in job.requirements)])
    if goal.target_roles:
        context += " " + " ".join(goal.target_roles)
    requested = _tokens(context) - _STOP_WORDS
    return sorted(
        evidence,
        key=lambda item: (-len(requested & _evidence_keywords(item)), item.evidence_id),
    )


def _matches_term(item: CandidateEvidence, term: str) -> bool:
    keywords = _tokens(term) - _STOP_WORDS
    return bool(keywords) and keywords <= _evidence_keywords(item)


def _scope(instructions: str) -> str:
    text = instructions.casefold()
    if "cover letter" in text or "covering letter" in text:
        return "cover"
    if re.search(r"\b(resume|résumé|cv)\b", text):
        return "resume"
    return "both"


def _apply_revision(
    instructions: str,
    resume_evidence: list[CandidateEvidence],
    cover_evidence: list[CandidateEvidence],
    all_evidence: list[CandidateEvidence],
) -> tuple[list[CandidateEvidence], list[CandidateEvidence]]:
    text = instructions.casefold().strip()
    scope = _scope(text)
    if not text:
        raise UnsupportedRevision("The revision instruction is empty.")

    if "unsupported claim" in text or "unsupported claims" in text:
        # Rebuild from the source records rather than carrying forward unknown text.
        return list(all_evidence), list(all_evidence)

    focus_match = re.search(r"\bfocus\b.*?\b(?:on|around)\s+(.+)$", text)
    if focus_match:
        term = focus_match.group(1).strip(" .!?:;")
        existing = list({item.evidence_id: item for item in [*resume_evidence, *cover_evidence]}.values())
        selected = [item for item in existing if _matches_term(item, term)]
        if not selected:
            raise UnsupportedRevision(f"No confirmed public evidence matches '{term}'.")
        return (
            selected if scope in {"resume", "both"} else resume_evidence,
            selected if scope in {"cover", "both"} else cover_evidence,
        )

    remove_match = re.search(r"\bremove\s+(?:the\s+|all\s+)?(.+)$", text)
    if remove_match:
        term = remove_match.group(1).strip(" .!?:;")
        return (
            [item for item in resume_evidence if not _matches_term(item, term)]
            if scope in {"resume", "both"} else resume_evidence,
            [item for item in cover_evidence if not _matches_term(item, term)]
            if scope in {"cover", "both"} else cover_evidence,
        )

    if "short" in text or "condense" in text:
        def shorten(items: list[CandidateEvidence]) -> list[CandidateEvidence]:
            if len(items) < 2:
                return items
            return items[: max(1, len(items) - 1)]

        return (
            shorten(resume_evidence) if scope in {"resume", "both"} else resume_evidence,
            shorten(cover_evidence) if scope in {"cover", "both"} else cover_evidence,
        )

    raise UnsupportedRevision("This replay adapter cannot carry out that revision instruction.")


def _render_bullets(evidence: list[CandidateEvidence]) -> list[str]:
    return [f"- {_normalise_text(item.text)} [{item.evidence_id}]" for item in evidence]


def _render_resume(job: JobRecord, evidence: list[CandidateEvidence]) -> str:
    lines = [f"# Resume draft for {_normalise_text(job.title)}", ""]
    if evidence:
        lines.extend(_render_bullets(evidence))
    else:
        lines.append("No confirmed, non-sensitive profile evidence is available yet.")
    return "\n".join(lines)


def _render_cover_letter(job: JobRecord, evidence: list[CandidateEvidence]) -> str:
    lines = [
        f"# Cover letter draft for {_normalise_text(job.title)} at {_normalise_text(job.company)}",
        "",
        f"Selected role: {_normalise_text(job.title)} at {_normalise_text(job.company)}.",
        "Source-backed candidate details follow.",
    ]
    if evidence:
        lines.extend(_render_bullets(evidence))
    else:
        lines.append("No confirmed, non-sensitive profile evidence is available yet.")
    return "\n".join(lines)


def _confirmed_public_evidence(evidence: list[CandidateEvidence]) -> list[CandidateEvidence]:
    return [item for item in evidence if item.confirmed and not item.sensitive]


def _answer_for_requirement(
    requirement, outcome: str, supporting: list[CandidateEvidence],
) -> str | None:
    if outcome == "unknown" or not supporting:
        return None
    values = {item.value for item in supporting}
    if len(values) != 1:
        return None
    value = next(iter(values))
    if any(item.sensitive for item in supporting):
        if requirement.operator == "eq" and type(requirement.expected) is bool and type(value) is bool:
            return "Yes" if value else "No"
        return None
    if type(value) is bool:
        return "Yes" if value else "No"
    if isinstance(value, (str, int, float)):
        return str(value)
    return None


def _build_ats_answers(
    job: JobRecord, evidence: list[CandidateEvidence],
) -> tuple[dict[str, str | None], dict[str, list[str]]]:
    answers: dict[str, str | None] = {}
    references: dict[str, list[str]] = {}
    for requirement in job.requirements:
        if requirement.ats_question is None:
            continue
        outcome, supporting = evaluate_requirement(requirement, evidence)
        answer = _answer_for_requirement(requirement, outcome, supporting)
        answers[requirement.ats_question] = answer
        if answer is not None:
            references[requirement.ats_question] = sorted(item.evidence_id for item in supporting)
    return answers, references


def _build_claim_evidence(
    resume_evidence: list[CandidateEvidence],
    cover_evidence: list[CandidateEvidence],
    ats_references: dict[str, list[str]],
) -> dict[str, list[str]]:
    claims: dict[str, set[str]] = {}
    for item in [*resume_evidence, *cover_evidence]:
        claims.setdefault(_normalise_text(item.text), set()).add(item.evidence_id)
    result = {claim: sorted(refs) for claim, refs in claims.items()}
    result.update({f"ATS: {question}": refs for question, refs in ats_references.items()})
    return result


def _parse_body(
    body: str,
    heading: str,
    allowed_static: set[str],
    evidence_by_id: dict[str, CandidateEvidence],
) -> dict[str, set[str]] | None:
    lines = body.splitlines()
    if not lines or lines[0] != heading:
        return None
    bullets: dict[str, set[str]] = {}
    for line in lines[1:]:
        if not line or line in allowed_static:
            continue
        match = _BULLET_PATTERN.fullmatch(line)
        if not match:
            return None
        claim, evidence_id = match.groups()
        item = evidence_by_id.get(evidence_id)
        if (
            item is None
            or not item.confirmed
            or item.sensitive
            or _normalise_text(item.text) != claim
        ):
            return None
        bullets.setdefault(claim, set()).add(evidence_id)
    return bullets


def _previous_public_evidence(
    body: str,
    heading: str,
    allowed_static: set[str],
    evidence_by_id: dict[str, CandidateEvidence],
) -> list[CandidateEvidence]:
    lines = body.splitlines()
    if not lines or lines[0] != heading:
        return []
    selected: list[CandidateEvidence] = []
    seen: set[str] = set()
    for line in lines[1:]:
        if not line or line in allowed_static:
            continue
        match = _BULLET_PATTERN.fullmatch(line)
        if not match:
            continue
        claim, evidence_id = match.groups()
        item = evidence_by_id.get(evidence_id)
        if (
            item is not None
            and item.confirmed
            and not item.sensitive
            and _normalise_text(item.text) == claim
            and evidence_id not in seen
        ):
            selected.append(item)
            seen.add(evidence_id)
    return selected


def _verify_draft_bodies(
    package: ApplicationPackage,
    job: JobRecord,
    evidence_by_id: dict[str, CandidateEvidence],
) -> dict[str, set[str]] | None:
    resume = _parse_body(
        package.resume_draft,
        f"# Resume draft for {_normalise_text(job.title)}",
        {"No confirmed, non-sensitive profile evidence is available yet."},
        evidence_by_id,
    )
    cover = _parse_body(
        package.cover_letter,
        f"# Cover letter draft for {_normalise_text(job.title)} at {_normalise_text(job.company)}",
        {
            f"Selected role: {_normalise_text(job.title)} at {_normalise_text(job.company)}.",
            "Source-backed candidate details follow.",
            "No confirmed, non-sensitive profile evidence is available yet.",
        },
        evidence_by_id,
    )
    if resume is None or cover is None:
        return None
    combined = {claim: set(refs) for claim, refs in resume.items()}
    for claim, refs in cover.items():
        combined.setdefault(claim, set()).update(refs)
    return combined


def _revise_result(
    error_code: str,
    instructions: str,
    evidence_refs: set[str] | None = None,
) -> ReviewResult:
    return ReviewResult(
        decision="revise",
        error_codes=[error_code],
        evidence_refs=sorted(evidence_refs or set()),
        repair_instructions=[instructions],
    )
