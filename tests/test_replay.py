"""Generic replay preparation and verification checks."""

from datetime import datetime, timezone

import pytest

from careerpilot.replay import ReplayPreparation, ReplayVerification
from careerpilot.schemas import (
    ApplicationPackage,
    CandidateEvidence,
    JobRecord,
    JobRequirement,
    ReviewResult,
    UnsupportedRevision,
    UserGoal,
)


def evidence(
    evidence_id: str,
    text: str,
    *,
    fact_key: str | None = None,
    boolean_value: bool | None = None,
    fact_value: str | int | float | None = None,
    sensitive: bool = False,
    confirmed: bool = True,
) -> CandidateEvidence:
    return CandidateEvidence(
        evidence_id=evidence_id,
        source_id=f"SRC-{evidence_id}",
        text=text,
        provenance="Synthetic test source record.",
        confirmed=confirmed,
        sensitive=sensitive,
        fact_key=fact_key,
        boolean_value=boolean_value,
        fact_value=fact_value,
    )


def job(
    requirements: list[JobRequirement] | None = None,
    *,
    requirements_complete: bool = True,
    title: str = "Data Analyst",
) -> JobRecord:
    return JobRecord(
        job_id="TEST-JOB-001",
        title=title,
        company="Example Organisation",
        location="Remote",
        source_url="https://example.org/jobs/TEST-JOB-001",
        description="A role for a synthetic demonstration.",
        fetched_at=datetime(2026, 11, 25, tzinfo=timezone.utc),
        status="open",
        requirements=requirements or [],
        requirements_complete=requirements_complete,
        role_aliases=["Data analyst", "Analytics specialist"],
    )


def base_evidence() -> list[CandidateEvidence]:
    return [
        evidence(
            "EV-PYTHON",
            "Completed Python coursework in data analysis.",
            fact_key="skill.python",
            boolean_value=True,
        ),
        evidence(
            "EV-ML",
            "Built a machine learning project for forecasting.",
            fact_key="skill.machine_learning",
            boolean_value=True,
        ),
        evidence(
            "EV-SQL",
            "Created SQL reports for a database class.",
            fact_key="skill.sql",
            boolean_value=True,
        ),
    ]


def prepare(
    selected_job: JobRecord,
    candidate_evidence: list[CandidateEvidence],
    *,
    goal: UserGoal | None = None,
    previous: ApplicationPackage | None = None,
    instruction: str | None = None,
) -> ApplicationPackage:
    review = (
        ReviewResult(decision="revise", repair_instructions=[instruction])
        if instruction is not None else None
    )
    return ReplayPreparation().prepare(
        goal or UserGoal(intent="prepare"),
        selected_job,
        candidate_evidence,
        previous,
        review,
        runtime=None,
    )


def test_no_authorisation_question_is_required_for_a_generic_job():
    candidate_evidence = base_evidence()
    selected_job = job()
    package = prepare(selected_job, candidate_evidence)

    result = ReplayVerification().review(package, selected_job, candidate_evidence, runtime=None)

    assert package.ats_answers == {}
    assert result.decision == "pass"
    assert all(item.text in package.resume_draft for item in candidate_evidence)


def test_two_sensitive_boolean_ats_answers_have_distinct_evidence_bindings():
    candidate_evidence = [
        *base_evidence(),
        evidence(
            "EV-ELIGIBILITY-A",
            "Private eligibility response for question A.",
            fact_key="eligibility.region_a",
            boolean_value=True,
            sensitive=True,
        ),
        evidence(
            "EV-ELIGIBILITY-B",
            "Private eligibility response for question B.",
            fact_key="eligibility.region_b",
            boolean_value=True,
            sensitive=True,
        ),
    ]
    selected_job = job(
        [
            JobRequirement(
                fact_key="eligibility.region_a",
                label="Eligibility for region A",
                expected=True,
                ats_question="Are you eligible for region A?",
            ),
            JobRequirement(
                fact_key="eligibility.region_b",
                label="Eligibility for region B",
                expected=True,
                ats_question="Are you eligible for region B?",
            ),
        ]
    )

    package = prepare(selected_job, candidate_evidence)
    result = ReplayVerification().review(package, selected_job, candidate_evidence, runtime=None)

    assert package.ats_answers == {
        "Are you eligible for region A?": "Yes",
        "Are you eligible for region B?": "Yes",
    }
    assert package.claim_evidence["ATS: Are you eligible for region A?"] == ["EV-ELIGIBILITY-A"]
    assert package.claim_evidence["ATS: Are you eligible for region B?"] == ["EV-ELIGIBILITY-B"]
    assert "Private eligibility response" not in package.resume_draft
    assert "Private eligibility response" not in package.cover_letter
    assert "Yes" not in package.resume_draft
    assert "Yes" not in package.cover_letter
    assert result.decision == "pass"


def test_unanswered_requirement_stays_none_and_requests_human_review():
    candidate_evidence = base_evidence()
    selected_job = job([
        JobRequirement(
            fact_key="eligibility.region_a",
            label="Eligibility for region A",
            expected=True,
            ats_question="Are you eligible for region A?",
        )
    ])
    package = prepare(selected_job, candidate_evidence)

    result = ReplayVerification().review(package, selected_job, candidate_evidence, runtime=None)

    assert package.ats_answers == {"Are you eligible for region A?": None}
    assert "ATS: Are you eligible for region A?" not in package.claim_evidence
    assert result.decision == "human_required"


def test_no_public_candidate_evidence_never_gets_invented():
    private_fact = evidence(
        "EV-PRIVATE",
        "Private fact not suitable for general application materials.",
        fact_key="eligibility.region_a",
        boolean_value=True,
        sensitive=True,
    )
    selected_job = job([
        JobRequirement(
            fact_key="eligibility.region_a",
            label="Eligibility for region A",
            expected=True,
            ats_question="Are you eligible for region A?",
        )
    ])
    package = prepare(selected_job, [private_fact])

    result = ReplayVerification().review(package, selected_job, [private_fact], runtime=None)

    assert "Private fact" not in package.resume_draft
    assert "Private fact" not in package.cover_letter
    assert result.decision == "human_required"
    assert result.error_codes == ["no_public_candidate_evidence"]


def test_resume_and_cover_letter_sort_evidence_by_job_relevance():
    candidate_evidence = [
        evidence("EV-UNRELATED", "Organised a weekend hiking group."),
        evidence("EV-RELATED", "Built a Python data analysis dashboard."),
    ]

    package = prepare(job(), candidate_evidence)

    assert package.resume_draft.index("EV-RELATED") < package.resume_draft.index("EV-UNRELATED")
    assert package.cover_letter.index("EV-RELATED") < package.cover_letter.index("EV-UNRELATED")


def test_focus_request_changes_only_requested_section_and_rebuilds_claim_map():
    candidate_evidence = base_evidence()
    selected_job = job()
    original = prepare(selected_job, candidate_evidence)

    revised = prepare(
        selected_job,
        candidate_evidence,
        previous=original,
        instruction="Focus the cover letter on Python coursework.",
    )

    assert "EV-PYTHON" in revised.cover_letter
    assert "EV-ML" not in revised.cover_letter
    assert "EV-SQL" not in revised.cover_letter
    assert revised.resume_draft == original.resume_draft
    assert revised.claim_evidence["Completed Python coursework in data analysis."] == ["EV-PYTHON"]
    assert revised.claim_evidence["Built a machine learning project for forecasting."] == ["EV-ML"]


def test_shortening_a_section_removes_trailing_claims_and_keeps_sources_consistent():
    candidate_evidence = base_evidence()
    selected_job = job()
    original = prepare(selected_job, candidate_evidence)

    revised = prepare(
        selected_job,
        candidate_evidence,
        previous=original,
        instruction="Shorten the cover letter.",
    )
    result = ReplayVerification().review(revised, selected_job, candidate_evidence, runtime=None)

    assert revised.resume_draft == original.resume_draft
    assert "EV-PYTHON" in revised.cover_letter
    assert "EV-ML" in revised.cover_letter
    assert "EV-SQL" not in revised.cover_letter
    assert "Created SQL reports for a database class." in revised.claim_evidence
    assert result.decision == "pass"


def test_consecutive_revisions_preserve_prior_removals():
    candidate_evidence = base_evidence()
    selected_job = job()
    original = prepare(selected_job, candidate_evidence)
    without_sql = prepare(
        selected_job,
        candidate_evidence,
        previous=original,
        instruction="Remove SQL.",
    )
    focused = prepare(
        selected_job,
        candidate_evidence,
        previous=without_sql,
        instruction="Focus the cover letter on Python coursework.",
    )

    assert "EV-SQL" not in focused.resume_draft
    assert "EV-SQL" not in focused.cover_letter
    assert "EV-PYTHON" in focused.cover_letter
    assert "EV-ML" not in focused.cover_letter
    assert "Completed Python coursework in data analysis." in focused.claim_evidence
    assert "Created SQL reports for a database class." not in focused.claim_evidence


def test_unsupported_claim_repair_rebuilds_from_public_sources():
    candidate_evidence = base_evidence()
    selected_job = job()
    original = prepare(selected_job, candidate_evidence)
    tampered = original.model_copy(update={
        "resume_draft": original.resume_draft + "\n- Managed a team of twenty engineers. [EV-PYTHON]",
    })
    review = ReplayVerification().review(tampered, selected_job, candidate_evidence, runtime=None)
    assert review.decision == "revise"

    repaired = prepare(
        selected_job,
        candidate_evidence,
        previous=tampered,
        instruction="Remove the unsupported claim.",
    )
    repaired_review = ReplayVerification().review(repaired, selected_job, candidate_evidence, runtime=None)

    assert "Managed a team" not in repaired.resume_draft
    assert repaired_review.decision == "pass"


@pytest.mark.parametrize(
    ("field", "addition"),
    [
        ("resume_draft", "\n- Managed five teams. [EV-PYTHON]"),
        ("cover_letter", "\nI led a global migration programme."),
    ],
)
def test_unmapped_or_unsupported_factual_claim_cannot_pass(field, addition):
    candidate_evidence = base_evidence()
    selected_job = job()
    original = prepare(selected_job, candidate_evidence)
    package = original.model_copy(update={field: getattr(original, field) + addition})

    result = ReplayVerification().review(package, selected_job, candidate_evidence, runtime=None)

    assert result.decision == "revise"
    assert "unverified_draft_content" in result.error_codes


def test_borrowing_an_unrelated_evidence_id_cannot_support_a_claim():
    candidate_evidence = base_evidence()
    selected_job = job()
    original = prepare(selected_job, candidate_evidence)
    forged_text = "Managed a team of twenty engineers."
    forged = original.model_copy(update={
        "resume_draft": original.resume_draft + f"\n- {forged_text} [EV-PYTHON]",
        "claim_evidence": {
            **original.claim_evidence,
            forged_text: ["EV-PYTHON"],
        },
    })

    result = ReplayVerification().review(forged, selected_job, candidate_evidence, runtime=None)

    assert result.decision == "revise"
    assert "unverified_draft_content" in result.error_codes


def test_sensitive_fact_citation_in_resume_or_cover_letter_is_rejected():
    private_fact = evidence(
        "EV-PRIVATE",
        "Private eligibility response for region A.",
        fact_key="eligibility.region_a",
        boolean_value=True,
        sensitive=True,
    )
    selected_job = job([
        JobRequirement(
            fact_key="eligibility.region_a",
            label="Eligibility for region A",
            expected=True,
            ats_question="Are you eligible for region A?",
        )
    ])
    candidate_evidence = [*base_evidence(), private_fact]
    original = prepare(selected_job, candidate_evidence)
    leaked = original.model_copy(update={
        "cover_letter": original.cover_letter + "\n- Private eligibility response for region A. [EV-PRIVATE]",
        "claim_evidence": {
            **original.claim_evidence,
            "Private eligibility response for region A.": ["EV-PRIVATE"],
        },
    })

    result = ReplayVerification().review(leaked, selected_job, candidate_evidence, runtime=None)

    assert result.decision == "revise"
    assert "unverified_draft_content" in result.error_codes


def test_unavailable_revision_and_noop_raise_clear_exception():
    candidate_evidence = [base_evidence()[0]]
    selected_job = job()
    original = prepare(selected_job, candidate_evidence)

    with pytest.raises(UnsupportedRevision, match="cannot carry out"):
        prepare(selected_job, candidate_evidence, previous=original, instruction="Make it sound more enthusiastic.")
    with pytest.raises(UnsupportedRevision, match="would not change"):
        prepare(selected_job, candidate_evidence, previous=original, instruction="Shorten the resume.")
