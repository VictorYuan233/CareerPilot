"""Generic matching with independently labeled jobs across unrelated occupations."""

from datetime import datetime, timezone

import pytest

from careerpilot.matching import ReplayDiscovery, evaluate_requirement
from careerpilot.schemas import CandidateEvidence, JobRecord, JobRequirement, UserGoal
from careerpilot.session import BudgetRuntime, SessionState


def fact(key, value, *, evidence_id="record-73", confirmed=True, sensitive=False):
    return CandidateEvidence(
        evidence_id=evidence_id, source_id="source-28", text="A source-backed synthetic fact.",
        provenance="Invented test evidence", confirmed=confirmed, sensitive=sensitive,
        fact_key=key, boolean_value=value if type(value) is bool else None,
        fact_value=value if type(value) is not bool else None,
    )


def job(job_id="vacancy-407", *, title="Landscape Architect", location="Porto",
        requirements=None, complete=True, aliases=None, description="Structured test vacancy.", status="open"):
    return JobRecord(
        job_id=job_id, title=title, company="Synthetic Employer", location=location,
        source_url=f"https://example.org/jobs/{job_id}", description=description,
        fetched_at=datetime(2026, 11, 25, tzinfo=timezone.utc), status=status,
        requirements=requirements or [], requirements_complete=complete, role_aliases=aliases or [],
    )


def runtime():
    return BudgetRuntime(SessionState())


@pytest.mark.parametrize(
    ("operator", "expected", "actual", "eligibility"),
    [
        ("eq", True, True, "eligible"),
        ("eq", True, False, "ineligible"),
        ("eq", False, False, "eligible"),
        ("eq", "Level B", " level b ", "eligible"),
        ("eq", "Level B", "Level A", "ineligible"),
        ("gte", 3.75, 4, "eligible"),
        ("gte", 8, 7.5, "ineligible"),
        ("lte", 2.5, 2, "eligible"),
        ("lte", 2.5, 3, "ineligible"),
        ("eq", 1, True, "unknown"),
        ("eq", True, "yes", "unknown"),
        ("gte", 3, "4", "unknown"),
    ],
)
def test_requirement_uses_explicit_typed_values(operator, expected, actual, eligibility):
    requirement = JobRequirement(fact_key="credential.any", label="A declared condition", operator=operator, expected=expected)
    evidence = [fact("credential.any", actual)]
    result, supporting = evaluate_requirement(requirement, evidence)
    assert result == eligibility
    assert [item.evidence_id for item in supporting] == ["record-73"]


def test_missing_or_unconfirmed_facts_do_not_use_text_or_record_ids_as_facts():
    requirement = JobRequirement(fact_key="licence.regional", label="A regional licence")
    evidence = [
        fact("other.fact", True, evidence_id="licence.regional"),
        fact("licence.regional", True, evidence_id="unconfirmed", confirmed=False),
        fact("licence.regional", None, evidence_id="not-answered"),
    ]
    evidence[0].text = "The candidate has the requested regional licence."
    result, supporting = evaluate_requirement(requirement, evidence)
    assert result == "unknown"
    assert supporting == []


def test_conflicting_confirmed_values_remain_unknown_even_if_both_meet_threshold():
    requirement = JobRequirement(fact_key="experience.marine", label="Marine survey experience", operator="gte", expected=2)
    records = [fact("experience.marine", 3), fact("experience.marine", 4, evidence_id="record-98")]
    eligibility, supporting = evaluate_requirement(requirement, records)
    assert eligibility == "unknown"
    assert {item.evidence_id for item in supporting} == {"record-73", "record-98"}


def test_consistent_sources_combine_without_using_unconfirmed_conflicts():
    requirement = JobRequirement(fact_key="licence.any", label="A required licence")
    records = [
        fact("licence.any", True),
        fact("licence.any", True, evidence_id="record-98"),
        fact("licence.any", False, evidence_id="unconfirmed", confirmed=False),
    ]
    eligibility, supporting = evaluate_requirement(requirement, records)
    assert eligibility == "eligible"
    assert len(supporting) == 2


@pytest.mark.parametrize("complete,expected", [(False, "unknown"), (True, "eligible")])
def test_listing_declares_whether_its_empty_requirements_are_complete(complete, expected):
    result = ReplayDiscovery([]).match(UserGoal(intent="match"), [job(complete=complete)], [], runtime=runtime())[0]
    assert result.eligibility == expected


def test_explicit_failure_is_preserved_in_an_incompletely_structured_listing():
    listing = job(complete=False, requirements=[JobRequirement(fact_key="licence.any", label="Required licence")])
    result = ReplayDiscovery([]).match(UserGoal(intent="match"), [listing], [fact("licence.any", False)], runtime=runtime())[0]
    assert result.eligibility == "ineligible"


@pytest.mark.parametrize("licence_value,expected", [(True, "unknown"), (False, "ineligible"), (None, "unknown")])
def test_unresolved_user_hard_conditions_preserve_failures_and_block_positive_eligibility(licence_value, expected):
    requirement = JobRequirement(fact_key="licence.any", label="A required licence")
    goal = UserGoal(intent="match", unresolved_hard_constraints=["Employer must offer a four-day work week."])
    records = [fact("licence.any", licence_value)] if licence_value is not None else []
    result = ReplayDiscovery([]).match(goal, [job(requirements=[requirement])], records, runtime=runtime())[0]
    assert result.eligibility == expected
    assert any(goal.unresolved_hard_constraints[0] in gap for gap in result.gaps)


def test_different_occupations_regions_and_ids_share_one_matching_contract():
    vacancies = [
        job("landscape-900", requirements=[JobRequirement(fact_key="design.landscape", label="Landscape portfolio")]),
        job("marine-622", title="Marine Surveyor", location="Reykjavík", requirements=[
            JobRequirement(fact_key="survey.marine", label="Survey certification", expected="Class A"),
        ]),
        job("clinical-314", title="Clinical Pharmacist", location="Tartu", requirements=[
            JobRequirement(fact_key="licence.pharmacy", label="Pharmacy licence"),
        ]),
    ]
    records = [
        fact("design.landscape", True, evidence_id="portfolio-z"),
        fact("survey.marine", "Class B", evidence_id="certificate-t"),
    ]
    results = ReplayDiscovery(vacancies).match(UserGoal(intent="match"), vacancies, records, runtime=runtime())
    assert {result.job_id: result.eligibility for result in results} == {
        "landscape-900": "eligible", "marine-622": "ineligible", "clinical-314": "unknown",
    }
    assert [result.job_id for result in results] == ["landscape-900", "clinical-314", "marine-622"]


@pytest.mark.parametrize("maximum,expected", [(5.5, "ineligible"), (8.25, "eligible")])
def test_experience_filter_uses_each_listing_threshold_and_user_limit(maximum, expected):
    requirement = JobRequirement(
        fact_key="experience.design", label="Seven years of design experience", operator="gte",
        expected=7, category="experience_years",
    )
    result = ReplayDiscovery([]).match(
        UserGoal(intent="match", max_required_experience_years=maximum),
        [job(requirements=[requirement])], [fact("experience.design", 9.5)], runtime=runtime(),
    )[0]
    assert result.eligibility == expected


def test_user_experience_cap_can_exclude_a_listing_without_candidate_experience():
    requirement = JobRequirement(
        fact_key="experience.any", label="Four years of experience", operator="gte", expected=4,
        category="experience_years",
    )
    result = ReplayDiscovery([]).match(
        UserGoal(intent="match", max_required_experience_years=1.5),
        [job(requirements=[requirement])], [], runtime=runtime(),
    )[0]
    assert result.eligibility == "ineligible"
    assert result.evidence_ids == []


def test_an_upper_experience_requirement_is_not_invented_as_a_minimum():
    requirement = JobRequirement(
        fact_key="experience.any", label="At most six years", operator="lte", expected=6,
        category="experience_years",
    )
    result = ReplayDiscovery([]).match(
        UserGoal(intent="match", max_required_experience_years=2),
        [job(requirements=[requirement])], [fact("experience.any", 3)], runtime=runtime(),
    )[0]
    assert result.eligibility == "eligible"


def test_discovery_uses_role_aliases_and_location_and_leaves_limit_to_orchestrator():
    vacancies = [
        job("planning-3", title="Senior City Planner", aliases=["urban planning"], description="GIS work."),
        job("planning-2", title="City Planning Specialist", aliases=["urban planning"]),
        job("planning-1", title="Urban Planning Intern"),
        job("other-region", title="Urban Planning Intern", location="Tartu"),
        job("other-role", title="Clinical Pharmacist"),
        job("closed", title="Urban Planning Intern", status="expired"),
    ]
    goal = UserGoal(intent="search", target_roles=["urban planning"], location="PORTO", limit=1, preferences=["GIS"])
    found = ReplayDiscovery(vacancies).discover(goal, [], runtime=runtime())
    assert [item.job_id for item in found] == ["planning-3", "planning-2", "planning-1"]


def test_internship_target_preserves_employment_type_filter():
    vacancies = [job("permanent", title="Marine Surveyor"), job("placement", title="Marine Survey Intern")]
    found = ReplayDiscovery(vacancies).discover(UserGoal(intent="search", target_roles=["internships"]), [], runtime=runtime())
    assert [item.job_id for item in found] == ["placement"]


def test_eligible_job_ranks_before_unknown_despite_preference_advantage():
    licence = JobRequirement(fact_key="licence.any", label="Required licence")
    vacancies = [
        job("unknown", title="Marine Surveyor", requirements=[licence], description="Radar remote sensing sonar navigation."),
        job("eligible", title="Marine Surveyor"),
    ]
    goal = UserGoal(intent="search", target_roles=["marine surveyor"], preferences=["radar remote sensing sonar navigation"])
    discovery = ReplayDiscovery(vacancies)
    found = discovery.discover(goal, [], runtime=runtime())
    assert [item.job_id for item in found] == ["unknown", "eligible"]
    ranked = discovery.match(goal, found, [], runtime=runtime())
    assert [item.job_id for item in ranked] == ["eligible", "unknown"]


def test_sensitive_fact_supports_eligibility_without_exposing_its_text():
    record = fact("permit.region", True, sensitive=True)
    record.text = "A private document contains confidential identifiers."
    listing = job(requirements=[JobRequirement(fact_key="permit.region", label="Required regional permit")])
    result = ReplayDiscovery([]).match(UserGoal(intent="match"), [listing], [record], runtime=runtime())[0]
    assert result.eligibility == "eligible"
    assert result.evidence_ids == [record.evidence_id]
    assert record.text not in result.model_dump_json()


def test_explicit_empty_catalog_is_not_replaced_by_default_data(monkeypatch):
    def unexpected_load():
        raise AssertionError("An explicit catalog must be respected.")

    monkeypatch.setattr("careerpilot.catalog.load_jobs", unexpected_load)
    assert ReplayDiscovery([]).discover(UserGoal(intent="search"), [], runtime=runtime()) == []


def test_catalog_inputs_and_returned_jobs_are_independent_snapshots():
    listing = job()
    discovery = ReplayDiscovery([listing])
    listing.status = "expired"
    goal = UserGoal(intent="search")
    found = discovery.discover(goal, [], runtime=runtime())
    assert len(found) == 1
    found[0].status = "expired"
    assert len(discovery.discover(goal, [], runtime=runtime())) == 1
