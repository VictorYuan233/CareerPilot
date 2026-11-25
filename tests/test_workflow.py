"""Integration claims for Member 1 using local, synthetic specialist adapters."""

import json

import pytest
from pydantic import ValidationError

from careerpilot.config import RuntimeConfig
from careerpilot.intent import ReplayClassifier
from careerpilot.orchestrator import Orchestrator
from careerpilot.replay import (
    ReplayDiscovery,
    ReplayPreparation,
    ReplayVerification,
    load_evidence,
    load_jobs,
)
from careerpilot.schemas import ReviewResult
from careerpilot.session import load_session, save_session


class CountingDiscovery(ReplayDiscovery):
    def __init__(self):
        super().__init__()
        self.discover_calls = 0
        self.match_calls = 0

    def discover(self, *args, **kwargs):
        self.discover_calls += 1
        return super().discover(*args, **kwargs)

    def match(self, *args, **kwargs):
        self.match_calls += 1
        return super().match(*args, **kwargs)


class CountingPreparation(ReplayPreparation):
    def __init__(self):
        self.calls = 0

    def prepare(self, *args, **kwargs):
        self.calls += 1
        return super().prepare(*args, **kwargs)


class AlwaysRevise:
    def __init__(self):
        self.calls = 0

    def review(self, *args, **kwargs):
        self.calls += 1
        instruction = "Focus the cover letter on Python coursework." if self.calls % 2 else "Remove SQL."
        return ReviewResult(decision="revise", repair_instructions=[instruction])


def confirmed_evidence():
    evidence = load_evidence()
    authorisation = next(item for item in evidence if item.evidence_id == "CP-EV-004")
    authorisation.confirmed = True
    authorisation.boolean_value = True
    authorisation.text = "The synthetic candidate confirms work authorisation for Singapore."
    return evidence


def workflow(**overrides):
    components = {
        "discovery": CountingDiscovery(),
        "preparation": CountingPreparation(),
        "verification": ReplayVerification(),
        "classifier": ReplayClassifier(),
    }
    components.update(overrides)
    return Orchestrator(**components)


def ready_for_approval(orchestrator=None):
    orchestrator = orchestrator or workflow()
    state = orchestrator.start(
        "Prepare an application for this job", confirmed_evidence(), jobs=load_jobs()[:1]
    )
    assert state.status == "awaiting_approval"
    return orchestrator, state


def test_default_entry_point_works_without_classifier_injection():
    orchestrator = Orchestrator(ReplayDiscovery(), ReplayPreparation(), ReplayVerification())
    state = orchestrator.start("Find three AI engineering internships in Singapore", load_evidence())
    assert state.status == "completed"
    assert state.goal.intent == "search"


def test_search_returns_shortlist_without_preparing_or_requiring_approval():
    orchestrator = workflow()
    state = orchestrator.start(
        "Find three roles in Singapore. "
        "Exclude roles requiring more than two years of experience.",
        load_evidence(),
    )
    assert state.status == "completed"
    assert len(state.matches) == 3
    assert {match.eligibility for match in state.matches} == {"unknown", "ineligible"}
    assert orchestrator.discovery.discover_calls == 1
    assert orchestrator.discovery.match_calls == 1
    assert orchestrator.preparation.calls == 0
    assert state.package is None
    assert state.approved_version is None


def test_known_job_match_skips_discovery_and_material_generation():
    orchestrator = workflow()
    state = orchestrator.start(
        "Assess my evidence against this job", load_evidence(), jobs=load_jobs()[:1]
    )
    assert state.status == "completed"
    assert state.goal.intent == "match"
    assert state.matches[0].eligibility == "unknown"
    assert orchestrator.discovery.discover_calls == 0
    assert orchestrator.preparation.calls == 0


def test_fresh_search_ignores_an_unrelated_supplied_job():
    orchestrator = workflow()
    state = orchestrator.start(
        "Find one data science internship in London", load_evidence(), jobs=load_jobs()[:1]
    )
    assert state.status == "completed" and state.matches == [] and state.jobs == []
    assert orchestrator.discovery.discover_calls == 1
    assert "discover" in [step.action for step in state.plan]


def test_limit_applies_to_ranked_results():
    state = workflow().start("Find one data science internship in Singapore", load_evidence())
    assert state.status == "completed"
    assert [row.job_id for row in state.matches] == ["CP-JOB-003"]


def test_structured_plan_describes_search_and_known_job_preparation_routes():
    orchestrator = workflow()
    search = orchestrator.start("Find three roles", load_evidence())
    known_job = orchestrator.start("Prepare a draft", confirmed_evidence(), jobs=load_jobs()[:1])
    search_actions = [step.action for step in search.plan]
    prepare_actions = [step.action for step in known_job.plan]
    assert search.status == "completed"
    assert search_actions.index("discover") < search_actions.index("match")
    assert search_actions[-1] == "deliver_shortlist"
    assert "prepare" not in search_actions
    assert "approve" not in search_actions
    assert "discover" not in prepare_actions
    assert "select_job" not in prepare_actions
    assert prepare_actions.index("match") < prepare_actions.index("prepare")
    assert prepare_actions.index("prepare") < prepare_actions.index("review")
    assert prepare_actions.index("review") < prepare_actions.index("approve")
    assert prepare_actions[-1] == "deliver_package"
    assert next(step for step in known_job.plan if step.action == "clarify").condition


def test_known_job_prepare_pauses_and_restores_before_clarification(tmp_path):
    orchestrator = workflow()
    state = orchestrator.start(
        "Prepare an application for this job", load_evidence(), jobs=load_jobs()[:1]
    )
    assert state.status == "awaiting_clarification"
    assert state.pending_questions
    assert state.package is None
    assert orchestrator.discovery.discover_calls == 0
    assert orchestrator.preparation.calls == 0
    checkpoint = save_session(state, tmp_path / "checkpoint.json")
    restored = load_session(checkpoint)
    assert restored.model_dump() == state.model_dump()

    fresh_orchestrator = workflow()
    fresh_orchestrator.advance(restored)
    assert fresh_orchestrator.discovery.match_calls == 0
    assert restored.status == "awaiting_clarification"
    clarification = [item for item in confirmed_evidence() if item.evidence_id == "CP-EV-004"]
    fresh_orchestrator.clarify(restored, clarification)
    assert restored.status == "awaiting_approval"
    assert restored.review.decision == "pass"
    assert fresh_orchestrator.preparation.calls == 1
    assert restored.approved_version is None
    fresh_orchestrator.approve(restored, restored.package.version)
    assert restored.status == "completed"
    assert restored.approved_version == restored.package.version


def test_discovered_job_requires_selection_and_excludes_ineligible_choice():
    orchestrator = workflow()
    state = orchestrator.start(
        "Find three roles and prepare a cover letter. "
        "Exclude roles requiring more than two years of experience.",
        confirmed_evidence(),
    )
    assert state.status == "awaiting_selection"
    assert orchestrator.preparation.calls == 0
    with pytest.raises(ValueError):
        orchestrator.select_job(state, "CP-JOB-002")
    assert state.status == "awaiting_selection"
    assert state.selected_job_id is None
    orchestrator.select_job(state, "CP-JOB-001")
    assert state.status == "awaiting_approval"
    assert state.package.job_id == "CP-JOB-001"


def test_unconfirmed_clarification_preserves_pending_checkpoint():
    orchestrator = workflow()
    state = orchestrator.start(
        "Prepare an application", load_evidence(), jobs=load_jobs()[:1]
    )
    with pytest.raises(ValueError):
        orchestrator.clarify(state, load_evidence()[-1:])
    assert state.status == "awaiting_clarification"
    assert state.package is None


@pytest.mark.parametrize(
    ("answer", "expected_status", "expected_eligibility"),
    [(False, "blocked", "ineligible"), (None, "awaiting_clarification", "unknown")],
)
def test_confirmed_authorisation_requires_an_explicit_positive_answer(
    answer, expected_status, expected_eligibility
):
    orchestrator = workflow()
    state = orchestrator.start(
        "Prepare an application", load_evidence(), jobs=load_jobs()[:1]
    )
    clarification = next(item for item in load_evidence() if item.evidence_id == "CP-EV-004")
    clarification.confirmed = True
    clarification.boolean_value = answer
    clarification.text = "The synthetic candidate supplied an eligibility clarification."
    orchestrator.clarify(state, [clarification])
    assert state.status == expected_status
    assert state.matches[0].eligibility == expected_eligibility
    assert state.package is None
    assert orchestrator.preparation.calls == 0


def test_legacy_checkpoint_without_boolean_answer_remains_unresolved(tmp_path):
    orchestrator = workflow()
    state = orchestrator.start(
        "Prepare an application", load_evidence(), jobs=load_jobs()[:1]
    )
    payload = state.model_dump(mode="json")
    for evidence in payload["evidence"]:
        evidence.pop("boolean_value", None)
        if evidence["evidence_id"] == "CP-EV-004":
            evidence["confirmed"] = True
    checkpoint = tmp_path / "legacy.json"
    checkpoint.write_text(json.dumps(payload), encoding="utf-8")
    restored = load_session(checkpoint)
    authorisation = next(item for item in restored.evidence if item.evidence_id == "CP-EV-004")
    assert authorisation.boolean_value is None
    orchestrator.clarify(restored, [authorisation])
    assert restored.status == "awaiting_clarification"
    assert restored.matches[0].eligibility == "unknown"
    assert restored.package is None


@pytest.mark.parametrize("checkpoint", ["selection", "clarification", "approval"])
def test_reject_ends_each_human_checkpoint(checkpoint):
    orchestrator = workflow()
    if checkpoint == "selection":
        state = orchestrator.start("Find three roles and prepare a draft", load_evidence())
    elif checkpoint == "clarification":
        state = orchestrator.start("Prepare a draft", load_evidence(), jobs=load_jobs()[:1])
    else:
        _, state = ready_for_approval(orchestrator)
    orchestrator.reject(state)
    calls = orchestrator.preparation.calls
    orchestrator.advance(state)
    assert state.status == "rejected"
    assert state.approved_version is None
    assert orchestrator.preparation.calls == calls
    with pytest.raises(ValueError):
        orchestrator.approve(state, 1)


def test_approval_is_bound_to_current_reviewed_version_and_contents():
    orchestrator, state = ready_for_approval()
    with pytest.raises(ValueError):
        orchestrator.approve(state, state.package.version + 1)
    assert state.status == "awaiting_approval"
    state.package.cover_letter += " An unreviewed claim."
    orchestrator.approve(state, state.package.version)
    assert state.status == "blocked"
    assert state.approved_version is None


def test_revising_approved_package_requires_new_review_and_approval():
    orchestrator, state = ready_for_approval()
    old_version = state.package.version
    old_letter = state.package.cover_letter
    orchestrator.approve(state, old_version)
    old_approval = next(record for record in state.approvals if record.action == "package")
    assert old_approval.valid
    orchestrator.revise(state, "Focus the cover letter on the Python coursework.")
    assert state.status == "awaiting_approval"
    assert state.package.version > old_version
    assert state.package.cover_letter != old_letter
    assert state.review.decision == "pass"
    assert state.approved_version is None
    assert old_approval.valid is False
    with pytest.raises(ValueError):
        orchestrator.approve(state, old_version)
    orchestrator.approve(state, state.package.version)
    assert state.status == "completed"
    valid_approvals = [record for record in state.approvals if record.action == "package" and record.valid]
    assert len(valid_approvals) == 1
    assert valid_approvals[0].package_version == state.package.version


def test_revise_route_accepts_previous_package_and_skips_discovery():
    _, previous_state = ready_for_approval()
    orchestrator = workflow()
    state = orchestrator.start(
        "Revise this cover letter to focus on Python coursework.", confirmed_evidence(), jobs=load_jobs()[:1],
        package=previous_state.package,
    )
    assert state.goal.intent == "revise"
    assert state.status == "awaiting_approval"
    assert state.package.version > previous_state.package.version
    assert orchestrator.discovery.discover_calls == 0


def test_unsupported_revision_cannot_produce_a_new_approved_version():
    orchestrator, state = ready_for_approval()
    previous = state.package.model_copy(deep=True)
    orchestrator.revise(state, "Turn the draft into a limerick.")
    assert state.error_code == "unsupported_revision"
    assert state.approved_version is None
    assert state.package == previous


def test_repeated_verifier_repair_stops_after_two_attempts():
    verifier = AlwaysRevise()
    orchestrator = workflow(verification=verifier)
    state = orchestrator.start("Prepare a draft", confirmed_evidence(), jobs=load_jobs()[:1])
    assert state.status == "exhausted"
    assert state.repair_count == 2
    assert verifier.calls == 3
    assert orchestrator.preparation.calls == 3
    assert state.approved_version is None
    orchestrator.advance(state)
    assert verifier.calls == 3


@pytest.mark.parametrize("repair_limit", [0, 1])
def test_runtime_config_can_reduce_repair_allowance(repair_limit):
    verifier = AlwaysRevise()
    orchestrator = workflow(
        config=RuntimeConfig(max_repairs=repair_limit), verification=verifier
    )
    state = orchestrator.start("Prepare a draft", confirmed_evidence(), jobs=load_jobs()[:1])
    assert state.status == "exhausted"
    assert state.max_repairs == repair_limit
    assert state.repair_count == repair_limit
    assert verifier.calls == repair_limit + 1
    assert orchestrator.preparation.calls == repair_limit + 1
    assert ("repair" in [step.action for step in state.plan]) == (repair_limit > 0)


def test_explicit_feedback_consent_survives_checkpoint_and_reaches_next_session(tmp_path):
    class FeedbackAwarePreparation(ReplayPreparation):
        def __init__(self):
            self.received_feedback = None

        def prepare(self, *args, runtime, **kwargs):
            self.received_feedback = runtime.approved_feedback_ids
            return super().prepare(*args, runtime=runtime, **kwargs)

    orchestrator, state = ready_for_approval()
    assert state.approved_feedback_ids == []
    orchestrator.approve(state, state.package.version, feedback_ids=["CP-FB-001", "CP-FB-001"])
    assert state.approved_feedback_ids == ["CP-FB-001"]
    consent = next(record for record in state.approvals if record.action == "package")
    assert consent.feedback_ids == ["CP-FB-001"]
    restored = load_session(save_session(state, tmp_path / "approved.json"))
    preparation = FeedbackAwarePreparation()
    next_orchestrator = workflow(preparation=preparation)
    next_session = next_orchestrator.start(
        "Prepare a draft", confirmed_evidence(), jobs=load_jobs()[:1],
        approved_feedback_ids=restored.approved_feedback_ids,
    )
    assert next_session.session_id != restored.session_id
    assert next_session.status == "awaiting_approval"
    assert preparation.received_feedback == ("CP-FB-001",)
    assert next_session.approved_version is None


def test_step_budget_stops_before_discovery_call():
    orchestrator = workflow(max_steps=1)
    state = orchestrator.start("Find three roles", load_evidence())
    assert state.status == "exhausted"
    assert state.steps_used == 1
    assert orchestrator.discovery.discover_calls == 0
    assert state.package is None


def test_model_budget_stops_before_live_classifier_call():
    class LocalLiveClassifier:
        mode = "live"

        def __init__(self):
            self.calls = 0

        def classify(self, text, *, known_job=False):
            self.calls += 1
            return ReplayClassifier().classify(text, known_job=known_job)

    classifier = LocalLiveClassifier()
    orchestrator = workflow(classifier=classifier, max_model_calls=0)
    state = orchestrator.start("Find three roles", load_evidence())
    assert state.status == "exhausted"
    assert classifier.calls == 0
    assert state.model_calls_used == 0


def test_specialist_model_call_is_counted_and_exhaustion_prevents_retry():
    class ModelDiscovery(CountingDiscovery):
        def discover(self, goal, evidence, *, runtime):
            runtime.reserve_model_call()
            return super().discover(goal, evidence, runtime=runtime)

        def match(self, goal, jobs, evidence, *, runtime):
            runtime.reserve_model_call()
            return super().match(goal, jobs, evidence, runtime=runtime)

    discovery = ModelDiscovery()
    orchestrator = workflow(discovery=discovery, max_model_calls=1)
    state = orchestrator.start("Find three roles", load_evidence())
    assert state.status == "exhausted"
    assert state.model_calls_used == 1
    assert discovery.discover_calls == 1
    assert discovery.match_calls == 0


def test_tool_failure_terminates_with_redacted_trace():
    secret = "private.person@example.test credential-value-123"

    class BrokenDiscovery(ReplayDiscovery):
        def discover(self, *args, **kwargs):
            raise RuntimeError(secret)

    state = workflow(discovery=BrokenDiscovery()).start("Find three roles", load_evidence())
    assert state.status == "blocked"
    trace_json = json.dumps([item.model_dump(mode="json") for item in state.trace])
    assert secret not in trace_json
    assert "private.person" not in trace_json
    assert any(item.outcome == "blocked" and item.error_code for item in state.trace)
    assert [item.tool for item in state.trace[-2:]] == ["discover", "terminate"]
    assert state.package is None


def test_match_result_with_unknown_evidence_id_cannot_reach_preparation():
    class BadEvidenceDiscovery(ReplayDiscovery):
        def match(self, *args, **kwargs):
            matches = super().match(*args, **kwargs)
            matches[0].evidence_ids.append("MISSING-EVIDENCE")
            return matches

    orchestrator = workflow(discovery=BadEvidenceDiscovery())
    state = orchestrator.start("Prepare a draft", confirmed_evidence(), jobs=load_jobs()[:1])
    assert state.status == "blocked"
    assert orchestrator.preparation.calls == 0


def test_job_expiring_after_review_prevents_final_approval():
    orchestrator, state = ready_for_approval()
    state.jobs[0].status = "expired"
    orchestrator.approve(state, state.package.version)
    assert state.status == "blocked"
    assert state.approved_version is None


def test_checkpoint_rejects_unknown_schema_version(tmp_path):
    _, state = ready_for_approval()
    payload = state.model_dump(mode="json")
    payload["schema_version"] = 2
    checkpoint = tmp_path / "unsupported.json"
    checkpoint.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_session(checkpoint)


def test_failed_checkpoint_replace_preserves_prior_valid_file(tmp_path, monkeypatch):
    _, state = ready_for_approval()
    checkpoint = save_session(state, tmp_path / "checkpoint.json")
    original = checkpoint.read_bytes()

    def fail_replace(*args):
        raise OSError("simulated filesystem failure")

    monkeypatch.setattr("careerpilot.session.os.replace", fail_replace)
    with pytest.raises(OSError):
        save_session(state, checkpoint)
    assert checkpoint.read_bytes() == original
    assert list(tmp_path.glob("*.tmp")) == []
