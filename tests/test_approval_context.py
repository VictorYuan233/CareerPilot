"""Approval must use the package and source context that the verifier reviewed."""

import json

import pytest

from careerpilot.orchestrator import Orchestrator
from careerpilot.replay import (
    ReplayDiscovery,
    ReplayPreparation,
    ReplayVerification,
    load_evidence,
    load_jobs,
)
from careerpilot.session import load_session, save_session


def workflow():
    return Orchestrator(ReplayDiscovery(), ReplayPreparation(), ReplayVerification())


def confirmed_evidence():
    evidence = load_evidence()
    answer = next(item for item in evidence if item.evidence_id == "CP-EV-004")
    answer.confirmed = True
    answer.boolean_value = True
    return evidence


def ready_for_approval():
    orchestrator = workflow()
    state = orchestrator.start("Prepare a draft", confirmed_evidence(), jobs=load_jobs()[:1])
    assert state.status == "awaiting_approval"
    return orchestrator, state


def test_start_owns_independent_evidence_and_job_snapshots():
    evidence = confirmed_evidence()
    jobs = load_jobs()[:1]
    orchestrator = workflow()
    state = orchestrator.start("Prepare a draft", evidence, jobs=jobs)
    expected_evidence = [item.model_dump() for item in state.evidence]
    expected_jobs = [item.model_dump() for item in state.jobs]
    evidence[-1].boolean_value = False
    evidence[0].text = "Changed by a later caller operation."
    jobs[0].description = "Changed job requirements."
    jobs[0].status = "expired"
    assert [item.model_dump() for item in state.evidence] == expected_evidence
    assert [item.model_dump() for item in state.jobs] == expected_jobs
    orchestrator.approve(state, state.package.version)
    assert state.status == "completed"
    assert state.approved_version == state.package.version


def test_clarification_answer_is_copied_before_entering_session():
    orchestrator = workflow()
    state = orchestrator.start("Prepare a draft", load_evidence(), jobs=load_jobs()[:1])
    answer = confirmed_evidence()[-1]
    orchestrator.clarify(state, [answer])
    assert state.status == "awaiting_approval"
    answer.boolean_value = False
    stored_answer = next(item for item in state.evidence if item.evidence_id == answer.evidence_id)
    assert stored_answer.boolean_value is True
    orchestrator.approve(state, state.package.version)
    assert state.status == "completed"


@pytest.mark.parametrize("change", ["authorisation", "evidence_text", "job_description", "selected_job"])
def test_changed_review_context_prevents_approval(change):
    orchestrator, state = ready_for_approval()
    if change == "authorisation":
        state.evidence[-1].boolean_value = False
    elif change == "evidence_text":
        state.evidence[0].text = "A materially different qualification."
    elif change == "job_description":
        state.jobs[0].description = "A different mandatory skill is now required."
    else:
        state.jobs.append(load_jobs()[2])
        state.selected_job_id = state.jobs[-1].job_id
    orchestrator.approve(state, state.package.version)
    assert state.status == "blocked"
    assert state.error_code == "review_context_changed"
    assert state.approved_version is None
    assert state.reviewed_evidence is None
    assert state.reviewed_job is None
    assert not any(item.action == "package" and item.valid for item in state.approvals)


def test_review_context_survives_json_resume_and_evidence_reordering(tmp_path):
    _, state = ready_for_approval()
    restored = load_session(save_session(state, tmp_path / "checkpoint.json"))
    assert restored.reviewed_job == state.reviewed_job
    assert restored.reviewed_evidence == state.reviewed_evidence
    restored.evidence.reverse()
    workflow().approve(restored, restored.package.version)
    assert restored.status == "completed"
    assert restored.approved_version == restored.package.version


@pytest.mark.parametrize("missing_field", ["reviewed_evidence", "reviewed_job"])
def test_legacy_checkpoint_without_review_context_cannot_be_approved(tmp_path, missing_field):
    _, state = ready_for_approval()
    payload = state.model_dump(mode="json")
    payload.pop(missing_field)
    checkpoint = tmp_path / "legacy.json"
    checkpoint.write_text(json.dumps(payload), encoding="utf-8")
    restored = load_session(checkpoint)
    workflow().approve(restored, restored.package.version)
    assert restored.status == "blocked"
    assert restored.error_code == "review_context_missing"
    assert restored.approved_version is None


def test_revision_replaces_the_review_context_with_independent_snapshots():
    orchestrator, state = ready_for_approval()
    previous_evidence = state.reviewed_evidence
    previous_job = state.reviewed_job
    orchestrator.approve(state, state.package.version)
    orchestrator.revise(state, "Shorten the draft.")
    assert state.status == "awaiting_approval"
    assert state.reviewed_evidence == previous_evidence
    assert state.reviewed_evidence is not previous_evidence
    assert state.reviewed_job == previous_job
    assert state.reviewed_job is not previous_job
    assert state.reviewed_evidence[0] is not state.evidence[0]
    assert state.reviewed_job is not state.jobs[0]
    orchestrator.approve(state, state.package.version)
    assert state.status == "completed"
