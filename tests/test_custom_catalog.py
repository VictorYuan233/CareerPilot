"""A new occupation, location and profile should require only new records."""

import json

import pytest

from careerpilot.catalog import load_evidence, load_jobs
from careerpilot.orchestrator import Orchestrator
from careerpilot.replay import ReplayDiscovery, ReplayPreparation, ReplayVerification


@pytest.mark.parametrize(
    "title,city,key,operator,value,expected,claim",
    [
        ("Museum Archivist", "Lisbon", "records.language", "eq", "Portuguese", "Portuguese",
         "Catalogued a photographic archive in Portuguese."),
        ("Shift Supervisor", "Melbourne", "service.team_size", "gte", 8, 5,
         "Supervised a team of eight during evening service."),
    ],
)
def test_user_supplied_files_work_from_search_to_approval(
    tmp_path, title, city, key, operator, value, expected, claim
):
    # No profiles, job IDs, source IDs or requirement keys from the course fixtures.
    job_file = tmp_path / "openings.json"
    job_file.write_text(json.dumps({"jobs": [{
        "job_id": "opening-x", "title": title, "company": "Example Cooperative",
        "location": city, "source_url": "https://example.org/opening", "description": title,
        "fetched_at": "2026-11-25T00:00:00Z", "status": "open", "requirements_complete": True,
        "requirements": [{"fact_key": key, "label": "Documented qualification",
                          "operator": operator, "expected": expected}],
    }]}), encoding="utf-8")
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"evidence": [{
        "evidence_id": "record-k", "source_id": "portfolio-k", "text": claim,
        "provenance": "Fictional work sample", "confirmed": True, "sensitive": False,
        "fact_key": key, "fact_value": value,
    }]}), encoding="utf-8")
    flow = Orchestrator(ReplayDiscovery(load_jobs(job_file)), ReplayPreparation(), ReplayVerification())
    state = flow.start(f"Find one {title} in {city} and prepare a draft.", load_evidence(profile))
    assert state.status == "awaiting_selection"
    assert [(m.job_id, m.eligibility) for m in state.matches] == [("opening-x", "eligible")]
    flow.select_job(state, "opening-x")
    assert state.status == "awaiting_approval" and state.review.decision == "pass"
    assert claim in state.package.resume_draft
    assert state.package.claim_evidence[claim] == ["record-k"]
    assert state.package.ats_answers == {}
    flow.approve(state, state.package.version)
    assert state.status == "completed" and state.approved_version == state.package.version
