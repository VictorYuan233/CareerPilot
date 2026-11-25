"""Intent routing and ADK adapter checks. All model responses here are local fixtures."""

import asyncio
import json

import pytest
from pydantic import ValidationError

from careerpilot.config import RuntimeConfig
from careerpilot.intent import (
    AdkClassifier,
    ClassificationError,
    ReplayClassifier,
    UnsupportedIntentError,
)


@pytest.mark.parametrize(
    ("user_text", "known_job", "expected"),
    [
        ("Find three AI engineering internships in Singapore", False, "search"),
        ("Look for five software engineering internships in London", False, "search"),
        ("Search for roles in Singapore", True, "search"),
        ("Find the strongest match", False, "search"),
        ("Assess whether my evidence meets this job's requirements", True, "match"),
        ("Compare my profile to the listed roles", False, "match"),
        ("Prepare an application for this job", True, "prepare"),
        ("Find three roles and draft a cover letter for the strongest match", False, "prepare"),
        ("Prepare a draft and submit the application", True, "prepare"),
        ("Revise the cover letter using the review feedback", True, "revise"),
    ],
)
def test_replay_routes(user_text, known_job, expected):
    assert ReplayClassifier().classify(user_text, known_job=known_job).intent == expected


def test_goal_preserves_decision_inputs():
    goal = ReplayClassifier().classify(
        "Find three suitable AI engineering internships in Singapore. "
        "Prioritize Python and machine learning. "
        "Exclude roles requiring more than two years of experience."
    )
    assert goal.target_roles == ["AI engineering internship"]
    assert goal.location == "Singapore"
    assert goal.limit == 3
    assert goal.preferences == ["python and machine learning"]
    assert goal.hard_constraints == ["exclude roles requiring more than two years of experience"]


def test_replay_extracts_size_and_location():
    goal = ReplayClassifier().classify("Show five data analyst internships in Hong Kong")
    assert (goal.limit, goal.location) == (5, "Hong Kong")
    assert goal.target_roles == ["data analyst internship"]


@pytest.mark.parametrize(("user_text", "role", "location", "limit", "intent"), [
    ("Find two editorial assistant roles in Melbourne", "editorial assistant", "Melbourne", 2, "search"),
    ("Find one logistics coordinator in Lisbon", "logistics coordinator", "Lisbon", 1, "search"),
    ("Show me four public health analyst positions in Buenos Aires", "public health analyst", "Buenos Aires", 4, "search"),
    ("Look for seven museum archivist positions in Toronto", "museum archivist", "Toronto", 7, "search"),
    ("Search two agricultural economist jobs in Nairobi", "agricultural economist", "Nairobi", 2, "search"),
    (
        "Find the top five textile design internships in Copenhagen and prepare a cover letter",
        "textile design internship", "Copenhagen", 5, "prepare",
    ),
    (
        "Discover 6 archival researcher jobs in Wellington, prioritize cataloguing, "
        "exclude roles requiring more than three years of experience, and draft an application",
        "archival researcher", "Wellington", 6, "prepare",
    ),
])
def test_unseen_roles_and_cities_use_the_same_grammar(user_text, role, location, limit, intent):
    goal = ReplayClassifier().classify(user_text)
    assert goal.target_roles == [role]
    assert goal.location == location
    assert goal.limit == limit
    assert goal.intent == intent


@pytest.mark.parametrize(("condition", "expected"), [
    ("exclude roles requiring more than two years of experience", 2),
    ("exclude roles requiring more than 3 years of experience", 3),
    ("require at most four years of experience", 4),
    ("requiring no more than 2.5 years of experience", 2.5),
    ("exclude roles requiring over seven years", 7),
    ("with up to five years of required experience", 5),
    ("with a maximum of twelve years of required experience", 12),
    ("exclude roles requiring more than twenty-five years", 25),
    ("I have three years of experience", None),
    ("I have at most four years of experience", None),
    ("requiring more than three years of experience", None),
    ("prioritize Python", None),
])
def test_required_experience_upper_bounds_are_numeric(condition, expected):
    goal = ReplayClassifier().classify(f"Find two logistics coordinator roles in Lisbon, {condition}.")
    assert goal.max_required_experience_years == expected


@pytest.mark.parametrize("condition", [
    "Exclude night shifts",
    "Must provide cold-chain facilities",
    "Only paid roles",
    "Without offshore rotations",
])
def test_unrecognized_job_conditions_are_explicitly_unresolved(condition):
    goal = ReplayClassifier().classify(f"Find roles in Singapore. {condition}.")
    assert goal.unresolved_hard_constraints == [condition.lower()]
    assert condition.lower() in goal.hard_constraints


@pytest.mark.parametrize("condition", [
    "Exclude roles requiring more than three years of experience and exclude night shifts",
    "Exclude roles requiring more than three years of experience and night shifts",
])
def test_experience_cap_does_not_hide_an_additional_job_condition(condition):
    goal = ReplayClassifier().classify(f"Find roles in Singapore. {condition}.")
    assert goal.max_required_experience_years == 3
    assert goal.unresolved_hard_constraints
    assert "night shifts" in goal.unresolved_hard_constraints[-1]


@pytest.mark.parametrize("condition", [
    "Without inventing qualifications",
    "Only use verified evidence",
    "Only include a cover letter",
])
def test_explicit_material_instructions_are_not_unresolved_job_filters(condition):
    goal = ReplayClassifier().classify(f"Find roles in Singapore. {condition}.")
    assert condition.lower() in goal.hard_constraints
    assert goal.unresolved_hard_constraints == []


def test_a_supported_experience_condition_is_fully_normalized():
    goal = ReplayClassifier().classify(
        "Find two roles in Lisbon. Exclude roles requiring more than 4 years of experience."
    )
    assert goal.max_required_experience_years == 4
    assert goal.unresolved_hard_constraints == []


@pytest.mark.parametrize("quantity", ["eleven", "twenty-five", "2.5"])
def test_generic_quantity_parser_preserves_shortlist_bounds(quantity):
    with pytest.raises(ClassificationError):
        ReplayClassifier().classify(f"Find {quantity} editorial assistant roles in Melbourne")


@pytest.mark.parametrize("user_text", [
    "Find three roles and prepare a cover letter",
    "Find jobs, prioritize Python and machine learning",
    "Show me suitable roles in Singapore",
    "Find the strongest match",
    "Find roles that match my profile",
])
def test_generic_search_targets_do_not_become_disciplines(user_text):
    assert ReplayClassifier().classify(user_text).target_roles == []


@pytest.mark.parametrize("user_text", [
    "Find internships",
    "Find intern roles in Singapore",
    "Show me suitable internships in Singapore",
])
def test_internship_type_is_preserved_without_a_discipline(user_text):
    assert ReplayClassifier().classify(user_text).target_roles == ["internship"]


@pytest.mark.parametrize("city", ["São Paulo", "Montréal", "München"])
def test_unicode_city_names_are_extracted(city):
    goal = ReplayClassifier().classify(f"Find two editorial assistant roles in {city}")
    assert goal.location == city


@pytest.mark.parametrize(("user_text", "expected"), [
    ("Find jobs requiring experience in Python", None),
    ("Find roles with experience in Montréal", None),
    ("Find jobs. My experience in Lisbon includes logistics.", None),
    ("Find jobs in São Paulo requiring experience in Python", "São Paulo"),
    ("Find jobs in Montréal and prepare a letter about experience in Python", "Montréal"),
])
def test_location_is_scoped_to_the_requested_search_clause(user_text, expected):
    assert ReplayClassifier().classify(user_text).location == expected


@pytest.mark.parametrize("user_text", [
    "Find a marine biology internship in Singapore",
    "Find marine biology internship roles and prepare an application",
    "Find marine biology internships with Python experience",
])
def test_unknown_disciplines_remain_search_constraints(user_text):
    target = ReplayClassifier().classify(user_text).target_roles
    assert len(target) == 1
    assert "marine biology" in target[0]
    assert "prepare" not in target[0]
    assert "python" not in target[0]


def test_generic_prepare_request_reaches_job_selection():
    from careerpilot.orchestrator import Orchestrator
    from careerpilot.replay import ReplayDiscovery, ReplayPreparation, ReplayVerification, load_evidence

    state = Orchestrator(ReplayDiscovery(), ReplayPreparation(), ReplayVerification()).start(
        "Find three roles and prepare a cover letter", load_evidence(),
    )
    assert state.status == "awaiting_selection"
    assert len(state.matches) == 3


def test_unknown_discipline_produces_an_empty_replay_search():
    from careerpilot.orchestrator import Orchestrator
    from careerpilot.replay import ReplayDiscovery, ReplayPreparation, ReplayVerification, load_evidence

    state = Orchestrator(ReplayDiscovery(), ReplayPreparation(), ReplayVerification()).start(
        "Find a marine biology internship in Singapore", load_evidence(),
    )
    assert state.status == "completed"
    assert state.jobs == []


def test_full_brief_goal_keeps_action_outside_constraints():
    goal = ReplayClassifier().classify(
        "Find three suitable AI engineering internships in Singapore, "
        "prioritize Python and machine learning, "
        "exclude roles requiring more than two years of experience, "
        "and prepare an application for the strongest match without inventing qualifications."
    )
    assert goal.intent == "prepare"
    assert goal.target_roles == ["AI engineering internship"]
    assert goal.location == "Singapore"
    assert goal.limit == 3
    assert goal.preferences == ["python and machine learning"]
    assert goal.hard_constraints == [
        "exclude roles requiring more than two years of experience",
        "without inventing qualifications",
    ]
    assert goal.max_required_experience_years == 2
    assert goal.unresolved_hard_constraints == []


@pytest.mark.parametrize("user_text", ["", "Hello", "Find 11 jobs", "Find 0 jobs", "x" * 8001])
def test_invalid_or_out_of_scope_inputs(user_text):
    with pytest.raises(ClassificationError):
        ReplayClassifier().classify(user_text)


def test_direct_submission_has_no_route():
    with pytest.raises(UnsupportedIntentError):
        ReplayClassifier().classify("Submit this application")
    assert ReplayClassifier().classify("Prepare a draft without submitting").intent == "prepare"


def test_config_defaults_require_no_credentials():
    assert RuntimeConfig.from_env({}).mode == "replay"
    assert RuntimeConfig().max_model_calls == 0
    assert RuntimeConfig().max_steps == 12
    with pytest.raises(ValidationError):
        RuntimeConfig.from_env({"CAREERPILOT_MODE": "live"})
    with pytest.raises(ValidationError):
        RuntimeConfig(max_repairs=3)


def test_live_config_is_explicit():
    config = RuntimeConfig.from_env({
        "CAREERPILOT_MODE": "live", "CAREERPILOT_MODEL": "course-approved-model",
        "CAREERPILOT_MAX_MODEL_CALLS": "2",
    })
    assert config.max_model_calls == 2
    assert config.model == "course-approved-model"


def _local_classifier(response, *, calls=1, usage=False):
    pytest.importorskip("google.adk")
    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types

    class LocalModel(BaseLlm):
        model: str = "local-fixture"
        request_count: int = 0

        async def generate_content_async(self, llm_request, stream=False):
            self.request_count += 1
            assert llm_request.config.max_output_tokens == 1024
            assert llm_request.config.response_schema is not None
            assert llm_request.config.http_options.retry_options.attempts == 1
            yield LlmResponse(
                content=types.Content(role="model", parts=[types.Part.from_text(text=response)]),
                usage_metadata=types.GenerateContentResponseUsageMetadata(
                    prompt_token_count=10, candidates_token_count=20, total_token_count=30,
                ) if usage else None,
            )

    model = LocalModel()
    classifier = AdkClassifier(
        RuntimeConfig(mode="live", model="course-approved-model", max_model_calls=calls),
        model=model,
    )
    return classifier, model


def test_real_adk_runner_limits_each_classification_to_one_request(monkeypatch):
    pytest.importorskip("google.adk")
    from google.adk.runners import Runner

    original_run = Runner.run_async
    call_limits = []

    def bounded_run(self, *args, **kwargs):
        call_limits.append(kwargs["run_config"].max_llm_calls)
        return original_run(self, *args, **kwargs)

    monkeypatch.setattr(Runner, "run_async", bounded_run)
    classifier, model = _local_classifier(json.dumps({
        "intent": "search", "target_roles": ["AI engineering internship"],
        "location": "Singapore", "limit": 3,
    }), usage=True)
    result = classifier.classify("Find three AI engineering internships in Singapore")
    assert result.intent == "search"
    assert model.request_count == 1
    assert classifier.last_usage == {
        "prompt_token_count": 10, "candidates_token_count": 20, "total_token_count": 30,
    }
    assert classifier.classify("Find three internships").intent == "search"
    assert model.request_count == 2
    assert call_limits == [1, 1]


def test_gemini_constructor_disables_http_retries(monkeypatch):
    pytest.importorskip("google.adk")
    from google.adk.models import google_llm

    classifier, model = _local_classifier('{"intent":"search"}')
    classifier._model = None
    options = {}

    class LocalGemini(google_llm.Gemini):
        def __init__(self, **kwargs):
            options.update(kwargs)
            super().__init__(**kwargs)

        async def generate_content_async(self, llm_request, stream=False):
            async for response in model.generate_content_async(llm_request, stream):
                yield response

    monkeypatch.setattr(google_llm, "Gemini", LocalGemini)
    assert classifier.classify("Find three internships").intent == "search"
    assert options["retry_options"].attempts == 1
    assert model.request_count == 1


def test_same_orchestrator_has_a_fresh_model_budget_for_each_session():
    from careerpilot.orchestrator import Orchestrator
    from careerpilot.replay import ReplayDiscovery, ReplayPreparation, ReplayVerification, load_evidence

    classifier, model = _local_classifier('{"intent":"search","location":"Singapore"}', usage=True)
    config = RuntimeConfig(mode="live", model="course-approved-model", max_model_calls=1)
    orchestrator = Orchestrator(
        ReplayDiscovery(), ReplayPreparation(), ReplayVerification(), classifier, config=config,
    )
    for count in (1, 2):
        state = orchestrator.start("Find three internships in Singapore", load_evidence())
        assert state.status == "completed"
        assert state.model_calls_used == 1
        assert model.request_count == count
        classification = next(event for event in state.trace if event.tool == "classify")
        assert classification.model_calls == 1
        assert classification.model_usage["total_token_count"] == 30

    exhausted = Orchestrator(
        ReplayDiscovery(), ReplayPreparation(), ReplayVerification(), classifier, max_model_calls=0,
    ).start("Find three internships in Singapore", load_evidence())
    assert exhausted.status == "exhausted"
    assert exhausted.model_calls_used == 0
    assert model.request_count == 2
    classification = next(event for event in exhausted.trace if event.tool == "classify")
    assert classification.model_calls == 0
    assert classification.model_usage == {}


def test_live_invalid_structured_output_fails_without_replay():
    classifier, model = _local_classifier('{"intent":"submit"}')
    with pytest.raises(ClassificationError):
        classifier.classify("Prepare an application draft")
    assert model.request_count == 1


def test_live_sync_entry_point_inside_notebook_loop():
    classifier, _ = _local_classifier('{"intent":"prepare"}')

    async def notebook_cell():
        return classifier.classify("Prepare a cover letter", known_job=True)

    assert asyncio.run(notebook_cell()).intent == "prepare"
    assert classifier.last_usage == {}
