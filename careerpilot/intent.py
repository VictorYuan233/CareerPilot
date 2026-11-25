"""Intent classification with a deterministic demonstration mode and optional ADK."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
from typing import Any

from .config import RuntimeConfig
from .schemas import UserGoal


class ClassificationError(ValueError):
    """The request could not be converted into a validated workflow goal."""


class UnsupportedIntentError(ClassificationError):
    """The requested action exceeds the available preparation workflow."""


_PREPARE = r"\b(prepare|draft|tailor|write|create|generate)\b|\bcover letter\b"
_REVISE = r"\b(revise|repair|rewrite|edit|update|change)\b"
_MATCH = r"\b(match|assess|evaluate|compare|fit|eligibility)\b"
_SEARCH = r"\b(search|find|show|discover|shortlist|recommend|look for|rank)\b"
_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}
_NUMBER = (
    r"(?:\d+(?:\.\d+)?|(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)"
    r"(?:[ -](?:one|two|three|four|five|six|seven|eight|nine))?|"
    r"zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|"
    r"fourteen|fifteen|sixteen|seventeen|eighteen|nineteen)"
)
_REQUEST_END = (
    r"\s+(?:with|without|where|requiring|that|experience)\b|[,.;]|"
    r"\s+(?:and|then|but)\s+(?:prepare|draft|tailor|write|create|generate|"
    r"revise|repair|rewrite|edit|update|change|prioriti[sz]e|prefer|exclude|assess|evaluate)\b|$"
)
_CLAUSE_END = rf"(?=\s+in\b|{_REQUEST_END})"
_CANDIDATE_TENURE = r"\b(?:i have|i possess|my experience is|i worked|i have worked)\b"


def _validate_request(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ClassificationError("Enter a career preparation request.")
    if len(text) > 8000:
        raise ClassificationError("Request exceeds the 8000-character input limit.")
    normalized = re.sub(r"\s+", " ", text.strip()).lower()
    submission_text = re.sub(
        r"\b(?:without|never|do not|don't|no)\s+(?:actually\s+)?submit(?:ting)?\b",
        "", normalized,
    )
    if re.search(r"\b(submit|auto-submit|autosubmit|apply for|apply to)\b", submission_text):
        if not re.search(_PREPARE, normalized) and not re.search(_REVISE, normalized):
            raise UnsupportedIntentError("A reviewable draft can be prepared. Application submission is unavailable.")
    return normalized


def _number_value(text: str) -> float:
    """Read decimal numerals and the documented English number words below 100."""
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        return float(text)
    return float(sum(_NUMBER_WORDS[part] for part in re.split(r"[ -]", text)))


def _limit(text: str) -> int:
    number = re.search(
        r"\b(?:find|show|discover|shortlist|recommend|rank|top|search(?: for)?|look for)\s+"
        rf"(?:(?:me|the|top|best)\s+)*({_NUMBER})\b", text,
    )
    if number is None:
        return 3
    limit = _number_value(number.group(1))
    if not limit.is_integer() or not 1 <= limit <= 10:
        raise ClassificationError("Shortlist size must be between 1 and 10.")
    return int(limit)


def _roles(text: str) -> list[str]:
    """Read a role phrase bounded by location, conditions and subsequent actions."""
    search = re.search(
        r"\b(?:find|search(?: for)?|look for|show|discover|recommend|shortlist|rank)\s+"
        rf"(?:(?:me|the|a|an|{_NUMBER}|top|best|strongest|suitable|some|available|remote)\s+)*"
        rf"(.+?){_CLAUSE_END}",
        text,
    )
    if search is None:
        return []
    role_text = search.group(1).strip()
    generic_words = {
        "intern", "interns", "internship", "internships", "role", "roles", "job", "jobs",
        "position", "positions", "opportunity", "opportunities", "match", "matches",
        "the", "a", "an", "best", "strongest", "suitable", "some", "available",
    }
    subject_words = set(re.findall(r"[a-z]+", role_text)) - generic_words
    if not subject_words:
        return ["internship"] if re.search(r"\bintern(?:ship)?s?\b", role_text) else []
    role_text = re.sub(r"\s+(?:roles?|jobs?|positions?|opportunities)$", "", role_text)
    role_text = re.sub(r"\binternships\b", "internship", role_text)
    role_text = re.sub(r"\bai\b", "AI", role_text)
    return [role_text]


def _location(text: str) -> str | None:
    """Read a Unicode location from the search clause, before requirements or context."""
    search = re.search(_SEARCH, text)
    if search is None:
        return None
    request = re.split(_REQUEST_END, text[search.end():], maxsplit=1)[0]
    match = re.search(r"\bin\s+([^\W\d_](?:[^\W\d_]|[ '-])*)$", request)
    if match:
        return match.group(1).strip().title()
    return "Remote" if re.search(r"\bremote\b", request) else None


def _experience_limit(text: str) -> float | None:
    """Read upper-limit or exclusion clauses about job-required experience."""
    pattern = (
        rf"\b(?:at most|no more than|up to|maximum(?: of)?)\s+({_NUMBER})[ -]+years?\b|"
        rf"\b(?:exclude|excluding|without)\b[^.;]*?\b(?:more than|over)\s+({_NUMBER})[ -]+years?\b"
    )
    limits = []
    for match in re.finditer(pattern, text):
        clause_prefix = re.split(r"[,.;]", text[:match.start()])[-1]
        if re.search(_CANDIDATE_TENURE, clause_prefix):
            continue
        limits.append(_number_value(match.group(1) or match.group(2)))
    return min(limits) if limits else None


def _hard_constraints(text: str) -> list[str]:
    """Preserve constraint clauses, with explicit conjunctions starting new clauses."""
    markers = r"exclude|excluding|must|only|no more than|at most|up to|maximum|without"
    following_actions = r"prioriti[sz]e|prefer|prepare|draft|tailor|write|create|generate|revise|repair|rewrite|edit|update|change"
    pattern = (
        rf"\b(?:{markers})\b.+?(?=[.;]|(?:,\s*(?:and\s+)?|\s+and\s+)"
        rf"(?:{markers}|{following_actions})\b|$)"
    )
    constraints = []
    for match in re.finditer(pattern, text):
        clause_prefix = re.split(r"[,.;]", text[:match.start()])[-1]
        if not re.search(_CANDIDATE_TENURE, clause_prefix):
            constraints.append(match.group())
    return constraints


def _unresolved_constraints(constraints: list[str], experience_limit: float | None) -> list[str]:
    """Only exact supported clauses are resolved. Mixed clauses retain their full text."""
    experience = (
        rf"(?:(?:exclude|excluding|without)\s+(?:roles?|jobs?|positions?)\s+"
        rf"(?:requiring|that require|needing|that need)\s+(?:more than|over)|"
        rf"at most|no more than|up to|maximum(?: of)?)\s+{_NUMBER}[ -]+years?"
        r"(?:\s+of\s+(?:required\s+|professional\s+)?experience)?"
    )
    truthfulness = (
        r"without\s+(?:inventing|fabricating|making up)\s+(?:any\s+)?"
        r"(?:qualifications|facts|experience|skills|achievements|numbers|credentials)|"
        r"only\s+use\s+(?:verified|confirmed|provided)\s+(?:facts|evidence)"
    )
    materials = (
        r"(?:must\s+(?:only\s+)?|only\s+)(?:include|produce|write|prepare|draft|export)\s+"
        r"(?:(?:a|an|the)\s+)?(?:resume|cover letter|application draft|application package|ats answers?)"
    )
    unresolved = []
    for constraint in constraints:
        if re.fullmatch(rf"(?:{truthfulness}|{materials})", constraint):
            continue
        if experience_limit is not None and re.fullmatch(experience, constraint):
            continue
        unresolved.append(constraint)
    return unresolved


class ReplayClassifier:
    """Parse bounded English request grammar without model calls.

    Search verbs introduce a role phrase and optional quantity. An 'in' clause
    supplies a location. Punctuation or a following action/condition ends each
    phrase. Numerals and English number words below 100 support quantities and
    experience caps. AdkClassifier handles richer language independently.
    """

    mode = "replay"

    def classify(self, text: str, *, known_job: bool = False) -> UserGoal:
        normalized = _validate_request(text)
        if re.search(_REVISE, normalized):
            intent = "revise"
        elif re.search(_PREPARE, normalized):
            intent = "prepare"
        elif re.search(_MATCH, normalized) and (
            known_job or not re.search(_SEARCH, normalized)
        ):
            intent = "match"
        elif re.search(_SEARCH, normalized):
            intent = "search"
        else:
            raise ClassificationError("Use a search, match, prepare or revise request.")
        constraints = _hard_constraints(normalized)
        experience_limit = _experience_limit(normalized)
        preferences = re.findall(
            r"\b(?:prioriti[sz]e|prefer|favor|favour)\s+"
            r"(.+?)(?=[.;]|,\s*(?:and\s+)?(?:exclude|excluding|must)|\s+and\s+exclude|$)",
            normalized,
        )
        return UserGoal(
            intent=intent,
            target_roles=_roles(normalized),
            location=_location(normalized),
            hard_constraints=constraints,
            preferences=preferences,
            limit=_limit(normalized),
            max_required_experience_years=experience_limit,
            unresolved_hard_constraints=_unresolved_constraints(constraints, experience_limit),
        )


class AdkClassifier:
    """One bounded ADK call per classification, with schema validation.

    The orchestrator reserves the session's allowance before each classification.
    This adapter enforces one request per classification and disables HTTP retries.
    The SDK obtains Gemini or Vertex credentials from its standard environment.
    An injected BaseLlm can exercise the complete ADK path in local tests.
    """

    mode = "live"

    def __init__(self, config: RuntimeConfig, *, model: Any = None):
        if config.mode != "live":
            raise ValueError("AdkClassifier requires live mode configuration.")
        self.config = config
        self._model = model
        self.last_usage: dict[str, int] = {}

    def classify(self, text: str, *, known_job: bool = False) -> UserGoal:
        """Support both a normal Python entry point and a running notebook loop."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.classify_async(text, known_job=known_job))
        with ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(
                lambda: asyncio.run(self.classify_async(text, known_job=known_job))
            ).result()

    async def classify_async(self, text: str, *, known_job: bool = False) -> UserGoal:
        _validate_request(text)
        self.last_usage = {}
        try:
            return await asyncio.wait_for(
                self._run(text, known_job=known_job), timeout=self.config.timeout_seconds,
            )
        except ImportError as error:
            raise ClassificationError("Install requirements-live.txt to enable ADK classification.") from error
        except ClassificationError:
            raise
        except TimeoutError as error:
            raise ClassificationError("Intent classification timed out.") from error
        except Exception as error:
            raise ClassificationError("Intent classification failed. Check the course model configuration.") from error

    async def _run(self, text: str, *, known_job: bool) -> UserGoal:
        from google.adk.agents import LlmAgent
        from google.adk.agents.run_config import RunConfig
        from google.adk.apps import App
        from google.adk.models.google_llm import Gemini
        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService
        from google.genai import types

        instruction = Path(__file__).with_name("prompts").joinpath("intent.md").read_text(encoding="utf-8")
        http_options = types.HttpOptions(
            timeout=int(self.config.timeout_seconds * 1000),
            retry_options=types.HttpRetryOptions(attempts=1),
        )
        agent = LlmAgent(
            name="intent_classifier",
            model=self._model if self._model is not None else Gemini(
                model=self.config.model,
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
            instruction=instruction,
            output_schema=UserGoal,
            disallow_transfer_to_parent=True,
            disallow_transfer_to_peers=True,
            generate_content_config=types.GenerateContentConfig(
                temperature=0, max_output_tokens=self.config.max_output_tokens,
                http_options=http_options,
            ),
        )
        app = App(name="careerpilot_intent", root_agent=agent)
        session_service = InMemorySessionService()
        session = await session_service.create_session(app_name=app.name, user_id="candidate")
        message = types.Content(
            role="user",
            parts=[types.Part.from_text(text=json.dumps({"request": text, "known_job": known_job}))],
        )
        final_text = ""
        async with Runner(app=app, session_service=session_service) as runner:
            async for event in runner.run_async(
                user_id="candidate", session_id=session.id, new_message=message,
                run_config=RunConfig(max_llm_calls=1),
            ):
                if event.usage_metadata:
                    for field in ("prompt_token_count", "candidates_token_count", "total_token_count"):
                        value = getattr(event.usage_metadata, field, None)
                        if value is not None:
                            self.last_usage[field] = value
                if event.is_final_response() and event.content and event.content.parts:
                    final_text = "".join(part.text for part in event.content.parts if part.text and not part.thought)
        if not final_text:
            raise ClassificationError("Intent classifier returned no structured result.")
        try:
            return UserGoal.model_validate_json(final_text)
        except ValueError as error:
            raise ClassificationError("Intent classifier returned an invalid workflow goal.") from error
