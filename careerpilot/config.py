"""Shared runtime limits and opt-in course model configuration."""

from __future__ import annotations

import os
from typing import Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RuntimeConfig(BaseModel):
    """Offline by default. Live operation requires an explicit model and budget."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: Literal["replay", "live"] = "replay"
    model: str | None = None
    max_model_calls: int = Field(default=0, ge=0, le=100)
    max_output_tokens: int = Field(default=1024, ge=128, le=8192)
    timeout_seconds: float = Field(default=30, gt=0, le=300)
    max_steps: int = Field(default=12, ge=1, le=100)
    max_repairs: int = Field(default=2, ge=0, le=2)

    @model_validator(mode="after")
    def require_live_configuration(self) -> RuntimeConfig:
        if self.mode == "live" and (
            not self.model or not self.model.strip() or self.max_model_calls == 0
        ):
            raise ValueError("Live mode requires a course-approved model and a positive call budget.")
        return self

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> RuntimeConfig:
        """Read only application settings. Credentials remain with the model SDK."""
        values = os.environ if environ is None else environ
        fields = {
            "mode": "CAREERPILOT_MODE",
            "model": "CAREERPILOT_MODEL",
            "max_model_calls": "CAREERPILOT_MAX_MODEL_CALLS",
            "max_output_tokens": "CAREERPILOT_MAX_OUTPUT_TOKENS",
            "timeout_seconds": "CAREERPILOT_TIMEOUT_SECONDS",
            "max_steps": "CAREERPILOT_MAX_STEPS",
            "max_repairs": "CAREERPILOT_MAX_REPAIRS",
        }
        return cls.model_validate({field: values[key] for field, key in fields.items() if key in values})


Config = RuntimeConfig
