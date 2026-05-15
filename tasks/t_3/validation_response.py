from pydantic import BaseModel, Field


class ValidationResponse(BaseModel):
    is_safe: bool = Field(description="True if the LLM output contains no restricted PII leaks")
    reason: str = Field(
        description="Brief explanation when PII is detected; empty string when is_safe is True"
    )
