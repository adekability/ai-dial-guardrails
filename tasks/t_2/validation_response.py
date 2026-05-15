from pydantic import BaseModel, Field


class ValidationResponse(BaseModel):
    is_safe: bool = Field(description="True if the user input is safe to process")
    reason: str = Field(
        description="Brief explanation when blocked; empty string when is_safe is True"
    )
