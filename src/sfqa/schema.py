"""모델이 내는 답의 형식. LM Studio에는 json_schema(strict)로 그대로 보낸다."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, ValidationError

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["answered", "not_found"]},
        "answer": {"type": "string"},
        "quotes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"chapter": {"type": "integer"}, "text": {"type": "string"}},
                "required": ["chapter", "text"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["status", "answer", "quotes"],
    "additionalProperties": False,
}


class ModelQuote(BaseModel):
    chapter: int
    text: str


class ModelAnswer(BaseModel):
    status: Literal["answered", "not_found"]
    answer: str
    quotes: list[ModelQuote] = Field(default_factory=list)


def parse_answer(data: dict | None) -> ModelAnswer | None:
    if data is None:
        return None
    try:
        return ModelAnswer.model_validate(data)
    except ValidationError:
        return None
