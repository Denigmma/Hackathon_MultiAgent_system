from __future__ import annotations

from typing import List, Literal, Union

from pydantic import BaseModel, Field


class _BaseAnswerSchema(BaseModel):
    step_by_step_analysis: str = Field(default="")
    reasoning_summary: str = Field(default="")
    relevant_pages: List[int] = Field(default_factory=list)


class AnswerName(_BaseAnswerSchema):
    final_answer: Union[str, Literal["N/A"]]


class AnswerNumber(_BaseAnswerSchema):
    final_answer: Union[float, int, Literal["N/A"]]


class AnswerBoolean(_BaseAnswerSchema):
    final_answer: bool


class AnswerNames(_BaseAnswerSchema):
    final_answer: Union[List[str], Literal["N/A"]]


class AnswerText(_BaseAnswerSchema):
    final_answer: Union[str, Literal["N/A"]]