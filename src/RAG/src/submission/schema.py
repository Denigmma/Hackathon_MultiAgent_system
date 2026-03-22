from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Optional, List, Union, Literal


class SourceReference(BaseModel):
    pdf_sha1: str = Field(..., description="SHA1 hash of the PDF file")
    page_index: int = Field(..., description="Zero-based physical page number in the PDF file")


class Answer(BaseModel):
    question_text: Optional[str] = Field(None, description="Text of the question")
    kind: Optional[Literal["number", "name", "boolean", "names"]] = Field(
        None, description="Kind of the question"
    )
    value: Union[float, int, str, bool, List[str], Literal["N/A"]] = Field(
        ..., description="Answer to the question, according to the question schema"
    )
    references: List[SourceReference] = Field([], description="References to the source material in the PDF file")


class AnswerSubmission(BaseModel):
    team_email: str = Field(..., description="Email that your team used to register for the challenge")
    submission_name: str = Field(..., description="Unique name of the submission (e.g. experiment name)")
    answers: List[Answer] = Field(..., description="List of answers to the questions")
