from __future__ import annotations

from pydantic import BaseModel, Field
from typing import List


class RerankBlock(BaseModel):
    page_no: int = Field(description="1-based page number of the block")
    reasoning: str = Field(description="Short reasoning about relevance")
    relevance_score: float = Field(description="Relevance score from 0 to 1, increments of 0.1")


class RerankMultipleBlocks(BaseModel):
    block_rankings: List[RerankBlock] = Field(description="List of blocks and relevance scores")
