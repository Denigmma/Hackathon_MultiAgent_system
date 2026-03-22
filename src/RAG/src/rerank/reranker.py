from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError

from .client import ModelClient
from .schemas import RerankMultipleBlocks
from .prompts import (
    RERANK_SYSTEM_PROMPT_MULTIPLE_BLOCKS,
    RERANK_USER_PROMPT,
    ANSWER_SCHEMA_FIX_SYSTEM_PROMPT,
    ANSWER_SCHEMA_FIX_USER_PROMPT,
)


def _extract_json_object(text: str) -> Optional[str]:
    if not text:
        return None
    s = text.strip()
    i = s.find("{")
    j = s.rfind("}")
    if i == -1 or j == -1 or j <= i:
        return None
    return s[i : j + 1]


async def _fix_json(
    client: ModelClient,
    system_prompt_with_schema: str,
    bad_response: str,
) -> Optional[str]:
    prompt = ANSWER_SCHEMA_FIX_USER_PROMPT.format(
        system_prompt=system_prompt_with_schema,
        response=bad_response,
    )
    res = await client.generate(
        {
            "system_prompt": ANSWER_SCHEMA_FIX_SYSTEM_PROMPT,
            "prompt": prompt,
        }
    )
    if res.get("error_flag"):
        return None
    return _extract_json_object(res.get("text") or "")


@dataclass(frozen=True)
class RankedPage:
    doc_id: str
    page_no: int
    llm_score: float
    reasoning: str = ""


@dataclass
class PageReranker:
    client: ModelClient

    async def rerank_pages(self, question: str, pages: List[Dict[str, Any]]) -> List[RankedPage]:
        """
        pages: [
            {
                "doc_id": str(optional),
                "page_no": int,
                "text": str,
                "vector_score": float(optional),
            }
        ]

        Returns pages sorted by llm_score desc.
        """
        if not pages:
            return []

        blocks_lines: List[str] = []
        for p in pages:
            page_no = int(p["page_no"])
            doc_id = str(p.get("doc_id") or "")
            text = (p.get("text") or "").strip()

            if doc_id:
                blocks_lines.append(
                    f"---\n"
                    f"doc_id: {doc_id}\n"
                    f"page_no: {page_no}\n"
                    f"text:\n{text}\n"
                )
            else:
                blocks_lines.append(
                    f"---\n"
                    f"page_no: {page_no}\n"
                    f"text:\n{text}\n"
                )

        blocks = "\n".join(blocks_lines)

        schema = """
class RerankBlock(BaseModel):
    page_no: int
    reasoning: str
    relevance_score: float

class RerankMultipleBlocks(BaseModel):
    block_rankings: List[RerankBlock]
""".strip()

        system_prompt = (
            RERANK_SYSTEM_PROMPT_MULTIPLE_BLOCKS
            + "\n\n"
            + "Your answer should be in JSON and strictly follow this schema:\n```\n"
            + schema
            + "\n```"
        )
        user_prompt = RERANK_USER_PROMPT.format(question=question, blocks=blocks)

        res = await self.client.generate(
            {
                "system_prompt": system_prompt,
                "prompt": user_prompt,
            }
        )
        if res.get("error_flag"):
            raise RuntimeError(f"Rerank LLM error: {res.get('error_msg')}")

        raw = res.get("text") or ""
        json_str = _extract_json_object(raw)
        if json_str is None:
            json_str = await _fix_json(self.client, system_prompt, raw)
        if json_str is None:
            raise RuntimeError("Failed to parse rerank JSON response")

        try:
            data = json.loads(json_str)
            parsed = RerankMultipleBlocks.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as e:
            fixed = await _fix_json(self.client, system_prompt, raw)
            if not fixed:
                raise RuntimeError(f"Invalid rerank JSON after fix attempt: {e}")
            data2 = json.loads(fixed)
            parsed = RerankMultipleBlocks.model_validate(data2)

        page_lookup: Dict[int, str] = {}
        for p in pages:
            pn = int(p["page_no"])
            doc_id = str(p.get("doc_id") or "")
            if pn not in page_lookup:
                page_lookup[pn] = doc_id

        ranked: List[RankedPage] = []
        for block in parsed.block_rankings:
            page_no = int(block.page_no)
            ranked.append(
                RankedPage(
                    doc_id=page_lookup.get(page_no, ""),
                    page_no=page_no,
                    llm_score=float(block.relevance_score),
                    reasoning=(block.reasoning or "").strip(),
                )
            )

        ranked.sort(key=lambda x: x.llm_score, reverse=True)
        return ranked