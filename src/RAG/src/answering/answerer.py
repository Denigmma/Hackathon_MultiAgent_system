from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

from pydantic import ValidationError

from .client import ModelClient
from .schemas import (
    AnswerBoolean,
    AnswerName,
    AnswerNames,
    AnswerNumber,
    AnswerText,
)
from .prompts import (
    AnswerWithRAGContextBooleanPrompt,
    AnswerWithRAGContextNamePrompt,
    AnswerWithRAGContextNamesPrompt,
    AnswerWithRAGContextNumberPrompt,
    AnswerWithRAGContextTextPrompt,
    ANSWER_SCHEMA_FIX_SYSTEM_PROMPT,
    ANSWER_SCHEMA_FIX_USER_PROMPT,
)
from .postprocess import (
    normalize_boolean_value,
    normalize_name_value,
    normalize_names_value,
    normalize_number_value,
    normalize_text_value,
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


def _build_context(pages: List[Dict[str, Any]], max_chars_per_page: int = 12000) -> str:
    chunks: List[str] = []

    for page in pages:
        page_no = int(page["page_no"])
        doc_id = str(page.get("doc_id") or "")
        text = page.get("text") or ""
        if not isinstance(text, str):
            text = ""
        text = text.strip()

        if len(text) > max_chars_per_page:
            text = text[:max_chars_per_page]

        if doc_id:
            chunks.append(f"=== Document {doc_id} / Page {page_no} ===\n{text}")
        else:
            chunks.append(f"=== Page {page_no} ===\n{text}")

    return "\n\n".join(chunks).strip()


@dataclass(frozen=True)
class AnswerResult:
    value: Any
    used_page_nos: List[int]
    doc_id: str = ""


@dataclass
class RAGAnswerer:
    client: ModelClient
    max_chars_per_page: int = 12000

    async def answer(
        self,
        question_text: str,
        kind: str,
        pages: List[Dict[str, Any]],
    ) -> AnswerResult:
        if not pages:
            return AnswerResult(value="N/A", used_page_nos=[], doc_id="")

        kind = (kind or "").strip().lower()
        context = _build_context(pages, max_chars_per_page=self.max_chars_per_page)

        system_prompt, user_prompt, parse_model = self._select_prompt_and_schema(
            kind=kind,
            question_text=question_text,
            context=context,
        )

        res = await self.client.generate(
            {
                "system_prompt": system_prompt,
                "prompt": user_prompt,
            }
        )
        if res.get("error_flag"):
            raise RuntimeError(f"Answer LLM error: {res.get('error_msg')}")

        raw = res.get("text") or ""
        json_str = _extract_json_object(raw)
        if json_str is None:
            json_str = await _fix_json(self.client, system_prompt, raw)
        if json_str is None:
            raise RuntimeError("Failed to parse answer JSON response")

        try:
            data = json.loads(json_str)
            parsed = parse_model.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as e:
            fixed = await _fix_json(self.client, system_prompt, raw)
            if not fixed:
                raise RuntimeError(f"Invalid answer JSON after fix attempt: {e}")
            data2 = json.loads(fixed)
            parsed = parse_model.model_validate(data2)

        used_pages = [int(x) for x in (parsed.relevant_pages or []) if isinstance(x, int)]
        used_pages = [p for p in used_pages if p > 0]

        value = self._normalize_value(kind=kind, value=parsed.final_answer)
        doc_id = self._resolve_doc_id_for_answer(pages)

        return AnswerResult(
            value=value,
            used_page_nos=used_pages,
            doc_id=doc_id,
        )

    def _select_prompt_and_schema(
        self,
        kind: str,
        question_text: str,
        context: str,
    ) -> tuple[str, str, Type]:
        if kind == "boolean":
            return (
                AnswerWithRAGContextBooleanPrompt.system_prompt_with_schema,
                AnswerWithRAGContextBooleanPrompt.user_prompt.format(
                    context=context,
                    question=question_text,
                ),
                AnswerBoolean,
            )

        if kind == "number":
            return (
                AnswerWithRAGContextNumberPrompt.system_prompt_with_schema,
                AnswerWithRAGContextNumberPrompt.user_prompt.format(
                    context=context,
                    question=question_text,
                ),
                AnswerNumber,
            )

        if kind == "name":
            return (
                AnswerWithRAGContextNamePrompt.system_prompt_with_schema,
                AnswerWithRAGContextNamePrompt.user_prompt.format(
                    context=context,
                    question=question_text,
                ),
                AnswerName,
            )

        if kind == "names":
            return (
                AnswerWithRAGContextNamesPrompt.system_prompt_with_schema,
                AnswerWithRAGContextNamesPrompt.user_prompt.format(
                    context=context,
                    question=question_text,
                ),
                AnswerNames,
            )

        if kind == "text":
            return (
                AnswerWithRAGContextTextPrompt.system_prompt_with_schema,
                AnswerWithRAGContextTextPrompt.user_prompt.format(
                    context=context,
                    question=question_text,
                ),
                AnswerText,
            )

        raise ValueError(f"Unsupported question kind: {kind}")

    def _normalize_value(self, kind: str, value: Any) -> Any:
        if kind == "boolean":
            return normalize_boolean_value(value)
        if kind == "number":
            return normalize_number_value(value)
        if kind == "name":
            return normalize_name_value(value)
        if kind == "names":
            return normalize_names_value(value)
        if kind == "text":
            return normalize_text_value(value)
        return value

    def _resolve_doc_id_for_answer(self, pages: List[Dict[str, Any]]) -> str:
        doc_scores: Dict[str, float] = {}

        for page in pages:
            doc_id = str(page.get("doc_id") or "")
            if not doc_id:
                continue
            score = float(page.get("llm_score", page.get("vector_score", 0.0)) or 0.0)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + score

        if not doc_scores:
            return ""

        ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[0][0]