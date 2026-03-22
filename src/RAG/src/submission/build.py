from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from .schema import AnswerSubmission, Answer, SourceReference


def _as_na_upper(val: Any) -> Any:
    if isinstance(val, str) and val.strip().lower() == "n/a":
        return "N/A"
    return val


def _sanitize_number(val: Any) -> Union[int, float, str]:
    # Must be a pure number or "N/A"
    if isinstance(val, (int, float)):
        return val
    if isinstance(val, str):
        t = val.strip()
        if t.lower() == "n/a":
            return "N/A"
        # if accidentally got numeric as string, try parse
        t2 = t.replace("−", "-").replace(",", "").replace(" ", "")
        try:
            if "." in t2:
                return float(t2)
            return int(t2)
        except Exception:
            return "N/A"
    return "N/A"


def _sanitize_names(val: Any) -> Union[List[str], str]:
    if isinstance(val, list):
        out = []
        for x in val:
            if isinstance(x, str):
                s = x.strip()
                if s:
                    out.append(s)
        return out if out else "N/A"
    if isinstance(val, str):
        if val.strip().lower() == "n/a":
            return "N/A"
    return "N/A"


def _sanitize_name(val: Any) -> str:
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return "N/A"
        if s.lower() == "n/a":
            return "N/A"
        return s
    return "N/A"


def _sanitize_boolean(val: Any) -> Union[bool, str]:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        t = val.strip().lower()
        if t in {"true", "yes", "y", "1"}:
            return True
        if t in {"false", "no", "n", "0"}:
            return False
        if t in {"n/a", "na"}:
            return "N/A"
    return False


def page_nos_to_references(pdf_sha1: str, page_nos: Sequence[int], max_refs: int = 3) -> List[SourceReference]:
    refs: List[SourceReference] = []
    seen = set()
    for pn in page_nos:
        if not isinstance(pn, int):
            continue
        if pn <= 0:
            continue
        if pn in seen:
            continue
        seen.add(pn)
        refs.append(SourceReference(pdf_sha1=pdf_sha1, page_index=pn - 1))
        if len(refs) >= int(max_refs):
            break
    return refs


def build_submission(
    team_email: str,
    submission_name: str,
    questions: Sequence[Dict[str, Any]],
    results: Sequence[Dict[str, Any]],
    max_refs_per_answer: int = 3,
) -> AnswerSubmission:
    if len(questions) != len(results):
        raise ValueError(f"questions/results length mismatch: {len(questions)} vs {len(results)}")

    answers: List[Answer] = []
    for q, r in zip(questions, results):
        q_text = str(q.get("text", ""))
        kind = str(q.get("kind", "")).strip().lower()
        sha1 = r.get("pdf_sha1")
        used_pages = r.get("used_page_nos") or []
        value = r.get("value")

        if kind == "number":
            value2 = _sanitize_number(value)
        elif kind == "names":
            value2 = _sanitize_names(value)
        elif kind == "name":
            value2 = _sanitize_name(value)
        elif kind == "boolean":
            value2 = _sanitize_boolean(value)
        else:
            value2 = _as_na_upper(value)

        refs: List[SourceReference] = []
        if isinstance(sha1, str) and sha1 and isinstance(used_pages, list):
            refs = page_nos_to_references(sha1, used_pages, max_refs=max_refs_per_answer)

        answers.append(
            Answer(
                question_text=q_text,
                kind=kind if kind in {"number", "name", "boolean", "names"} else None,
                value=value2,
                references=refs,
            )
        )

    return AnswerSubmission(team_email=team_email, submission_name=submission_name, answers=answers)


def save_submission_json(submission: AnswerSubmission, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = submission.model_dump()
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
