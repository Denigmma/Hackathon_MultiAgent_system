from __future__ import annotations

import re
from typing import List, Tuple


_MULTI_RE = re.compile(
    r"\bwhich of the companies\b|\bwhich company\b|\blowest\b|\bhighest\b|\bminimum\b|\bmaximum\b",
    flags=re.IGNORECASE,
)

_QUOTED_COMPANIES_RE = re.compile(r"\"([^\"]+)\"")


def is_multi_company_question(question_text: str) -> bool:
    if not question_text:
        return False
    if _MULTI_RE.search(question_text) is None:
        return False
    companies = _QUOTED_COMPANIES_RE.findall(question_text)
    return len(companies) >= 2


def extract_companies_from_quotes(question_text: str) -> List[str]:
    comps = _QUOTED_COMPANIES_RE.findall(question_text or "")
    out: List[str] = []
    seen = set()
    for c in comps:
        c = c.strip()
        if not c:
            continue
        if c in seen:
            continue
        seen.add(c)
        out.append(c)
    return out


def parse_min_max_direction(question_text: str) -> Tuple[bool, bool]:
    q = (question_text or "").lower()
    want_min = ("lowest" in q) or ("minimum" in q)
    want_max = ("highest" in q) or ("maximum" in q)
    return want_min, want_max
