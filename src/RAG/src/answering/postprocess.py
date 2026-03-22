from __future__ import annotations

import re
from typing import Any, List, Union


_NUM_CLEAN_RE = re.compile(r"[,\s]")
_PERCENT_RE = re.compile(r"%$")
_PARENS_NEG_RE = re.compile(r"^\((.*)\)$")


def _is_na(val: Any) -> bool:
    if not isinstance(val, str):
        return False
    return val.strip().upper() == "N/A"


def normalize_number_value(val: Any) -> Union[int, float, str]:
    if isinstance(val, (int, float)):
        return val

    if isinstance(val, str):
        s = val.strip()
        if _is_na(s):
            return "N/A"

        s = s.replace("−", "-")

        m = _PARENS_NEG_RE.match(s)
        if m:
            s = "-" + m.group(1)

        if _PERCENT_RE.search(s):
            s = _PERCENT_RE.sub("", s)

        if s.count(",") == 1 and "." not in s:
            s = s.replace(",", ".")

        s = _NUM_CLEAN_RE.sub("", s)

        try:
            if "." in s:
                return float(s)
            return int(s)
        except Exception:
            return "N/A"

    return "N/A"


def normalize_names_value(val: Any) -> Union[List[str], str]:
    if isinstance(val, list):
        out: List[str] = []
        seen = set()

        for item in val:
            if not isinstance(item, str):
                continue

            t = item.strip()
            if not t:
                continue
            if t.upper() == "N/A":
                continue

            if t not in seen:
                out.append(t)
                seen.add(t)

        return out if out else "N/A"

    if isinstance(val, str) and _is_na(val):
        return "N/A"

    return "N/A"


def normalize_name_value(val: Any) -> str:
    if isinstance(val, str):
        t = val.strip()
        return t if t else "N/A"
    return "N/A"


def normalize_boolean_value(val: Any) -> bool:
    if isinstance(val, bool):
        return val

    if isinstance(val, str):
        t = val.strip().lower()
        if t in {"true", "yes", "1", "y"}:
            return True
        if t in {"false", "no", "0", "n"}:
            return False

    return False


def normalize_text_value(val: Any) -> str:
    if isinstance(val, str):
        t = val.strip()
        if not t:
            return "N/A"
        if _is_na(t):
            return "N/A"

        t = re.sub(r"\s+", " ", t)
        return t

    return "N/A"