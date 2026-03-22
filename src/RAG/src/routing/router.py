from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any


_PUNCT_RE = re.compile(r"[.,()]")


def _normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def normalize_company_name(name: str) -> str:
    s = name.lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _normalize_spaces(s)

    # light suffix unification (non-aggressive)
    # keep it deterministic and small
    repl = {
        "incorporated": "inc",
        "inc.": "inc",
        "corp.": "corp",
        "corporation": "corp",
        "co.": "co",
        "company": "co",
        "ltd.": "ltd",
        "limited": "ltd",
        "plc.": "plc",
    }
    tokens = s.split(" ")
    tokens2 = [repl.get(t, t) for t in tokens]
    s = _normalize_spaces(" ".join(tokens2))
    return s


def normalize_question_text(q: str) -> str:
    s = q.lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _normalize_spaces(s)
    return s


@dataclass
class RouteResult:
    sha1: str
    company_name: str
    method: str  # "substring" | "regex" | "fuzzy" (if ever used)
    confidence: float


@dataclass
class CompanyRouter:
    """
    Deterministic company routing using normalized substring matching.

    Strategy:
      - build (company_name, sha1) sorted by length desc
      - normalized company_name must appear as substring in normalized question_text
    """
    sha1_to_company: Dict[str, str]

    def __post_init__(self) -> None:
        pairs: List[Tuple[str, str]] = []
        for sha1, name in self.sha1_to_company.items():
            if not sha1 or not name:
                continue
            pairs.append((name, sha1))

        # sort by raw length desc (as requested)
        pairs.sort(key=lambda x: len(x[0]), reverse=True)

        self._pairs_raw = pairs
        self._pairs_norm = [(normalize_company_name(n), n, sha1) for (n, sha1) in pairs]

        # Regex for common patterns: For <Company>, / Did <Company> / ... of <Company> ...
        # We still use substring match as primary; regex here mainly helps to narrow candidates quickly.
        self._for_re = re.compile(r"\bfor\s+(.+?),")
        self._did_re = re.compile(r"\bdid\s+(.+?)\s+(mention|announce|report|disclose|state)\b")
        self._of_re = re.compile(r"\bof\s+(.+?)\s+(at|in|on|for|according|within|during|by)\b")

    def _try_extract_candidate_phrase(self, question: str) -> Optional[str]:
        q = question.strip()
        m = self._for_re.search(q.lower())
        if m:
            return q[m.start(1):m.end(1)]
        m = self._did_re.search(q.lower())
        if m:
            return q[m.start(1):m.end(1)]
        m = self._of_re.search(q.lower())
        if m:
            return q[m.start(1):m.end(1)]
        return None

    def route(self, question_text: str) -> RouteResult:
        q_norm = normalize_question_text(question_text)
        phrase = self._try_extract_candidate_phrase(question_text)
        phrase_norm = normalize_question_text(phrase) if phrase else None

        # First pass: if we extracted a likely company phrase, try matching inside that phrase
        if phrase_norm:
            for c_norm, c_raw, sha1 in self._pairs_norm:
                if c_norm and c_norm in phrase_norm:
                    return RouteResult(sha1=sha1, company_name=c_raw, method="substring", confidence=1.0)

        # Main deterministic routing: normalized company substring in normalized question
        for c_norm, c_raw, sha1 in self._pairs_norm:
            if c_norm and c_norm in q_norm:
                return RouteResult(sha1=sha1, company_name=c_raw, method="substring", confidence=1.0)

        # If nothing matched, do a very conservative fallback: try weaker containment by removing corporate suffix tokens
        # Still deterministic, no fuzzy by default.
        q_tokens = set(q_norm.split(" "))
        best: Optional[Tuple[float, str, str]] = None  # score, company_raw, sha1

        drop_tokens = {"inc", "corp", "co", "ltd", "plc"}
        for c_norm, c_raw, sha1 in self._pairs_norm:
            toks = [t for t in c_norm.split(" ") if t and t not in drop_tokens]
            if not toks:
                continue
            hit = sum(1 for t in toks if t in q_tokens)
            score = hit / max(len(toks), 1)
            if score >= 0.95:  # very strict
                # prefer longer raw name (already sorted) but keep best score
                if best is None or score > best[0]:
                    best = (score, c_raw, sha1)

        if best:
            score, c_raw, sha1 = best
            return RouteResult(sha1=sha1, company_name=c_raw, method="regex", confidence=float(score))

        raise ValueError("Could not route company deterministically from question_text")
