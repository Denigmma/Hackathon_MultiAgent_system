from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Union


def read_json(path: str | Path) -> Any:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: Any, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_questions(path: str | Path) -> List[Dict[str, Any]]:
    data = read_json(path)
    if not isinstance(data, list):
        raise ValueError("questions.json must be a list of {text, kind}")
    out: List[Dict[str, Any]] = []
    for i, x in enumerate(data):
        if not isinstance(x, dict):
            raise ValueError(f"questions[{i}] must be an object")
        t = x.get("text")
        k = x.get("kind")
        if not isinstance(t, str) or not isinstance(k, str):
            raise ValueError(f"questions[{i}] must have 'text' and 'kind' as strings")
        out.append({"text": t, "kind": k})
    return out
