from __future__ import annotations

import enum
import re
from typing import List, Sequence

from app.NeuralSearch.models import get_llm


class ParaphaseMode(enum.Enum):
    EXPAND = 1
    SIMPLIFY = 0


SYSTEM_PROMPT = """
Ты улучшаешь поисковые запросы для нейропоиска по интернет-источникам.
Сохраняй исходный смысл запроса. Не выдумывай факты.
Отвечай только списком готовых поисковых запросов, без пояснений и без кавычек.
""".strip()


def _history_block(history: Sequence[str] | None) -> str:
    if not history:
        return ""
    recent = list(history)[-6:]
    return "\nКонтекст диалога:\n" + "\n".join(f"- {item}" for item in recent)


def paraphrase_query(
    query: str,
    history: Sequence[str] | None = None,
    mode: ParaphaseMode = ParaphaseMode.SIMPLIFY,
) -> List[str]:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string")

    llm = get_llm()
    variants_count = 5 if mode == ParaphaseMode.EXPAND else 1

    prompt = f"""
Исходный запрос: {query.strip()}
{_history_block(history)}

Сгенерируй {variants_count} поисковых формулировок на русском языке.
Требования:
- каждая формулировка пригодна для веб-поиска;
- не меняй смысл запроса;
- убери лишнюю разговорность;
- если запрос слишком краткий, мягко уточни формулировку, не искажая смысл.
- каждая формулировка должна быть на отдельной строке.
""".strip()

    raw = llm.complete_text(prompt=prompt, system_prompt=SYSTEM_PROMPT, temperature=0.1)
    lines = [line.strip(" -*	") for line in raw.splitlines() if line.strip()]
    paraphrases = [re.sub(r"^\s*\d+[\.)\-]\s*", "", line).strip() for line in lines]
    paraphrases = [p for p in paraphrases if p and len(p) > 2]

    if not paraphrases:
        return [query.strip()]
    return paraphrases[:variants_count]
