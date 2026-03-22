from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Callable

from langchain.chat_models import init_chat_model

MODEL_AGENT = os.getenv("MODEL_AGENT", "openai/gpt-5.4-nano-thinking-xhigh")
MODEL_PROVIDER_AGENT = os.getenv("MODEL_PROVIDER_AGENT", "openai")
VSEGPT_API_KEY = os.getenv("VSEGPT_API_KEY", "")
BASE_URL = os.getenv("URL", "https://api.vsegpt.ru/v1")
AGENT_TIMEOUT_SECONDS = float(os.getenv("AGENT_TIMEOUT_SECONDS", "60"))
PREFER_SEARCH_BACKEND = os.getenv("PREFER_SEARCH_BACKEND", "auto").lower()

# --------------------------------------------------------------------
# ВАЖНО:
# Ниже предполагается, что retrieval уже реализован у тебя в проекте.
# Ты писал, что "RAG/NeuroSearch уже реализованы и импортированы в этом файле".
#
# Поэтому здесь оставлен блок импорта с ПРИМЕРНЫМИ именами.
# Подставь реальные функции проекта:
# - rag_search(query: str) -> str
# - neuro_search(query: str) -> str
#
# Если у тебя есть только одна функция, оставь только её.
# --------------------------------------------------------------------
try:
    from src.retrieval.rag import rag_search  # type: ignore
except Exception:
    rag_search = None  # type: ignore

try:
    from src.retrieval.neuro_search import neuro_search  # type: ignore
except Exception:
    neuro_search = None  # type: ignore


class LiteratureRAGAgent:
    """
    Агент для поиска и ответа по литературе / справочным источникам.

    Задачи агента:
    - взять запрос из state["literature_query"] или state["task"];
    - вызвать retrieval (RAG / NeuroSearch);
    - получить строковый контекст;
    - на основе контекста сформировать итоговый grounded-ответ;
    - вернуть результат в history и literature_result.

    Ожидаемые входы из state:
    - literature_query: str (приоритетный поисковый запрос)
    - task: str (fallback, если literature_query не задан)

    Возвращаемые поля:
    - history: список из одного нового события агента
    - literature_result: словарь с answer / context / query / backend
    """

    def __init__(
            self,
            model: str = MODEL_AGENT,
            temperature: float = 0.0,
            prefer_backend: str = PREFER_SEARCH_BACKEND,
    ) -> None:
        if not VSEGPT_API_KEY:
            raise ValueError("VSEGPT_API_KEY не задан в окружении.")

        self.model = init_chat_model(
            model=model,
            model_provider=MODEL_PROVIDER_AGENT,
            api_key=VSEGPT_API_KEY,
            base_url=BASE_URL,
            temperature=temperature,
            timeout=AGENT_TIMEOUT_SECONDS,
        )
        self.prefer_backend = prefer_backend

    def _pick_query(self, state: Dict[str, Any]) -> str:
        query = str(state.get("literature_query") or "").strip()
        if query:
            return query

        query = str(state.get("task") or "").strip()
        if query:
            return query

        raise ValueError(
            "Для LiteratureRAGAgent не найден текст запроса: "
            "ожидается `literature_query` или `task`."
        )

    def _pick_backend_callable(self) -> tuple[str, Callable[[str], str]]:
        """
        Выбирает retrieval backend.

        Логика:
        - if prefer_backend == "rag" -> rag_search
        - if prefer_backend == "neurosearch" -> neuro_search
        - if prefer_backend == "auto" -> сначала rag_search, потом neuro_search
        """
        backend = self.prefer_backend

        if backend == "rag":
            if rag_search is None:
                raise ValueError("prefer_backend=rag, но функция rag_search недоступна.")
            return "rag", rag_search

        if backend == "neurosearch":
            if neuro_search is None:
                raise ValueError(
                    "prefer_backend=neurosearch, но функция neuro_search недоступна."
                )
            return "neurosearch", neuro_search

        # auto
        if rag_search is not None:
            return "rag", rag_search
        if neuro_search is not None:
            return "neurosearch", neuro_search

        raise ValueError(
            "Не найден retrieval backend: ни rag_search, ни neuro_search недоступны."
        )

    def _retrieve_context(self, query: str) -> Dict[str, Any]:
        backend_name, backend_callable = self._pick_backend_callable()
        raw_context = backend_callable(query)

        if raw_context is None:
            raw_context = ""

        context = str(raw_context).strip()

        return {
            "backend": backend_name,
            "query": query,
            "context": context,
        }

    def _build_system_prompt(self) -> str:
        return """Ты — scientific literature assistant для химического мультиагентного пайплайна.

Твоя задача:
- Отвечать только на основе переданного retrieval-контекста.
- Давать точный, сжатый, профессиональный ответ.
- Если контекст неполный, явно указывать ограничения.
- Не выдумывать факты, которых нет в контексте.
- Если в контексте мало данных, скажи об этом прямо.

Требования к ответу:
- Верни только валидный JSON.
- Без markdown и без поясняющего текста вокруг JSON.
- Формат:
{
  "answer": "<итоговый ответ для пользователя>",
  "sources": [],
  "confidence": "<low|medium|high>",
  "limitations": "<краткое описание ограничений или пустая строка>",
  "prediction": "<краткая ключевая формулировка результата>"
}

Правила:
- Поле answer должно быть полезным и завершённым.
- Если retrieval-контекст пустой или почти пустой, отрази это в answer и limitations.
- Если источники не выделены структурированно, верни пустой список в sources.
- prediction должен быть короткой выжимкой answer в 1 предложении.
"""

    def _build_user_prompt(self, query: str, context: str) -> str:
        return (
            f"Запрос пользователя:\n{query}\n\n"
            f"Retrieval-контекст:\n{context if context else '[пустой контекст]'}"
        )

    def _parse_llm_json(self, raw: Any) -> Dict[str, Any]:
        """
        Пытается извлечь dict из ответа модели.
        Поддерживает частый формат VseGPTWrapper/LLM:
        - dict
        - {'data': {...}}
        - строка JSON
        """
        if isinstance(raw, dict):
            payload = raw.get("data", raw)
            if isinstance(payload, dict):
                return payload

        if isinstance(raw, str):
            raw = raw.strip()
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass

        return {
            "answer": "Не удалось корректно распарсить ответ literature-агента.",
            "sources": [],
            "confidence": "low",
            "limitations": "Ошибка парсинга ответа модели.",
            "prediction": "Не удалось корректно распарсить ответ.",
        }

    def _generate_answer(self, query: str, context: str) -> Dict[str, Any]:
        """
        Формирует grounded-ответ на основе retrieval-контекста.
        """
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(query, context)

        response = self.model.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )

        content = getattr(response, "content", response)
        parsed = self._parse_llm_json(content)

        answer = str(parsed.get("answer", "")).strip()
        prediction = str(parsed.get("prediction", "")).strip()
        confidence = str(parsed.get("confidence", "low")).strip() or "low"
        limitations = str(parsed.get("limitations", "")).strip()

        sources = parsed.get("sources", [])
        if not isinstance(sources, list):
            sources = []

        if not answer:
            if context:
                answer = (
                    "Контекст по запросу найден, но финальный ответ не удалось "
                    "сформировать в структурированном виде."
                )
            else:
                answer = "По запросу не найден релевантный retrieval-контекст."

        if not prediction:
            prediction = answer

        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "limitations": limitations,
            "prediction": prediction,
        }

    def _build_empty_result(self, query: str) -> Dict[str, Any]:
        return {
            "query": query,
            "backend": None,
            "context": "",
            "answer": (
                "Не удалось получить retrieval-контекст по запросу. "
                "Проверьте подключение RAG/NeuroSearch."
            ),
            "sources": [],
            "confidence": "low",
            "limitations": "Retrieval backend не вернул контекст.",
            "prediction": "Не удалось получить retrieval-контекст.",
        }

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        query = self._pick_query(state)

        retrieval_data = self._retrieve_context(query)
        context = retrieval_data["context"]
        backend = retrieval_data["backend"]

        if not context:
            result = self._build_empty_result(query)
            result["backend"] = backend
        else:
            llm_result = self._generate_answer(query, context)
            result = {
                "query": query,
                "backend": backend,
                "context": context,
                "answer": llm_result["answer"],
                "sources": llm_result["sources"],
                "confidence": llm_result["confidence"],
                "limitations": llm_result["limitations"],
                "prediction": llm_result["prediction"],
            }

        event = {
            "agent": "LiteratureRAGAgent",
            "input": query,
            "output": {
                "answer": result["answer"],
                "sources": result["sources"],
                "confidence": result["confidence"],
                "limitations": result["limitations"],
                "prediction": result["prediction"],
                "backend": result["backend"],
            },
        }

        return {
            "history": [event],
            "literature_result": result,
        }

    def as_node(self):
        def node(state: Dict[str, Any]) -> Dict[str, Any]:
            return self.run(state)

        return node
