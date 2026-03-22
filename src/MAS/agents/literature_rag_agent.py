from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, Optional

from langchain.chat_models import init_chat_model

MODEL_AGENT = os.getenv("MODEL_AGENT", "openai/gpt-5.4-nano-thinking-xhigh")
MODEL_PROVIDER_AGENT = os.getenv("MODEL_PROVIDER_AGENT", "openai")
VSEGPT_API_KEY = os.getenv("VSEGPT_API_KEY", "")
BASE_URL = os.getenv("URL", "https://api.vsegpt.ru/v1")
AGENT_TIMEOUT_SECONDS = float(os.getenv("AGENT_TIMEOUT_SECONDS", "60"))
PREFER_SEARCH_BACKEND = os.getenv("PREFER_SEARCH_BACKEND", "auto").lower()
MAX_CONTEXT_PREVIEW_CHARS = int(os.getenv("LITERATURE_CONTEXT_PREVIEW_CHARS", "1500"))

try:
    from src.RAG.rag_main import main as answer_query
except Exception:
    answer_query = None

try:
    from src.NeuralSearch.main import main as neural_search_main  # type: ignore
except Exception:
    neural_search_main = None  # type: ignore


class LiteratureRAGAgent:
    """Агент ответа по литературе и retrieval-источникам."""

    def __init__(
        self,
        model: str = MODEL_AGENT,
        temperature: float = 0.0,
        prefer_backend: str = PREFER_SEARCH_BACKEND,
    ) -> None:
        self.prefer_backend = (prefer_backend or "auto").lower()
        self.model = None

        if VSEGPT_API_KEY:
            self.model = init_chat_model(
                model=model,
                model_provider=MODEL_PROVIDER_AGENT,
                api_key=VSEGPT_API_KEY,
                base_url=BASE_URL,
                temperature=temperature,
                timeout=AGENT_TIMEOUT_SECONDS,
            )

    def _pick_query(self, state: Dict[str, Any]) -> str:
        query = str(state.get("literature_query") or "").strip()
        if query:
            return query

        query = str(state.get("task") or "").strip()
        if query:
            return query

        raise ValueError(
            "Для LiteratureRAGAgent не найден текст запроса: ожидается `literature_query` или `task`."
        )

    def _pick_backend_callable(self) -> tuple[Optional[str], Optional[Callable[[str], Any]]]:
        backend = self.prefer_backend

        if backend == "rag":
            return ("rag", answer_query) if answer_query is not None else (None, None)

        if backend in {"neurosearch", "neural", "neuralsearch"}:
            return (
                ("neurosearch", neural_search_main)
                if neural_search_main is not None
                else (None, None)
            )

        if answer_query is not None:
            return "rag", answer_query
        if neural_search_main is not None:
            return "neurosearch", neural_search_main
        return None, None

    @staticmethod
    def _normalize_context(raw_context: Any) -> str:
        if raw_context is None:
            return ""
        if isinstance(raw_context, str):
            return raw_context.strip()
        try:
            return json.dumps(raw_context, ensure_ascii=False, indent=2, default=str).strip()
        except Exception:
            return str(raw_context).strip()

    def _retrieve_context(self, query: str) -> Dict[str, Any]:
        backend_name, backend_callable = self._pick_backend_callable()
        if backend_callable is None:
            return {
                "backend": None,
                "query": query,
                "context": "",
                "error": "Не найден доступный retrieval backend.",
            }

        try:
            raw_context = backend_callable(query)
        except Exception as exc:
            return {
                "backend": backend_name,
                "query": query,
                "context": "",
                "error": f"Ошибка retrieval backend: {exc}",
            }

        return {
            "backend": backend_name,
            "query": query,
            "context": self._normalize_context(raw_context),
            "error": "",
        }

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        text = (text or "").strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) >= 3 and lines[-1].strip().startswith("```"):
                return "\n".join(lines[1:-1]).strip()
        return text

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
"""

    def _build_user_prompt(self, query: str, context: str) -> str:
        return (
            f"Запрос пользователя:\n{query}\n\n"
            f"Retrieval-контекст:\n{context if context else '[пустой контекст]'}"
        )

    def _parse_llm_json(self, raw: Any) -> Dict[str, Any]:
        if isinstance(raw, dict):
            payload = raw.get("data", raw)
            if isinstance(payload, dict):
                return payload

        text = self._strip_code_fences(str(raw))
        try:
            parsed = json.loads(text)
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
        if self.model is None:
            preview = context[:MAX_CONTEXT_PREVIEW_CHARS]
            limitations = (
                "LLM недоступна, поэтому возвращён упрощённый ответ по retrieval-контексту."
            )
            answer = (
                "Найден retrieval-контекст по запросу. Ниже приведён фрагмент контекста для ручной интерпретации: "
                f"{preview}"
            )
            return {
                "answer": answer,
                "sources": [],
                "confidence": "low",
                "limitations": limitations,
                "prediction": "Найден контекст, но LLM-суммаризация недоступна.",
            }

        response = self.model.invoke(
            [
                {"role": "system", "content": self._build_system_prompt()},
                {"role": "user", "content": self._build_user_prompt(query, context)},
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
            answer = "Контекст найден, но финальный ответ не удалось сформировать в структурированном виде."
        if not prediction:
            prediction = answer

        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "limitations": limitations,
            "prediction": prediction,
        }

    def _build_empty_result(self, query: str, backend: Optional[str], error: str) -> Dict[str, Any]:
        return {
            "query": query,
            "backend": backend,
            "context": "",
            "answer": "Не удалось получить retrieval-контекст по запросу.",
            "sources": [],
            "confidence": "low",
            "limitations": error or "Retrieval backend не вернул контекст.",
            "prediction": "Не удалось получить retrieval-контекст.",
        }

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        query = self._pick_query(state)
        retrieval_data = self._retrieve_context(query)
        context = retrieval_data.get("context", "")
        backend = retrieval_data.get("backend")
        error = str(retrieval_data.get("error") or "")

        if not context:
            result = self._build_empty_result(query, backend, error)
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
