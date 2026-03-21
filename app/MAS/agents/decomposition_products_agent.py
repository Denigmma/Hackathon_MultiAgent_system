"""
Агент выбора методов разделения смеси.

Агент использует LangChain-модель и возвращает структурированный JSON-ответ.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool

from app.NeuralSearch.main import main as neural_search_main
# from app.RAG.main import main as rag_main

MODEL_AGENT = os.getenv("MODEL_AGENT", "openai/gpt-5.4-nano-thinking-xhigh")
MODEL_PROVIDER_AGENT = os.getenv("MODEL_PROVIDER_AGENT", "openai")
VSEGPT_API_KEY = os.getenv("VSEGPT_API_KEY", "")
BASE_URL = os.getenv("URL", "https://api.vsegpt.ru/v1")
AGENT_TIMEOUT_SECONDS = float(os.getenv("AGENT_TIMEOUT_SECONDS", "60"))


class SeparationMethodsAgent:
    """Агент для подбора методов разделения смеси."""

    def __init__(
            self,
            model: str = MODEL_AGENT,
            temperature: float = 0.01,
            system_prompt: Optional[str] = None,
    ) -> None:
        """Инициализирует LLM-агента."""
        if not VSEGPT_API_KEY:
            raise ValueError("VSEGPT_API_KEY не задан в окружении.")

        self.model = init_chat_model(
            model,
            model_provider=MODEL_PROVIDER_AGENT,
            temperature=temperature,
            api_key=VSEGPT_API_KEY,
            base_url=BASE_URL,
            timeout=AGENT_TIMEOUT_SECONDS,
        )

        self.system_prompt = system_prompt or (
            "Ты химический ассистент. Получаешь задачу о разделении смеси как обычный текст. "
            "Нужно выбрать подходящие методы разделения. "
            "Возвращай ТОЛЬКО валидный JSON без markdown и без пояснений вокруг него. "
            "Формат ответа: "
            "{"
            '"target_name": string, '
            '"suggestions": ['
            "{"
            '"method": string, '
            '"score": number, '
            '"rationale": string, '
            '"prerequisites": [string], '
            '"limitations": [string], '
            '"confidence": string'
            "}"
            "], "
            '"warnings": [string]'
            "}"
        )

        self.agent = create_agent(
            model=self.model,
            tools=[],
            system_prompt=self.system_prompt,
            name="separation_methods_agent",
        )

    @staticmethod
    def _extract_content(agent_state: Any) -> str:
        """Извлекает текст ответа из разных форматов состояния агента."""
        if isinstance(agent_state, str):
            return agent_state

        if hasattr(agent_state, "content"):
            return str(getattr(agent_state, "content", ""))

        if not isinstance(agent_state, dict):
            return str(agent_state)

        if "output" in agent_state:
            output = agent_state.get("output")
            return output if isinstance(output, str) else str(output)

        structured = agent_state.get("structured_response")
        if structured is not None:
            return structured if isinstance(structured, str) else json.dumps(structured, ensure_ascii=False)

        messages = agent_state.get("messages") or []
        if not messages:
            return ""

        last = messages[-1]
        content = getattr(last, "content", last if isinstance(last, str) else "")
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    parts.append(str(block["text"]))
                else:
                    parts.append(str(block))
            return "".join(parts)

        return str(content)

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        """Пытается распарсить JSON, включая fallback из текста."""
        text = (text or "").strip()

        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) >= 3 and lines[-1].strip().startswith("```"):
                text = "\n".join(lines[1:-1]).strip()

        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
            return {"error": "LLM returned non-dict JSON", "raw": data}
        except Exception:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(0))
                    if isinstance(data, dict):
                        return data
                    return {"error": "LLM returned non-dict JSON", "raw": data}
                except Exception:
                    pass
            return {"error": "Invalid JSON from LLM", "raw": text}

    @staticmethod
    def _normalize_text(data: Any) -> str:
        """Приводит ответ поиска к строке."""
        if data is None:
            return ""
        if isinstance(data, str):
            return data.strip()
        if isinstance(data, dict):
            return json.dumps(data, ensure_ascii=False, indent=2)
        if isinstance(data, list):
            parts = []
            for item in data:
                parts.append(SeparationMethodsAgent._normalize_text(item))
            return "\n".join([p for p in parts if p.strip()]).strip()
        return str(data).strip()

    @staticmethod
    def _looks_useful(text: str) -> bool:
        """
        Определяет, есть ли в результате что-то полезное.
        Здесь используется мягкая эвристика: не пусто и не технический мусор.
        """
        text = (text or "").strip()
        if not text:
            return False

        low = text.lower()

        bad_markers = [
            "not found",
            "no results",
            "empty",
            "error",
            "exception",
            "traceback",
            "null",
            "none",
            "[]",
            "{}",
        ]
        if any(marker in low for marker in bad_markers):
            return False

        # Должно быть хотя бы немного содержательного текста
        return len(text) >= 30

    def _search_with_rag(self, task: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Ищет релевантный контекст через RAG."""
        payload = {
            "task": task,
            "context": context or {},
        }

        try:
            result = rag_main(payload)
        except TypeError:
            # На случай, если main принимает просто строку
            try:
                result = rag_main(task)
            except Exception as exc:
                return f"RAG_ERROR: {exc}"
        except Exception as exc:
            return f"RAG_ERROR: {exc}"

        return self._normalize_text(result)

    def _search_with_neural(self, task: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Ищет релевантный контекст через NeuralSearch."""
        payload = {
            "task": task,
            "context": context or {},
        }

        try:
            result = neural_search_main(payload)
        except TypeError:
            try:
                result = neural_search_main(task)
            except Exception as exc:
                return f"NEURAL_SEARCH_ERROR: {exc}"
        except Exception as exc:
            return f"NEURAL_SEARCH_ERROR: {exc}"

        return self._normalize_text(result)

    def _build_augmented_prompt(self, task: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Сначала ищет через RAG.
        Если RAG не дал полезного результата — использует NeuralSearch.
        """
        rag_result = self._search_with_rag(task, context=context)

        if self._looks_useful(rag_result):
            source_block = f"Релевантный контекст из RAG:\n{rag_result}"
        else:
            neural_result = self._search_with_neural(task, context=context)
            if self._looks_useful(neural_result):
                source_block = f"Релевантный контекст из NeuralSearch:\n{neural_result}"
            else:
                source_block = (
                    "Релевантный контекст не найден ни через RAG, ни через NeuralSearch."
                )

        prompt = task.strip()

        if context:
            prompt += "\n\nДополнительный контекст:\n" + json.dumps(context, ensure_ascii=False, indent=2)

        prompt += "\n\n" + source_block
        return prompt

    def run(
            self,
            task: str,
            context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Запускает агента и возвращает JSON-совместимый dict."""
        prompt = self._build_augmented_prompt(task, context=context)

        state = self.agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ]
            }
        )

        raw = self._extract_content(state)
        parsed = self._parse_json(raw)

        if "warnings" not in parsed:
            parsed["warnings"] = []

        return parsed

    def as_tool(self):
        """Возвращает агент как LangChain tool."""

        agent_self = self

        @tool("separation_methods")
        def separation_methods(task: str) -> dict:
            return agent_self.run(task)

        return separation_methods

    def as_node(self):
        """Возвращает callable-узел для LangGraph."""

        agent_self = self

        def node(state: Dict[str, Any]) -> Dict[str, Any]:
            task = state.get("task") or state.get("separation_task") or ""
            if not isinstance(task, str):
                task = json.dumps(task, ensure_ascii=False)

            result = agent_self.run(task, context=state.get("context"))

            state["separation_result"] = result
            state.setdefault("history", []).append(
                {
                    "agent": "SeparationMethodsAgent",
                    "input": task,
                    "output": result,
                }
            )
            return state

        return node
