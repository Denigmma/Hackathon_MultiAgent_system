"""Агент выбора методов разделения смеси.

Важно: агент инициализирует модель через `init_chat_model` с явным `model_provider`.
Это устраняет ошибку вида:
"Unable to infer model provider for model='openai/...'."
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool


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
        temperature: float = 0.0,
        system_prompt: Optional[str] = None,
    ) -> None:
        """Инициализирует LLM-агента.

        Args:
            model: Имя модели.
            temperature: Температура генерации.
            system_prompt: Системный промпт (можно переопределить).
        """
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
    def _extract_content(agent_state: Dict[str, Any]) -> str:
        """Извлекает текст ответа из состояния LangChain-агента."""
        if agent_state.get("structured_response") is not None:
            structured_response = agent_state["structured_response"]
            if isinstance(structured_response, str):
                return structured_response
            try:
                return json.dumps(structured_response, ensure_ascii=False)
            except Exception:
                return str(structured_response)

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
        """Пытается распарсить JSON, включая fallback для fenced-блоков."""
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

    def run(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Запускает агента и возвращает JSON-совместимый dict."""
        prompt = task.strip()
        if context:
            prompt += "\n\nКонтекст:\n" + json.dumps(context, ensure_ascii=False, indent=2)

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
