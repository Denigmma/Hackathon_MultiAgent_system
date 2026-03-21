from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional
import os

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool

MODEL_AGENT = str(os.getenv("MODEL_AGENT"))


class SeparationMethodsAgent:
    """
    Агент для выбора методов разделения смеси.

    Вход:
        - одна строка с задачей;
        - опционально context как dict.

    Выход:
        - обычный dict, похожий на JSON.
    """

    def __init__(
        self,
        model: str = MODEL_AGENT,
        temperature: float = 0.0,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.model = init_chat_model(model, temperature=temperature)
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
        if agent_state.get("structured_response") is not None:
            sr = agent_state["structured_response"]
            if isinstance(sr, str):
                return sr
            try:
                return json.dumps(sr, ensure_ascii=False)
            except Exception:
                return str(sr)

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
        text = (text or "").strip()

        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) >= 3 and lines[-1].strip().startswith("```"):
                text = "\n".join(lines[1:-1]).strip()

        try:
            data = json.loads(text)
            return (
                data
                if isinstance(data, dict)
                else {"error": "LLM returned non-dict JSON", "raw": data}
            )
        except Exception:
            m = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if m:
                try:
                    data = json.loads(m.group(0))
                    return (
                        data
                        if isinstance(data, dict)
                        else {"error": "LLM returned non-dict JSON", "raw": data}
                    )
                except Exception:
                    pass
            return {"error": "Invalid JSON from LLM", "raw": text}

    def run(
        self, task: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Основной метод.

        Параметры:
            task: строка с задачей для LLM.
            context: необязательный dict с дополнительными условиями.

        Возвращает:
            dict с JSON-подобным результатом.
        """
        prompt = task.strip()
        if context:
            prompt += "\n\nКонтекст:\n" + json.dumps(
                context, ensure_ascii=False, indent=2
            )

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

        if not isinstance(parsed, dict):
            parsed = {"error": "Unexpected output", "raw": parsed}

        if "warnings" not in parsed:
            parsed["warnings"] = []

        return parsed

    def as_tool(self):
        """
        Возвращает LangChain tool.
        Tool принимает одну строку с задачей.
        """

        agent_self = self

        @tool("separation_methods")
        def separation_methods(task: str) -> dict:
            return agent_self.run(task)

        return separation_methods

    def as_node(self):
        """
        Возвращает callable-узел для графов и пайплайнов.
        Ожидает state с ключом task или separation_task.
        """

        agent_self = self

        def node(state: Dict[str, Any]) -> Dict[str, Any]:
            task = state.get("task") or state.get("separation_task") or ""
            if not isinstance(task, str):
                task = json.dumps(task, ensure_ascii=False)

            result = agent_self.run(task, context=state.get("context"))

            state["separation_result"] = result
            state.setdefault("history", []).append(
                {
                    "agent": "separation_methods_agent",
                    "input": task,
                    "output": result,
                }
            )
            return state

        return node
