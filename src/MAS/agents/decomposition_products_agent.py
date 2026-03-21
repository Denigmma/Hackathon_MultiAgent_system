from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool

from app.NeuralSearch.main import main as neural_search_main

MODEL_AGENT = os.getenv("MODEL_AGENT", "openai/gpt-5.4-nano")
MODEL_PROVIDER_AGENT = os.getenv("MODEL_PROVIDER_AGENT", "openai")
VSEGPT_API_KEY = os.getenv("VSEGPT_API_KEY", "")
BASE_URL = os.getenv("URL", "https://api.vsegpt.ru/v1")


class SeparationMethodsAgent:
    def __init__(self, temperature: float = 0.1):
        if not VSEGPT_API_KEY:
            raise ValueError("VSEGPT_API_KEY не задан")

        self.model = init_chat_model(
            MODEL_AGENT,
            model_provider=MODEL_PROVIDER_AGENT,
            temperature=temperature,
            api_key=VSEGPT_API_KEY,
            base_url=BASE_URL,
        )

        self.agent = create_agent(
            model=self.model,
            tools=[
                self._neural_tool(),
                self._eval_tool(),
            ],
            system_prompt=(
                "Ты химический ассистент.\n"
                "Решаешь задачи разделения смесей.\n"
                "Используй инструменты при необходимости.\n"
                "Всегда возвращай ТОЛЬКО JSON формата:\n"
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
            ),
        )

    def _neural_tool(self):
        @tool("neural_search")
        def neural_search(query: str) -> str:
            """Поиск через NeuralSearch"""
            try:
                return str(neural_search_main(query))
            except Exception as e:
                return f"NEURAL_ERROR: {e}"

        return neural_search

    def _eval_tool(self):
        @tool("evaluate_expression")
        def evaluate_expression(expr: str) -> str:
            """Вычисляет математическое выражение"""
            try:
                return str(eval(expr, {"__builtins__": {}}))
            except Exception as e:
                return f"EVAL_ERROR: {e}"

        return evaluate_expression

    @staticmethod
    def _extract_output(state: Any) -> str:
        """Унифицированное извлечение текста ответа агента."""
        if isinstance(state, str):
            return state

        if isinstance(state, dict):
            if "output" in state:
                return str(state["output"])
            if "messages" in state and state["messages"]:
                last = state["messages"][-1]
                return getattr(last, "content", str(last))

        return str(state)

    def run(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        prompt = task.strip()

        if context:
            prompt += "\n\nКонтекст:\n" + json.dumps(context, ensure_ascii=False, indent=2)

        state = self.agent.invoke(
            {"messages": [{"role": "user", "content": prompt}]}
        )

        text = self._extract_output(state)

        try:
            return json.loads(text)
        except Exception:
            return {
                "error": "invalid_json",
                "raw": text,
            }

    def as_tool(self):
        agent_self = self

        @tool("separation_methods")
        def separation_methods(task: str) -> dict:
            return agent_self.run(task)

        return separation_methods

    def as_node(self):
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
