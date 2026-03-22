from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool

from src.NeuralSearch.main import main as neural_search_main
# from src.RAG.main import main as rag_search_main

MODEL_AGENT = os.getenv("MODEL_AGENT", "openai/gpt-5.4-nano")
MODEL_PROVIDER_AGENT = os.getenv("MODEL_PROVIDER_AGENT", "openai")
VSEGPT_API_KEY = os.getenv("VSEGPT_API_KEY", "")
BASE_URL = os.getenv("URL", "https://api.vsegpt.ru/v1")


class SynthesisProtocolSearchAgent:
    """
    Агент для поиска и структурирования методик синтеза.

    Задачи:
    - искать протоколы и маршруты синтеза в статьях, базах и RAG-корпусе;
    - извлекать условия реакций;
    - возвращать как минимум 3 различных методики/маршрута, если они существуют;
    - если найдено меньше 3, возвращать все найденные;
    - НЕ выбирать лучший маршрут и НЕ давать финальную рекомендацию.
    """

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
                self._rag_tool(),
            ],
            system_prompt=self._build_system_prompt(),
        )

    @staticmethod
    def _build_system_prompt() -> str:
        return (
            "Ты химический агент, специализирующийся на поиске методик и маршрутов синтеза.\n"
            "Твоя задача:\n"
            "1. Искать в доступных источниках различные методики или маршруты синтеза целевого соединения.\n"
            "2. Извлекать условия реакций и составлять структурированное описание каждой найденной методики.\n"
            "3. Возвращать не менее 3 различных методик/маршрутов, если они существуют.\n"
            "4. Если найдено меньше 3 методик, возвращать все найденные.\n"
            "5. Не выбирать лучший маршрут, не рекомендовать один финальный путь, не делать окончательный отбор.\n"
            "6. Возвращать только валидный JSON без markdown и без текста вне JSON.\n\n"

            "Определение различия методик:\n"
            "- Считай методики различными, если различаются ключевые реагенты, тип превращения, "
            "порядок стадий, катализатор, условия реакции или синтетический маршрут.\n"
            "- Не дублируй почти идентичные варианты как отдельные маршруты.\n\n"

            "Не выдумывай статьи, DOI, выходы, условия или реагенты.\n"
            "Если информация не найдена, ставь null, пустой список или указывай это в notes/warnings.\n\n"

            "Формат ответа:\n"
            "{"
            '"target": {'
            '"name": string, '
            '"reaction_description": string, '
            '"desired_product": string'
            "}, "
            '"protocols": ['
            "{"
            '"route_id": string, '
            '"route_type": string, '
            '"source": {'
            '"title": string, '
            '"authors": [string], '
            '"year": number | null, '
            '"journal": string | null, '
            '"doi": string | null, '
            '"url": string | null'
            "}, "
            '"reaction": {'
            '"starting_materials": [string], '
            '"reagents": [string], '
            '"catalysts": [string], '
            '"solvents": [string], '
            '"temperature": string | null, '
            '"time": string | null, '
            '"atmosphere": string | null, '
            '"workup": [string], '
            '"purification": [string]'
            "}, "
            '"outcome": {'
            '"yield_percent": number | null, '
            '"selectivity": string | null, '
            '"scale": string | null'
            "}, "
            '"notes": [string], '
            '"confidence": string'
            "}"
            "], "
            '"summary": {'
            '"route_count_found": number, '
            '"returned_route_count": number, '
            '"minimum_target_route_count": number, '
            '"enough_routes_found": boolean, '
            '"key_differences": [string], '
            '"coverage_note": string | null'
            "}, "
            '"warnings": [string]'
            "}\n\n"

            "Правила:\n"
            "- Стремись вернуть как минимум 3 различных маршрута/методики, если они доступны.\n"
            "- Если найдено меньше 3, верни все найденные.\n"
            "- Не добавляй recommendation, best route, most practical route или аналогичные поля.\n"
            "- Не выбирай лучший протокол.\n"
            "- summary.key_differences должен описывать различия между маршрутами, а не советовать один из них.\n"
            "- Всегда возвращай только JSON."
        )

    def _neural_tool(self):
        @tool("neural_search")
        def neural_search(query: str) -> str:
            """Широкий поиск по статьям, базам и индексам для поиска методик синтеза."""
            try:
                result = neural_search_main(query)
                if isinstance(result, (dict, list)):
                    return json.dumps(result, ensure_ascii=False)
                return str(result)
            except Exception as e:
                return f"NEURAL_ERROR: {e}"

        return neural_search

    # def _rag_tool(self):
    #     @tool("rag_search")
    #     def rag_search(query: str) -> str:
    #         """Поиск по локальному RAG-корпусу: статьям, PDF, заметкам, базе методик."""
    #         try:
    #             result = rag_search_main(query)
    #             if isinstance(result, (dict, list)):
    #                 return json.dumps(result, ensure_ascii=False)
    #             return str(result)
    #         except Exception as e:
    #             return f"RAG_ERROR: {e}"
    #
    #     return rag_search

    @staticmethod
    def _extract_output(state: Any) -> str:
        if isinstance(state, str):
            return state

        if isinstance(state, dict):
            if "output" in state:
                return str(state["output"])
            if "messages" in state and state["messages"]:
                last = state["messages"][-1]
                return getattr(last, "content", str(last))

        return str(state)

    @staticmethod
    def _safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(text)
        except Exception:
            return None

    @staticmethod
    def _build_user_prompt(task: str, context: Optional[Dict[str, Any]] = None) -> str:
        prompt = (
            "Найди и структурируй различные методики или маршруты синтеза по следующему запросу.\n\n"
            f"Запрос:\n{task.strip()}\n\n"
            "Что нужно сделать:\n"
            "1. Найти различные методики/маршруты синтеза.\n"
            "2. Извлечь условия реакции: исходные вещества, реагенты, катализаторы, растворители, температуру, время, атмосферу.\n"
            "3. Извлечь результат: выход, селективность, масштаб.\n"
            "4. Вернуть не менее 3 различных маршрутов, если они существуют.\n"
            "5. Если найдено меньше 3, вернуть все найденные.\n"
            "6. Не выбирать лучший маршрут и не давать финальную рекомендацию.\n"
            "7. Вернуть только JSON заданного формата.\n"
        )

        if context:
            prompt += "\nКонтекст:\n" + json.dumps(context, ensure_ascii=False, indent=2)

        return prompt

    def run(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        prompt = self._build_user_prompt(task, context=context)

        state = self.agent.invoke(
            {"messages": [{"role": "user", "content": prompt}]}
        )

        text = self._extract_output(state)
        parsed = self._safe_json_loads(text)

        if parsed is not None:
            return parsed

        return {
            "error": "invalid_json",
            "raw": text,
            "target": {
                "name": "",
                "reaction_description": task,
                "desired_product": "",
            },
            "protocols": [],
            "summary": {
                "route_count_found": 0,
                "returned_route_count": 0,
                "minimum_target_route_count": 3,
                "enough_routes_found": False,
                "key_differences": [],
                "coverage_note": "Агент вернул невалидный JSON.",
            },
            "warnings": [
                "Агент вернул невалидный JSON.",
                "Проверь format instructions или добавь post-validation.",
            ],
        }

    def as_tool(self):
        agent_self = self

        @tool("synthesis_protocol_search")
        def synthesis_protocol_search(task: str) -> dict:
            return agent_self.run(task)

        return synthesis_protocol_search

    def as_node(self):
        agent_self = self

        def node(state: Dict[str, Any]) -> Dict[str, Any]:
            task = (
                state.get("task")
                or state.get("synthesis_task")
                or state.get("protocol_search_task")
                or state.get("reaction_task")
                or ""
            )

            if not isinstance(task, str):
                task = json.dumps(task, ensure_ascii=False)

            result = agent_self.run(task, context=state.get("context"))

            state["synthesis_protocol_result"] = result
            state.setdefault("history", []).append(
                {
                    "agent": "SynthesisProtocolSearchAgent",
                    "input": task,
                    "output": result,
                }
            )
            return state

        return node