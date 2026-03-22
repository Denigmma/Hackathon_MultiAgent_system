from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

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
    Агент для поиска, структурирования и выбора лучшего маршрута синтеза.

    Что делает:
    1. Ищет различные методики / маршруты синтеза.
    2. Возвращает по возможности >= 3 различных маршрута.
    3. Сохраняет структуру взаимодействия с агентом и инструментами.
    4. На втором шаге выбирает лучший маршрут по мнению модели
       (не только корректный, но и наиболее удобный / практичный).
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

        self.search_agent = create_agent(
            model=self.model,
            tools=[
                self._neural_tool(),
                self._rag_tool(),
            ],
            system_prompt=self._build_search_system_prompt(),
        )

    @staticmethod
    def _build_search_system_prompt() -> str:
        return (
            "Ты химический агент, специализирующийся на поиске методик и маршрутов синтеза.\n"
            "Твоя задача:\n"
            "1. Искать в доступных источниках различные методики или маршруты синтеза целевого соединения.\n"
            "2. Извлекать условия реакций и составлять структурированное описание каждой найденной методики.\n"
            "3. Возвращать не менее 3 различных методик/маршрутов, если они существуют.\n"
            "4. Если найдено меньше 3 методик, возвращать все найденные.\n"
            "5. Не выдумывать статьи, DOI, выходы, условия или реагенты.\n"
            "6. Возвращать только валидный JSON без markdown и без текста вне JSON.\n\n"

            "Определение различия методик:\n"
            "- Считай методики различными, если различаются ключевые реагенты, тип превращения, "
            "порядок стадий, катализатор, условия реакции или синтетический маршрут.\n"
            "- Не дублируй почти идентичные варианты как отдельные маршруты.\n\n"

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
            "- Не выбирай лучший протокол на этом шаге.\n"
            "- summary.key_differences должен описывать различия между маршрутами.\n"
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
    def _serialize_message(msg: Any) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "type": msg.__class__.__name__,
            "content": getattr(msg, "content", None),
        }

        additional_kwargs = getattr(msg, "additional_kwargs", None)
        if additional_kwargs:
            data["additional_kwargs"] = additional_kwargs

        name = getattr(msg, "name", None)
        if name:
            data["name"] = name

        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            data["tool_calls"] = tool_calls

        response_metadata = getattr(msg, "response_metadata", None)
        if response_metadata:
            data["response_metadata"] = response_metadata

        return data

    def _extract_interaction_trace(self, state: Any) -> List[Dict[str, Any]]:
        if isinstance(state, dict) and "messages" in state and isinstance(state["messages"], list):
            return [self._serialize_message(m) for m in state["messages"]]
        return []

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
            "6. Не выбирать лучший маршрут на этом шаге.\n"
            "7. Вернуть только JSON заданного формата.\n"
        )

        if context:
            prompt += "\nКонтекст:\n" + json.dumps(context, ensure_ascii=False, indent=2)

        return prompt

    @staticmethod
    def _build_selector_prompt(search_result: Dict[str, Any]) -> str:
        return (
                "Ты химический эксперт по выбору наиболее практичного маршрута синтеза.\n"
                "Тебе дан список найденных маршрутов синтеза. "
                "Выбери лучший маршрут не только по химической корректности, но и по общей практичности/удобству.\n\n"

                "Критерии выбора:\n"
                "1. Простота маршрута и операционная удобность.\n"
                "2. Меньшее число сложных стадий / меньше синтетической сложности.\n"
                "3. Более мягкие и реалистичные условия.\n"
                "4. Более доступные реагенты/катализаторы/растворители.\n"
                "5. Более высокий выход, если он указан.\n"
                "6. Более простой workup / purification.\n"
                "7. Лучшая масштабируемость.\n"
                "8. Более высокая надежность и практическая воспроизводимость.\n\n"

                "Если данных недостаточно, всё равно выбери лучший из доступных вариантов и явно укажи ограничения.\n"
                "Не выдумывай факты. Опирайся только на переданные маршруты.\n"
                "Верни только валидный JSON без markdown.\n\n"

                "Формат ответа:\n"
                "{"
                '"best_route": {'
                '"route_id": string | null, '
                '"reasoning": [string], '
                '"strengths": [string], '
                '"weaknesses": [string], '
                '"practicality_score": number | null, '
                '"confidence": string'
                "}, "
                '"ranking": ['
                "{"
                '"route_id": string, '
                '"rank": number, '
                '"score": number | null, '
                '"why": [string]'
                "}"
                "], "
                '"selection_warnings": [string]'
                "}\n\n"

                "Данные для анализа:\n"
                + json.dumps(search_result, ensure_ascii=False, indent=2)
        )

    def _select_best_protocol(self, parsed_result: Dict[str, Any]) -> Dict[str, Any]:
        protocols = parsed_result.get("protocols", [])
        if not protocols:
            return {
                "best_route": {
                    "route_id": None,
                    "reasoning": [],
                    "strengths": [],
                    "weaknesses": ["Не найдено ни одного маршрута для сравнения."],
                    "practicality_score": None,
                    "confidence": "low",
                },
                "ranking": [],
                "selection_warnings": ["Невозможно выбрать лучший маршрут: protocols пуст."]
            }

        prompt = self._build_selector_prompt(parsed_result)
        response = self.model.invoke(prompt)

        text = getattr(response, "content", str(response))
        parsed = self._safe_json_loads(text)

        if parsed is not None:
            return parsed

        # fallback: простой эвристический выбор, если второй LLM-ответ сломан
        best_route_id = None
        best_score = -10 ** 9
        ranking = []

        for idx, p in enumerate(protocols, start=1):
            score = 0.0

            outcome = p.get("outcome", {}) or {}
            reaction = p.get("reaction", {}) or {}

            y = outcome.get("yield_percent")
            if isinstance(y, (int, float)):
                score += float(y)

            catalysts = reaction.get("catalysts") or []
            purification = reaction.get("purification") or []
            reagents = reaction.get("reagents") or []

            score -= 5 * max(0, len(catalysts) - 1)
            score -= 2 * max(0, len(purification) - 1)
            score -= 0.5 * len(reagents)

            temp = reaction.get("temperature")
            if isinstance(temp, str):
                low_temp_markers = ["room", "rt", "25", "20", "ambient", "комнат"]
                if any(m in temp.lower() for m in low_temp_markers):
                    score += 5

            if score > best_score:
                best_score = score
                best_route_id = p.get("route_id")

            ranking.append(
                {
                    "route_id": p.get("route_id", f"route_{idx}"),
                    "rank": 0,
                    "score": round(score, 2),
                    "why": ["Fallback-оценка по выходу, простоте и мягкости условий."]
                }
            )

        ranking = sorted(ranking, key=lambda x: (x["score"] is None, -(x["score"] or -999999)))
        for rank_idx, item in enumerate(ranking, start=1):
            item["rank"] = rank_idx

        return {
            "best_route": {
                "route_id": best_route_id,
                "reasoning": [
                    "Использован fallback-режим, потому что JSON от шага выбора маршрута оказался невалидным."
                ],
                "strengths": ["Маршрут выбран по упрощённой практической эвристике."],
                "weaknesses": ["Оценка менее надёжна, чем полноценный LLM ranking."],
                "practicality_score": ranking[0]["score"] if ranking else None,
                "confidence": "medium",
            },
            "ranking": ranking,
            "selection_warnings": [
                "Шаг model-based ranking вернул невалидный JSON, использована fallback-эвристика."
            ],
        }

    @staticmethod
    def _build_invalid_json_result(task: str, raw: str, trace: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        return {
            "error": "invalid_json",
            "raw": raw,
            "target": {
                "name": "",
                "reaction_description": task,
                "desired_product": "",
            },
            "protocols": [],
            "best_route": {
                "route_id": None,
                "reasoning": [],
                "strengths": [],
                "weaknesses": ["Агент вернул невалидный JSON на этапе поиска маршрутов."],
                "practicality_score": None,
                "confidence": "low",
            },
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
            "interaction_trace": trace or [],
        }

    def run(
            self,
            task: str,
            context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        prompt = self._build_user_prompt(task, context=context)

        state = self.search_agent.invoke(
            {"messages": [{"role": "user", "content": prompt}]}
        )

        trace = self._extract_interaction_trace(state)
        raw_text = self._extract_output(state)
        parsed = self._safe_json_loads(raw_text)

        if parsed is None:
            return self._build_invalid_json_result(task=task, raw=raw_text, trace=trace)

        selection = self._select_best_protocol(parsed)

        parsed["best_route"] = selection.get("best_route", {})
        parsed["ranking"] = selection.get("ranking", [])
        parsed["selection_warnings"] = selection.get("selection_warnings", [])

        # сохраняем структуру взаимодействия
        parsed["interaction_trace"] = trace
        parsed["agent_meta"] = {
            "task": task,
            "context": context,
            "raw_output": raw_text,
        }

        return parsed

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

            # сохраняем отдельную структуру взаимодействия
            state.setdefault("agent_interactions", {})
            state["agent_interactions"]["SynthesisProtocolSearchAgent"] = {
                "input": task,
                "context": state.get("context"),
                "interaction_trace": result.get("interaction_trace", []),
                "best_route": result.get("best_route"),
                "ranking": result.get("ranking", []),
            }

            state.setdefault("history", []).append(
                {
                    "agent": "SynthesisProtocolSearchAgent",
                    "input": task,
                    "output": result,
                }
            )
            return state

        return node
