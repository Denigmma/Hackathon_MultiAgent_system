from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool

try:
    from src.NeuralSearch.main import main as neural_search_main  # type: ignore
except Exception:
    neural_search_main = None  # type: ignore

try:
    from src.RAG.main import main as rag_search_main  # type: ignore
except Exception:
    rag_search_main = None  # type: ignore

MODEL_AGENT = os.getenv("MODEL_AGENT", "openai/gpt-5.4-nano")
MODEL_PROVIDER_AGENT = os.getenv("MODEL_PROVIDER_AGENT", "openai")
VSEGPT_API_KEY = os.getenv("VSEGPT_API_KEY", "")
BASE_URL = os.getenv("URL", "https://api.vsegpt.ru/v1")
AGENT_TIMEOUT_SECONDS = float(os.getenv("AGENT_TIMEOUT_SECONDS", "60"))
MIN_ROUTE_TARGET = int(os.getenv("MIN_SYNTHESIS_ROUTE_TARGET", "3"))


class SynthesisProtocolSearchAgent:
    """Агент для поиска, структурирования и выбора лучшего маршрута синтеза."""

    def __init__(self, temperature: float = 0.1):
        self.temperature = temperature
        self.model = None
        self.search_agent = None
        self.available_tools: List[str] = []

        if VSEGPT_API_KEY:
            self.model = init_chat_model(
                MODEL_AGENT,
                model_provider=MODEL_PROVIDER_AGENT,
                temperature=temperature,
                api_key=VSEGPT_API_KEY,
                base_url=BASE_URL,
                timeout=AGENT_TIMEOUT_SECONDS,
            )

        tools = self._build_tools()
        if self.model is not None and tools:
            self.search_agent = create_agent(
                model=self.model,
                tools=tools,
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
            "}"
        )

    def _build_tools(self):
        tools = []

        if neural_search_main is not None:
            @tool("neural_search")
            def neural_search(query: str) -> str:
                """Широкий поиск по статьям, базам и индексам для поиска методик синтеза."""
                try:
                    result = neural_search_main(query)
                    if isinstance(result, (dict, list)):
                        return json.dumps(result, ensure_ascii=False)
                    return str(result)
                except Exception as exc:  # pragma: no cover - external dependency
                    return f"NEURAL_ERROR: {exc}"

            tools.append(neural_search)
            self.available_tools.append("neural_search")

        if rag_search_main is not None:
            @tool("rag_search")
            def rag_search(query: str) -> str:
                """Поиск по локальному RAG-корпусу: статьям, PDF, заметкам, базе методик."""
                try:
                    result = rag_search_main(query)
                    if isinstance(result, (dict, list)):
                        return json.dumps(result, ensure_ascii=False)
                    return str(result)
                except Exception as exc:  # pragma: no cover - external dependency
                    return f"RAG_ERROR: {exc}"

            tools.append(rag_search)
            self.available_tools.append("rag_search")

        return tools

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        text = (text or "").strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) >= 3 and lines[-1].strip().startswith("```"):
                return "\n".join(lines[1:-1]).strip()
        return text

    @staticmethod
    def _extract_output(state: Any) -> str:
        if isinstance(state, str):
            return state

        if isinstance(state, dict):
            if "output" in state:
                return str(state["output"])
            messages = state.get("messages") or []
            if messages:
                last = messages[-1]
                content = getattr(last, "content", last)
                if isinstance(content, list):
                    parts = []
                    for block in content:
                        if isinstance(block, dict) and "text" in block:
                            parts.append(str(block["text"]))
                        else:
                            parts.append(str(block))
                    return "".join(parts)
                return str(content)

        if hasattr(state, "content"):
            return str(getattr(state, "content"))
        return str(state)

    @staticmethod
    def _safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
        text = SynthesisProtocolSearchAgent._strip_code_fences(text)
        try:
            value = json.loads(text)
            return value if isinstance(value, dict) else None
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
        if isinstance(state, dict) and isinstance(state.get("messages"), list):
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
            f"4. Вернуть не менее {MIN_ROUTE_TARGET} различных маршрутов, если они существуют.\n"
            "5. Если найдено меньше, вернуть все найденные.\n"
            "6. Не выбирать лучший маршрут на этом шаге.\n"
            "7. Вернуть только JSON заданного формата.\n"
        )

        if context:
            prompt += "\nКонтекст:\n" + json.dumps(context, ensure_ascii=False, indent=2, default=str)
        return prompt

    @staticmethod
    def _build_selector_prompt(search_result: Dict[str, Any]) -> str:
        return (
            "Ты химический эксперт по выбору наиболее практичного маршрута синтеза.\n"
            "Тебе дан список найденных маршрутов синтеза. Выбери лучший маршрут не только по химической корректности, но и по общей практичности/удобству.\n\n"
            "Критерии выбора:\n"
            "1. Простота маршрута и операционная удобность.\n"
            "2. Меньшее число сложных стадий и меньшая синтетическая сложность.\n"
            "3. Более мягкие и реалистичные условия.\n"
            "4. Более доступные реагенты, катализаторы и растворители.\n"
            "5. Более высокий выход, если он указан.\n"
            "6. Более простой workup и purification.\n"
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
            '"comment": string'
            "}"
            "]"
            "}\n\n"
            f"Маршруты:\n{json.dumps(search_result.get('protocols', []), ensure_ascii=False, indent=2, default=str)}"
        )

    @staticmethod
    def _route_complexity_penalty(protocol: Dict[str, Any]) -> float:
        reaction = protocol.get("reaction", {}) or {}
        starting = len(reaction.get("starting_materials", []) or [])
        reagents = len(reaction.get("reagents", []) or [])
        catalysts = len(reaction.get("catalysts", []) or [])
        purification = len(reaction.get("purification", []) or [])
        workup = len(reaction.get("workup", []) or [])
        return 0.8 * starting + 0.5 * reagents + 0.3 * catalysts + 0.4 * purification + 0.2 * workup

    @staticmethod
    def _condition_bonus(protocol: Dict[str, Any]) -> float:
        reaction = protocol.get("reaction", {}) or {}
        temp = str(reaction.get("temperature") or "").lower()
        atmosphere = str(reaction.get("atmosphere") or "").lower()
        score = 0.0
        if temp:
            if any(token in temp for token in ["room", "rt", "25", "20"]):
                score += 1.0
            if any(token in temp for token in ["reflux", "120", "150", "180"]):
                score -= 0.8
        if atmosphere:
            if atmosphere in {"air", "ambient", "open air"}:
                score += 0.5
            if any(token in atmosphere for token in ["argon", "nitrogen", "inert"]):
                score -= 0.3
        return score

    @classmethod
    def _heuristic_rank_protocols(cls, protocols: List[Dict[str, Any]]) -> Dict[str, Any]:
        ranking: List[Dict[str, Any]] = []
        scored: List[tuple[float, Dict[str, Any]]] = []

        for protocol in protocols:
            outcome = protocol.get("outcome", {}) or {}
            confidence = str(protocol.get("confidence") or "").lower()
            yield_percent = outcome.get("yield_percent")
            try:
                yield_score = float(yield_percent) / 10.0 if yield_percent is not None else 4.0
            except Exception:
                yield_score = 4.0

            confidence_bonus = {"high": 1.5, "medium": 0.8, "low": 0.2}.get(confidence, 0.0)
            score = yield_score + confidence_bonus + cls._condition_bonus(protocol) - cls._route_complexity_penalty(protocol)
            scored.append((score, protocol))

        scored.sort(key=lambda item: item[0], reverse=True)

        for idx, (score, protocol) in enumerate(scored, start=1):
            ranking.append(
                {
                    "route_id": protocol.get("route_id") or f"route_{idx}",
                    "rank": idx,
                    "score": round(score, 3),
                    "comment": "Эвристический ранг по выходу, простоте и мягкости условий.",
                }
            )

        if not scored:
            return {
                "best_route": {
                    "route_id": None,
                    "reasoning": ["Не удалось выделить ни одного маршрута для ранжирования."],
                    "strengths": [],
                    "weaknesses": ["Маршруты отсутствуют."],
                    "practicality_score": None,
                    "confidence": "low",
                },
                "ranking": [],
            }

        best_score, best_protocol = scored[0]
        best_route = {
            "route_id": best_protocol.get("route_id"),
            "reasoning": [
                "Маршрут выбран по эвристической оценке практичности.",
                "Учитывались выход, сложность набора реагентов и жёсткость условий.",
            ],
            "strengths": [
                f"Эвристический score={round(best_score, 3)}",
                "Маршрут занимает верхнюю позицию в ранжировании.",
            ],
            "weaknesses": [
                "Ранжирование получено без дополнительной LLM-оценки.",
            ],
            "practicality_score": round(best_score, 3),
            "confidence": "medium" if len(scored) > 1 else "low",
        }
        return {"best_route": best_route, "ranking": ranking}

    def _select_best_protocol(self, parsed_result: Dict[str, Any]) -> Dict[str, Any]:
        protocols = parsed_result.get("protocols", [])
        if not isinstance(protocols, list):
            protocols = []

        if not protocols:
            return self._heuristic_rank_protocols([])

        if self.model is None:
            return self._heuristic_rank_protocols(protocols)

        try:
            response = self.model.invoke(
                [
                    {"role": "system", "content": "Верни только JSON без markdown."},
                    {"role": "user", "content": self._build_selector_prompt(parsed_result)},
                ]
            )
            content = getattr(response, "content", response)
            parsed = self._safe_json_loads(str(content))
            if isinstance(parsed, dict) and "best_route" in parsed and "ranking" in parsed:
                return parsed
        except Exception:
            pass

        return self._heuristic_rank_protocols(protocols)

    @staticmethod
    def _normalize_protocol(protocol: Dict[str, Any], index: int) -> Dict[str, Any]:
        protocol = protocol if isinstance(protocol, dict) else {}
        source = protocol.get("source") if isinstance(protocol.get("source"), dict) else {}
        reaction = protocol.get("reaction") if isinstance(protocol.get("reaction"), dict) else {}
        outcome = protocol.get("outcome") if isinstance(protocol.get("outcome"), dict) else {}
        return {
            "route_id": protocol.get("route_id") or f"route_{index}",
            "route_type": protocol.get("route_type") or "unspecified",
            "source": {
                "title": source.get("title") or "",
                "authors": source.get("authors") if isinstance(source.get("authors"), list) else [],
                "year": source.get("year"),
                "journal": source.get("journal"),
                "doi": source.get("doi"),
                "url": source.get("url"),
            },
            "reaction": {
                "starting_materials": reaction.get("starting_materials") if isinstance(reaction.get("starting_materials"), list) else [],
                "reagents": reaction.get("reagents") if isinstance(reaction.get("reagents"), list) else [],
                "catalysts": reaction.get("catalysts") if isinstance(reaction.get("catalysts"), list) else [],
                "solvents": reaction.get("solvents") if isinstance(reaction.get("solvents"), list) else [],
                "temperature": reaction.get("temperature"),
                "time": reaction.get("time"),
                "atmosphere": reaction.get("atmosphere"),
                "workup": reaction.get("workup") if isinstance(reaction.get("workup"), list) else [],
                "purification": reaction.get("purification") if isinstance(reaction.get("purification"), list) else [],
            },
            "outcome": {
                "yield_percent": outcome.get("yield_percent"),
                "selectivity": outcome.get("selectivity"),
                "scale": outcome.get("scale"),
            },
            "notes": protocol.get("notes") if isinstance(protocol.get("notes"), list) else [],
            "confidence": protocol.get("confidence") or "low",
        }

    def _normalize_search_result(self, task: str, parsed_result: Dict[str, Any]) -> Dict[str, Any]:
        target = parsed_result.get("target") if isinstance(parsed_result.get("target"), dict) else {}
        raw_protocols = parsed_result.get("protocols") if isinstance(parsed_result.get("protocols"), list) else []
        protocols = [self._normalize_protocol(protocol, i) for i, protocol in enumerate(raw_protocols, start=1)]
        summary = parsed_result.get("summary") if isinstance(parsed_result.get("summary"), dict) else {}
        warnings = parsed_result.get("warnings") if isinstance(parsed_result.get("warnings"), list) else []

        route_count = len(protocols)
        normalized_summary = {
            "route_count_found": int(summary.get("route_count_found", route_count) or route_count),
            "returned_route_count": int(summary.get("returned_route_count", route_count) or route_count),
            "minimum_target_route_count": int(summary.get("minimum_target_route_count", MIN_ROUTE_TARGET) or MIN_ROUTE_TARGET),
            "enough_routes_found": bool(summary.get("enough_routes_found", route_count >= MIN_ROUTE_TARGET)),
            "key_differences": summary.get("key_differences") if isinstance(summary.get("key_differences"), list) else [],
            "coverage_note": summary.get("coverage_note") or None,
        }

        return {
            "target": {
                "name": target.get("name") or "",
                "reaction_description": target.get("reaction_description") or task,
                "desired_product": target.get("desired_product") or "",
            },
            "protocols": protocols,
            "summary": normalized_summary,
            "warnings": warnings,
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
            "summary": {
                "route_count_found": 0,
                "returned_route_count": 0,
                "minimum_target_route_count": MIN_ROUTE_TARGET,
                "enough_routes_found": False,
                "key_differences": [],
                "coverage_note": "Агент вернул невалидный JSON.",
            },
            "warnings": [
                "Агент вернул невалидный JSON.",
                "Проверь format instructions или добавь post-validation.",
            ],
            "interaction_trace": trace or [],
            "agent_meta": {
                "tools": [],
                "selection_mode": "none",
            },
        }

    @staticmethod
    def _build_unavailable_result(task: str, reason: str) -> Dict[str, Any]:
        return {
            "error": "search_unavailable",
            "raw": "",
            "target": {
                "name": "",
                "reaction_description": task,
                "desired_product": "",
            },
            "protocols": [],
            "summary": {
                "route_count_found": 0,
                "returned_route_count": 0,
                "minimum_target_route_count": MIN_ROUTE_TARGET,
                "enough_routes_found": False,
                "key_differences": [],
                "coverage_note": reason,
            },
            "warnings": [reason],
            "interaction_trace": [],
            "agent_meta": {
                "tools": [],
                "selection_mode": "none",
            },
        }

    def run(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        task = (task or "").strip()
        if not task:
            return self._build_unavailable_result("", "Пустой запрос для поиска методик синтеза.")

        if self.search_agent is None:
            reasons = []
            if self.model is None:
                reasons.append("LLM-модель не инициализирована")
            if not self.available_tools:
                reasons.append("не найден ни один retrieval backend")
            return self._build_unavailable_result(task, "; ".join(reasons) or "search_agent недоступен")

        prompt = self._build_user_prompt(task, context=context)
        state = self.search_agent.invoke({"messages": [{"role": "user", "content": prompt}]})
        text = self._extract_output(state)
        parsed = self._safe_json_loads(text)
        trace = self._extract_interaction_trace(state)

        if parsed is None:
            return self._build_invalid_json_result(task, text, trace=trace)

        normalized = self._normalize_search_result(task, parsed)
        selection = self._select_best_protocol(normalized)
        normalized["best_route"] = selection.get("best_route")
        normalized["ranking"] = selection.get("ranking", [])
        normalized["interaction_trace"] = trace
        normalized["agent_meta"] = {
            "tools": list(self.available_tools),
            "selection_mode": "llm" if self.model is not None else "heuristic",
        }
        return normalized

    def as_tool(self):
        agent_self = self

        @tool("synthesis_protocol_search")
        def synthesis_protocol_search(task: str) -> dict:
            """Поиск и сравнение методик синтеза."""
            return agent_self.run(task)

        return synthesis_protocol_search

    def as_node(self):
        agent_self = self

        def node(state: Dict[str, Any]) -> Dict[str, Any]:
            task = (
                state.get("synthesis_protocol_task")
                or state.get("synthesis_task")
                or state.get("protocol_search_task")
                or state.get("reaction_task")
                or state.get("task")
                or ""
            )

            if not isinstance(task, str):
                task = json.dumps(task, ensure_ascii=False, default=str)

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
