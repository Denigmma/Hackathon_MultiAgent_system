from __future__ import annotations

import json
import operator
import os
import re
from time import perf_counter
from typing import Annotated, Any, Callable, Dict, List, TypedDict

from langgraph.graph import END, START, StateGraph
from loguru import logger
from rdkit import Chem

from src.llm_client import VseGPTConfig, VseGPTWrapper
from src.MAS.agents.literature_rag_agent import LiteratureRAGAgent
from src.MAS.agents.properties_agent import StructurePropertiesAgent
from src.MAS.agents.solver_agent import SynthesisProtocolSearchAgent

API_KEY = os.getenv("VSEGPT_API_KEY", "")
BASE_URL = os.getenv("URL", "https://api.vsegpt.ru/v1")
MODEL_ORCHESTOR = os.getenv("MODEL_ORCHESTOR", "openai/gpt-5.4-nano-thinking-xhigh")
MODEL_AGENT = os.getenv("MODEL_AGENT", "openai/gpt-5.4-nano-thinking-xhigh")
MAX_WORKER_STEPS = int(os.getenv("MAX_WORKER_STEPS", "8"))
SUPERVISOR_TIMEOUT_SECONDS = float(os.getenv("SUPERVISOR_TIMEOUT_SECONDS", "45"))
LOG_VALUE_MAX_CHARS = int(os.getenv("LOG_VALUE_MAX_CHARS", "1200"))
SMILES_CANDIDATE_RE = re.compile(r"[A-Za-z0-9@+\-\[\]\(\)=#$\\/%.]+")

ALL_WORKER_NODES = [
    "StructureAnalyzer",
    "SynthesisProtocolSearchAgent",
    "LiteratureRAGAgent",
]
KNOWN_NODES = set(ALL_WORKER_NODES + ["FINISH"])


class TeamState(TypedDict, total=False):
    task: str
    target_molecule: str
    synthesis_protocol_task: str
    synthesis_protocol_result: Dict[str, Any]
    literature_query: str
    literature_result: Dict[str, Any]
    history: Annotated[List[Dict[str, Any]], operator.add]
    properties: Dict[str, Any]
    next_worker: str
    agent_interactions: Dict[str, Any]
    supervisor_trace: Annotated[List[Dict[str, Any]], operator.add]


llm: VseGPTWrapper | None = None
if API_KEY:
    config = VseGPTConfig(
        base_url=BASE_URL,
        api_key=API_KEY,
        model=MODEL_ORCHESTOR,
    )
    llm = VseGPTWrapper(config)


def _to_log_text(value: Any, max_chars: int = LOG_VALUE_MAX_CHARS) -> str:
    if isinstance(value, (dict, list)):
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            text = str(value)
    else:
        text = str(value)

    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}... [truncated, total={len(text)} chars]"


def _parse_available_agents(raw_value: str) -> List[str]:
    raw_items = [item.strip() for item in raw_value.split(",") if item.strip()]
    if not raw_items:
        return ALL_WORKER_NODES + ["FINISH"]

    filtered = [item for item in raw_items if item in KNOWN_NODES]
    if not filtered:
        logger.warning(
            "AVAILABLE_AGENTS содержит только неизвестные узлы. Использую дефолтный список."
        )
        return ALL_WORKER_NODES + ["FINISH"]

    if "FINISH" not in filtered:
        filtered.append("FINISH")
    return filtered


AVAILABLE_AGENTS = _parse_available_agents(os.getenv("AVAILABLE_AGENTS", ""))


def _history_as_text(history: List[Dict[str, Any]]) -> str:
    if not history:
        return "История пуста (начало работы)."
    return "\n".join(json.dumps(item, ensure_ascii=False, default=str) for item in history)


def _agent_interactions_as_text(agent_interactions: Dict[str, Any] | None) -> str:
    if not agent_interactions:
        return "Структуры взаимодействия с агентами пока отсутствуют."
    try:
        return json.dumps(agent_interactions, ensure_ascii=False, default=str)
    except Exception:
        return str(agent_interactions)


def _extract_smiles_from_text(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""

    for token in SMILES_CANDIDATE_RE.findall(text):
        candidate = token.strip(".,;:!?\"'")
        if not candidate:
            continue
        mol = Chem.MolFromSmiles(candidate)
        if mol is None:
            continue
        return Chem.MolToSmiles(mol, canonical=True)

    return ""


def _safe_copy_agent_interactions(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, default=str))
    except Exception:
        return dict(value)


def _normalize_state(state: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(state)
    task = str(normalized.get("task") or "").strip()

    target_molecule = str(normalized.get("target_molecule") or "").strip()
    if not target_molecule and task:
        target_molecule = _extract_smiles_from_text(task)
    if target_molecule:
        normalized["target_molecule"] = target_molecule

    if not normalized.get("synthesis_protocol_task") and task:
        normalized["synthesis_protocol_task"] = task
    if not normalized.get("literature_query") and task:
        normalized["literature_query"] = task

    normalized["history"] = list(normalized.get("history", []))
    normalized["agent_interactions"] = _safe_copy_agent_interactions(
        normalized.get("agent_interactions", {})
    )
    normalized["supervisor_trace"] = list(normalized.get("supervisor_trace", []))
    return normalized


def _extract_latest_agent_event(history: List[Dict[str, Any]], agent_name: str) -> Dict[str, Any] | None:
    for item in reversed(history):
        if str(item.get("agent")) == agent_name:
            return item
    return None


def _extract_latest_worker_event(history: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    for item in reversed(history):
        if str(item.get("agent")) in ALL_WORKER_NODES:
            return item
    return None


def _build_best_route_summary(best_route: Dict[str, Any] | None) -> str | None:
    if not isinstance(best_route, dict):
        return None

    route_id = best_route.get("route_id")
    confidence = best_route.get("confidence")
    score = best_route.get("practicality_score")
    if not route_id:
        return None

    parts = [f"модель выбрала лучший маршрут: {route_id}"]
    if score is not None:
        parts.append(f"practicality_score={score}")
    if confidence:
        parts.append(f"confidence={confidence}")
    return ", ".join(parts)


def _format_worker_summary(event: Dict[str, Any]) -> str:
    agent_name = str(event.get("agent", "UnknownAgent"))
    output = event.get("output")

    if agent_name == "LiteratureRAGAgent" and isinstance(output, dict):
        answer = output.get("answer")
        sources = output.get("sources", [])
        if answer:
            if sources:
                return f"По результатам LiteratureRAGAgent найден ответ: {answer} (источников: {len(sources)})."
            return f"По результатам LiteratureRAGAgent найден ответ: {answer}"
        return "LiteratureRAGAgent завершил анализ, но точный ответ по источникам не найден."

    if agent_name == "SynthesisProtocolSearchAgent" and isinstance(output, dict):
        protocols = output.get("protocols", [])
        summary = output.get("summary", {}) or {}
        enough_routes_found = summary.get("enough_routes_found")
        best_route = output.get("best_route", {}) or {}
        best_route_summary = _build_best_route_summary(best_route)
        if protocols:
            if enough_routes_found:
                base = f"Найдены и структурированы методики синтеза: {len(protocols)} (достигнут целевой минимум маршрутов)."
            else:
                base = f"Найдены и структурированы методики синтеза: {len(protocols)} (возвращены все доступные найденные маршруты)."
            return f"{base} Дополнительно {best_route_summary}." if best_route_summary else base
        if output.get("error") == "invalid_json":
            return "Агент поиска методик синтеза отработал, но вернул невалидный JSON; маршруты не удалось надёжно извлечь."
        if output.get("error") == "search_unavailable":
            note = output.get("summary", {}).get("coverage_note")
            return f"Агент методик синтеза недоступен: {note}" if note else "Агент методик синтеза недоступен."
        return "Агент поиска методик синтеза завершил анализ, но релевантные маршруты не найдены."

    if agent_name == "StructureAnalyzer" and isinstance(output, dict):
        summary = output.get("summary")
        prediction = output.get("prediction")
        if summary:
            return f"Выполнен анализ структуры целевой молекулы. {summary}"
        if prediction is not None:
            return f"Выполнен анализ структуры целевой молекулы. Ключевой результат: {prediction}"

    if isinstance(output, dict) and output.get("prediction") is not None:
        return f"Получен ответ от {agent_name}. Ключевая часть результата (prediction): {output.get('prediction')}"

    return f"Получен ответ от {agent_name}: {json.dumps(output, ensure_ascii=False, default=str)}"


def _build_supervisor_event(task: str, message: str, source_agent: str = "Supervisor") -> Dict[str, Any]:
    return {
        "agent": "Supervisor",
        "input": task,
        "output": {
            "summary": message,
            "prediction": message,
            "source_agent": source_agent,
        },
    }


def _called_workers(history: List[Dict[str, Any]]) -> set[str]:
    return {
        str(item.get("agent"))
        for item in history
        if str(item.get("agent")) in ALL_WORKER_NODES
    }


def _failed_init_workers(history: List[Dict[str, Any]]) -> set[str]:
    failed: set[str] = set()
    for item in history:
        agent = str(item.get("agent"))
        if agent not in ALL_WORKER_NODES:
            continue
        output = item.get("output")
        if isinstance(output, dict) and output.get("initialization_error"):
            failed.add(agent)
    return failed


def _pick_next_available_worker(
    called: set[str],
    failed: set[str],
    exclude: set[str] | None = None,
    prefer_not_called: bool = True,
) -> str:
    excluded = exclude or set()
    for node in ALL_WORKER_NODES:
        if node not in AVAILABLE_AGENTS or node in excluded or node in failed:
            continue
        if prefer_not_called and node in called:
            continue
        return node
    return "FINISH"


def _merge_agent_interactions(base: Dict[str, Any] | None, patch: Dict[str, Any] | None) -> Dict[str, Any]:
    merged = _safe_copy_agent_interactions(base)
    if not isinstance(patch, dict):
        return merged
    for key, value in patch.items():
        merged[key] = value
    return merged


def _build_worker_interaction_snapshot(
    agent_name: str,
    task: str,
    state_before: Dict[str, Any],
    updated_state: Dict[str, Any],
    last_event: Dict[str, Any],
    result_key: str,
) -> Dict[str, Any]:
    result_payload = updated_state.get(result_key)
    if result_payload is None and isinstance(last_event, dict):
        result_payload = last_event.get("output")

    snapshot: Dict[str, Any] = {
        "agent": agent_name,
        "input": {
            "task": task,
            "context": state_before.get("context"),
            "target_molecule": state_before.get("target_molecule"),
            "synthesis_protocol_task": state_before.get("synthesis_protocol_task"),
            "literature_query": state_before.get("literature_query"),
        },
        "result_key": result_key,
        "history_event": last_event,
    }

    if isinstance(result_payload, dict):
        for key in [
            "query",
            "backend",
            "answer",
            "sources",
            "confidence",
            "limitations",
            "prediction",
            "best_route",
            "ranking",
            "protocols",
            "warnings",
            "summary",
        ]:
            if key in result_payload:
                snapshot[key] = result_payload.get(key)
    return snapshot


def _build_supervisor_system_prompt() -> str:
    allowed_nodes = json.dumps(AVAILABLE_AGENTS, ensure_ascii=False)
    return f"""Ты — Главный Supervisor мультиагентной системы химического ассистента.

Твоя роль:
- Координировать работу worker-агентов.
- На каждом шаге выбирать РОВНО один следующий узел: worker-агент или FINISH.
- Если worker-агенты не нужны, завершать задачу прямым полезным ответом через FINISH.
- Не использовать worker-агентов без необходимости.

Главная цель:
- Дать максимально полный и профессиональный результат по запросу пользователя.
- Сначала оценить, можно ли ответить самостоятельно, не вызывая worker-агентов.
- Вызывать worker-агентов только тогда, когда они реально добавляют ценность.
- Не вызывать одного и того же агента повторно для одной и той же задачи.

Доступные узлы:
{allowed_nodes}

Worker-агенты и условия готовности:
1) StructureAnalyzer
- Выбирай только если есть непустой `target_molecule` (SMILES).
- Используй его для анализа конкретной молекулы и её свойств.
- Не выбирай, если `target_molecule` пуст.

2) SynthesisProtocolSearchAgent
- Выбирай, когда пользователь просит найти методики или маршруты синтеза, подобрать литературные процедуры получения вещества,
  собрать несколько вариантов синтеза, сравнить условия реакций, выходы, реагенты, катализаторы, растворители,
  либо выбрать наиболее практичный маршрут синтеза.

3) LiteratureRAGAgent
- Выбирай только тогда, когда действительно нужен retrieval по литературе, источникам, статьям, обзорам, патентам,
  внутреннему индексу или базе знаний.
- Не используй LiteratureRAGAgent как дефолтный маршрут для любых общих химических вопросов.
- Если вопрос общий теоретический, образовательный, объяснительный или справочный и ты можешь ответить сам без retrieval,
  выбери FINISH и дай полезный прямой ответ.

Политика маршрутизации:
- Сначала определи, можно ли качественно ответить напрямую без worker-агентов.
- Для общих теоретических, объяснительных и образовательных вопросов по химии предпочитай прямой ответ через FINISH.
- Для анализа молекулы по SMILES предпочитай StructureAnalyzer.
- Для синтетических протоколов и выбора маршрута предпочитай SynthesisProtocolSearchAgent.
- Для retrieval/литературных задач предпочитай LiteratureRAGAgent.
- Если агент уже вызывался или был недоступен, не выбирай его повторно.
- Если данных действительно недостаточно, выбери FINISH и перечисли только недостающие данные.

Формат ответа:
- Верни только валидный JSON-объект.
- Без markdown, без комментариев, без текста вокруг JSON.
- Ровно два поля:
{{
  "next_node": "<ИМЯ_УЗЛА_ИЛИ_FINISH>",
  "user_message": "<краткий профессиональный текст для пользователя>"
}}

Правила для user_message:
- Если next_node != FINISH: коротко опиши, какой следующий шаг выполняется.
- Если next_node == FINISH: дай прямой ответ по существу или сжатый итог по результатам history.
"""


def _strip_code_fences(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    if text.startswith("```"):
        text = re.sub(r"^```(?:json|JSON)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _parse_jsonish_dict(value: Any) -> Dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return None

    candidates = [value, _strip_code_fences(value)]
    for candidate in candidates:
        candidate = (candidate or "").strip()
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        start = candidate.find("{")
        end = candidate.rfind("}")
        if 0 <= start < end:
            fragment = candidate[start : end + 1]
            try:
                parsed = json.loads(fragment)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
    return None


def _parse_supervisor_decision(result: Any) -> tuple[str, str]:
    if not isinstance(result, dict):
        return "FINISH", ""

    payload = _parse_jsonish_dict(result.get("data"))
    if payload is None:
        payload = _parse_jsonish_dict(result.get("raw_text"))
    if payload is None:
        payload = _parse_jsonish_dict(result)
    if payload is None:
        return "FINISH", ""

    next_node = str(payload.get("next_node", "FINISH")).strip() or "FINISH"
    user_message = str(payload.get("user_message", "")).strip()
    return next_node, user_message


def _task_lower(state: TeamState) -> str:
    return str(state.get("task") or "").lower()


def _looks_like_structure_task(state: TeamState) -> bool:
    task = _task_lower(state)
    return bool(state.get("target_molecule")) or any(
        token in task
        for token in ["smiles", "молекул", "структур", "дескриптор", "свойств", "logp", "tpsa"]
    )


def _looks_like_synthesis_task(state: TeamState) -> bool:
    task = _task_lower(state)
    return any(
        token in task
        for token in [
            "синтез",
            "маршрут",
            "методик",
            "протокол",
            "услови",
            "реагент",
            "катализ",
            "растворител",
            "выход",
            "получени",
        ]
    )


def _looks_like_literature_task(state: TeamState) -> bool:
    task = _task_lower(state)
    return any(
        token in task
        for token in [
            "литератур",
            "стат",
            "обзор",
            "источник",
            "doi",
            "патент",
            "что пишут",
            "по данным",
            "rag",
            "paper",
            "article",
            "review",
        ]
    )


def _infer_relevant_workers(state: TeamState) -> List[str]:
    relevant: List[str] = []
    if _looks_like_structure_task(state) and state.get("target_molecule"):
        relevant.append("StructureAnalyzer")
    if _looks_like_synthesis_task(state):
        relevant.append("SynthesisProtocolSearchAgent")
    if _looks_like_literature_task(state):
        relevant.append("LiteratureRAGAgent")
    return [node for node in relevant if node in AVAILABLE_AGENTS]


def _heuristic_supervisor_decision(state: TeamState) -> tuple[str, str, str]:
    history = list(state.get("history", []))
    called = _called_workers(history)
    failed = _failed_init_workers(history)
    relevant = [node for node in _infer_relevant_workers(state) if node not in failed]

    if relevant:
        for node in relevant:
            if node not in called:
                if node == "StructureAnalyzer" and not state.get("target_molecule"):
                    continue
                message_map = {
                    "StructureAnalyzer": "Анализирую структуру и свойства молекулы по SMILES.",
                    "SynthesisProtocolSearchAgent": "Ищу и структурирую маршруты синтеза, затем выберу наиболее практичный.",
                    "LiteratureRAGAgent": "Ищу ответ по литературным и retrieval-источникам.",
                }
                return node, message_map.get(node, "Запускаю профильного агента."), "heuristic_route"

        last_worker = _extract_latest_worker_event(history)
        if last_worker is not None:
            return "FINISH", _format_worker_summary(last_worker), "heuristic_finish_after_workers"

    if history:
        last_worker = _extract_latest_worker_event(history)
        if last_worker is not None:
            return "FINISH", _format_worker_summary(last_worker), "heuristic_finish_after_any_worker"

    if state.get("target_molecule") and "StructureAnalyzer" in AVAILABLE_AGENTS and "StructureAnalyzer" not in called and "StructureAnalyzer" not in failed:
        return "StructureAnalyzer", "Анализирую структуру и свойства молекулы по SMILES.", "heuristic_default_structure"

    return (
        "FINISH",
        "Оркестратор не видит необходимости вызывать специализированный инструмент по текущему запросу. Уточните, нужен ли анализ по SMILES, поиск маршрутов синтеза или литературный поиск.",
        "heuristic_direct_finish",
    )


def _validate_or_repair_decision(
    state: TeamState,
    proposed_next: str,
    user_message: str,
    reason: str,
) -> tuple[str, str, str]:
    history = list(state.get("history", []))
    called = _called_workers(history)
    failed = _failed_init_workers(history)

    next_worker = proposed_next if proposed_next in AVAILABLE_AGENTS else "FINISH"
    if next_worker == "StructureAnalyzer" and not str(state.get("target_molecule") or "").strip():
        return (
            "FINISH",
            user_message or "Для структурного анализа не удалось извлечь корректный SMILES из запроса.",
            f"{reason}_no_smiles",
        )

    if next_worker != "FINISH" and (next_worker in called or next_worker in failed):
        alternative = _pick_next_available_worker(called, failed, exclude={next_worker}, prefer_not_called=True)
        if alternative == "FINISH":
            last_worker = _extract_latest_worker_event(history)
            if last_worker is not None:
                return "FINISH", _format_worker_summary(last_worker), f"{reason}_already_called_finish"
            return "FINISH", user_message or "Релевантные агенты уже были вызваны.", f"{reason}_already_called_finish"
        return alternative, user_message, f"{reason}_rerouted"

    return next_worker, user_message, reason


def _error_node(agent_name: str, error: Exception):
    def node(_: TeamState) -> Dict[str, Any]:
        logger.error("{} не инициализирован: {}", agent_name, error)
        return {
            "history": [
                {
                    "agent": agent_name,
                    "output": {
                        "error": str(error),
                        "initialization_error": True,
                    },
                }
            ],
            "agent_interactions": {
                agent_name: {
                    "agent": agent_name,
                    "initialization_error": True,
                    "error": str(error),
                }
            },
        }

    return node


def _timed_worker_node(
    agent_name: str,
    node_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    result_key: str,
):
    def wrapped(state: TeamState) -> Dict[str, Any]:
        started = perf_counter()
        logger.info("{}: запуск...", agent_name)

        working_state: Dict[str, Any] = _normalize_state(dict(state))
        try:
            updated_state = node_fn(working_state)
        except Exception as exc:  # pragma: no cover - safety net
            elapsed = perf_counter() - started
            logger.exception("{}: ошибка за {:.2f} c: {}", agent_name, elapsed, exc)
            return {
                "history": [
                    {
                        "agent": agent_name,
                        "output": {
                            "error": str(exc),
                            "runtime_error": True,
                        },
                    }
                ],
                "agent_interactions": {
                    agent_name: {
                        "agent": agent_name,
                        "runtime_error": True,
                        "error": str(exc),
                    }
                },
            }

        elapsed = perf_counter() - started
        history = updated_state.get("history", [])
        last_event = _extract_latest_agent_event(history, agent_name)
        if last_event is None:
            last_event = {
                "agent": agent_name,
                "output": {"error": "Worker completed without history event."},
            }

        logger.info("{}: ответ за {:.2f} c: {}", agent_name, elapsed, _to_log_text(last_event.get("output")))

        incoming_interactions = _safe_copy_agent_interactions(state.get("agent_interactions", {}))
        updated_interactions = _safe_copy_agent_interactions(updated_state.get("agent_interactions", {}))
        snapshot = _build_worker_interaction_snapshot(
            agent_name=agent_name,
            task=str(state.get("task", "")),
            state_before=working_state,
            updated_state=updated_state,
            last_event=last_event,
            result_key=result_key,
        )
        merged_interactions = _merge_agent_interactions(incoming_interactions, updated_interactions)
        merged_interactions[agent_name] = snapshot

        updates: Dict[str, Any] = {
            "history": [last_event],
            "agent_interactions": merged_interactions,
        }
        if result_key in updated_state:
            updates[result_key] = updated_state[result_key]
        if "target_molecule" in updated_state:
            updates["target_molecule"] = updated_state["target_molecule"]
        return updates

    return wrapped


try:
    structure_agent_instance = StructurePropertiesAgent(temperature=0.01)
    structure_node_legacy = structure_agent_instance.as_node()
except Exception as exc:
    structure_node_legacy = _error_node("StructureAnalyzer", exc)

try:
    synthesis_protocol_agent_instance = SynthesisProtocolSearchAgent(temperature=0.01)
    synthesis_protocol_node_legacy = synthesis_protocol_agent_instance.as_node()
except Exception as exc:
    synthesis_protocol_node_legacy = _error_node("SynthesisProtocolSearchAgent", exc)

try:
    literature_rag_agent_instance = LiteratureRAGAgent(model=MODEL_AGENT, temperature=0.0)
    literature_rag_node_legacy = literature_rag_agent_instance.as_node()
except Exception as exc:
    literature_rag_node_legacy = _error_node("LiteratureRAGAgent", exc)


def supervisor_node(state: TeamState) -> Dict[str, Any]:
    started_total = perf_counter()
    state = _normalize_state(state)
    history = list(state.get("history", []))
    agent_interactions = _safe_copy_agent_interactions(state.get("agent_interactions", {}))

    logger.info("Supervisor: анализирую историю и принимаю решение...")

    worker_steps = sum(1 for item in history if str(item.get("agent")) in ALL_WORKER_NODES)
    if worker_steps >= MAX_WORKER_STEPS:
        message = "Завершаю работу: достигнут лимит шагов. Уточните входные данные, чтобы продолжить."
        event = _build_supervisor_event(state.get("task", ""), message)
        logger.warning("Supervisor: достигнут лимит шагов worker ({}) -> FINISH", MAX_WORKER_STEPS)
        logger.info("Supervisor: завершил шаг за {:.2f} c", perf_counter() - started_total)
        return {
            "next_worker": "FINISH",
            "history": [event],
            "supervisor_trace": [
                {
                    "decision": "FINISH",
                    "reason": "max_worker_steps_reached",
                    "message": message,
                }
            ],
        }

    if llm is None:
        next_worker, user_message, reason = _heuristic_supervisor_decision(state)
    else:
        system_prompt = _build_supervisor_system_prompt()
        prompt = (
            f"Задача: {state.get('task', '')}\n"
            f"SMILES: {state.get('target_molecule', 'Не указан')}\n"
            f"synthesis_protocol_task: {state.get('synthesis_protocol_task', state.get('task', ''))}\n"
            f"literature_query: {state.get('literature_query', state.get('task', ''))}\n\n"
            f"Текущая история:\n{_history_as_text(history)}\n\n"
            f"Структуры взаимодействия с агентами:\n{_agent_interactions_as_text(agent_interactions)}"
        )

        started_llm = perf_counter()
        try:
            result = llm.ask(
                prompt=prompt,
                system_prompt=system_prompt,
                json_mode=True,
                temperature=0.0,
                timeout=SUPERVISOR_TIMEOUT_SECONDS,
            )
            llm_elapsed = perf_counter() - started_llm
            logger.info("Supervisor: ответ LLM за {:.2f} c: {}", llm_elapsed, _to_log_text(result))
            proposed_next, user_message = _parse_supervisor_decision(result)
            if proposed_next == "FINISH" and not user_message:
                next_worker, user_message, reason = _heuristic_supervisor_decision(state)
                reason = f"llm_unparsed->{reason}"
            else:
                next_worker, user_message, reason = _validate_or_repair_decision(
                    state, proposed_next, user_message, "llm_route"
                )
        except Exception as exc:
            llm_elapsed = perf_counter() - started_llm
            logger.error("Supervisor: ошибка вызова LLM за {:.2f} c: {}", llm_elapsed, exc)
            next_worker, user_message, reason = _heuristic_supervisor_decision(state)
            reason = f"supervisor_llm_error->{reason}"

    logger.info("Supervisor решил: передаю задачу -> {}", next_worker)

    if next_worker == "FINISH":
        if not user_message:
            last_worker = _extract_latest_worker_event(history)
            if last_worker:
                user_message = _format_worker_summary(last_worker)
                source_agent = str(last_worker.get("agent", "Supervisor"))
            else:
                user_message = (
                    "Завершаю обработку. Уточните задачу или добавьте входные данные "
                    "(SMILES, описание целевого синтеза, ограничения, критерии выбора маршрута)."
                )
                source_agent = "Supervisor"
        else:
            source_agent = "Supervisor"

        event = _build_supervisor_event(state.get("task", ""), user_message, source_agent=source_agent)
        logger.info("Supervisor: завершил шаг за {:.2f} c", perf_counter() - started_total)
        return {
            "next_worker": "FINISH",
            "history": [event],
            "supervisor_trace": [
                {
                    "decision": "FINISH",
                    "reason": reason,
                    "message": user_message,
                }
            ],
        }

    logger.info("Supervisor: завершил шаг за {:.2f} c", perf_counter() - started_total)
    return {
        "next_worker": next_worker,
        "target_molecule": state.get("target_molecule", ""),
        "synthesis_protocol_task": state.get("synthesis_protocol_task", state.get("task", "")),
        "literature_query": state.get("literature_query", state.get("task", "")),
        "supervisor_trace": [
            {
                "decision": next_worker,
                "reason": reason,
                "message": user_message,
            }
        ],
    }


workflow = StateGraph(TeamState)
workflow.add_node("Supervisor", supervisor_node)
workflow.add_node(
    "StructureAnalyzer",
    _timed_worker_node("StructureAnalyzer", structure_node_legacy, "properties"),
)
workflow.add_node(
    "SynthesisProtocolSearchAgent",
    _timed_worker_node(
        "SynthesisProtocolSearchAgent",
        synthesis_protocol_node_legacy,
        "synthesis_protocol_result",
    ),
)
workflow.add_node(
    "LiteratureRAGAgent",
    _timed_worker_node("LiteratureRAGAgent", literature_rag_node_legacy, "literature_result"),
)

workflow.add_edge(START, "Supervisor")
workflow.add_edge("StructureAnalyzer", "Supervisor")
workflow.add_edge("SynthesisProtocolSearchAgent", "Supervisor")
workflow.add_edge("LiteratureRAGAgent", "Supervisor")


def route_supervisor(state: TeamState):
    return state.get("next_worker", "FINISH")


workflow.add_conditional_edges(
    "Supervisor",
    route_supervisor,
    {
        "StructureAnalyzer": "StructureAnalyzer",
        "SynthesisProtocolSearchAgent": "SynthesisProtocolSearchAgent",
        "LiteratureRAGAgent": "LiteratureRAGAgent",
        "FINISH": END,
    },
)

app = workflow.compile()
