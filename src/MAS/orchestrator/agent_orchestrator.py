from __future__ import annotations

import json
import operator
import os
from time import perf_counter
from typing import Annotated, Any, Callable, Dict, List, TypedDict

from langgraph.graph import END, START, StateGraph
from loguru import logger

from src.llm_client import VseGPTConfig, VseGPTWrapper
from src.MAS.agents.properties_agent import StructurePropertiesAgent
from src.MAS.agents.literature_rag_agent import LiteratureRAGAgent
from src.MAS.agents.solver_agent import SynthesisProtocolSearchAgent

API_KEY = os.getenv("VSEGPT_API_KEY", "")
BASE_URL = os.getenv("URL", "https://api.vsegpt.ru/v1")
MODEL_ORCHESTOR = os.getenv("MODEL_ORCHESTOR", "openai/gpt-5.4-nano-thinking-xhigh")
MODEL_AGENT = os.getenv("MODEL_AGENT", "openai/gpt-5.4-nano-thinking-xhigh")
MAX_WORKER_STEPS = int(os.getenv("MAX_WORKER_STEPS", "8"))
SUPERVISOR_TIMEOUT_SECONDS = float(os.getenv("SUPERVISOR_TIMEOUT_SECONDS", "45"))
LOG_VALUE_MAX_CHARS = int(os.getenv("LOG_VALUE_MAX_CHARS", "1200"))

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

    # сохранение структур взаимодействий
    agent_interactions: Dict[str, Any]
    supervisor_trace: Annotated[List[Dict[str, Any]], operator.add]


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
            "AVAILABLE_AGENTS содержит только неизвестные узлы. "
            "Использую дефолтный список."
        )
        return ALL_WORKER_NODES + ["FINISH"]

    if "FINISH" not in filtered:
        filtered.append("FINISH")

    return filtered


AVAILABLE_AGENTS = _parse_available_agents(os.getenv("AVAILABLE_AGENTS", ""))


def _history_as_text(history: List[Dict[str, Any]]) -> str:
    if not history:
        return "История пуста (начало работы)."

    lines: List[str] = []
    for item in history:
        lines.append(json.dumps(item, ensure_ascii=False, default=str))
    return "\n".join(lines)


def _agent_interactions_as_text(agent_interactions: Dict[str, Any] | None) -> str:
    if not agent_interactions:
        return "Структуры взаимодействия с агентами пока отсутствуют."

    try:
        return json.dumps(agent_interactions, ensure_ascii=False, default=str)
    except Exception:
        return str(agent_interactions)


def _extract_latest_agent_event(
    history: List[Dict[str, Any]],
    agent_name: str,
) -> Dict[str, Any] | None:
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
                return (
                    f"По результатам LiteratureRAGAgent найден ответ: {answer} "
                    f"(источников: {len(sources)})."
                )
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
                base = (
                    f"Найдены и структурированы методики синтеза: {len(protocols)} "
                    f"(достигнут целевой минимум маршрутов)."
                )
            else:
                base = (
                    f"Найдены и структурированы методики синтеза: {len(protocols)} "
                    f"(возвращены все доступные найденные маршруты)."
                )

            if best_route_summary:
                return f"{base} Дополнительно {best_route_summary}."
            return base

        if output.get("error") == "invalid_json":
            return (
                "Агент поиска методик синтеза отработал, но вернул невалидный JSON; "
                "маршруты не удалось надёжно извлечь."
            )

        return "Агент поиска методик синтеза завершил анализ, но релевантные маршруты не найдены."

    if agent_name == "StructureAnalyzer" and isinstance(output, dict):
        prediction = output.get("prediction")
        if prediction is not None:
            return (
                "Выполнен анализ структуры целевой молекулы. "
                f"Ключевой результат: {prediction}"
            )

    if isinstance(output, dict) and output.get("prediction") is not None:
        return (
            f"Получен ответ от {agent_name}. "
            f"Ключевая часть результата (prediction): {output.get('prediction')}"
        )

    return f"Получен ответ от {agent_name}: {json.dumps(output, ensure_ascii=False, default=str)}"


def _build_supervisor_event(
    task: str,
    message: str,
    source_agent: str = "Supervisor",
) -> Dict[str, Any]:
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


def _safe_copy_agent_interactions(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, default=str))
    except Exception:
        return dict(value)


def _merge_agent_interactions(
    base: Dict[str, Any] | None,
    patch: Dict[str, Any] | None,
) -> Dict[str, Any]:
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
        "output": result_payload,
        "history_event": last_event,
    }

    if isinstance(result_payload, dict):
        if "interaction_trace" in result_payload:
            snapshot["interaction_trace"] = result_payload.get("interaction_trace", [])
        if "agent_meta" in result_payload:
            snapshot["agent_meta"] = result_payload.get("agent_meta")
        if "best_route" in result_payload:
            snapshot["best_route"] = result_payload.get("best_route")
        if "ranking" in result_payload:
            snapshot["ranking"] = result_payload.get("ranking")
        if "sources" in result_payload:
            snapshot["sources"] = result_payload.get("sources")
        if "warnings" in result_payload:
            snapshot["warnings"] = result_payload.get("warnings")
        if "summary" in result_payload:
            snapshot["summary"] = result_payload.get("summary")

    return snapshot


def _build_supervisor_system_prompt() -> str:
    allowed_nodes = json.dumps(AVAILABLE_AGENTS, ensure_ascii=False)
    return f"""Ты — Главный Supervisor мультиагентной системы химического ассистента.

Твоя роль:
- Координировать работу worker-агентов.
- На каждом шаге выбирать РОВНО один следующий узел: worker-агент или FINISH.
- Давать пользователю понятный итог в поле user_message.
- Если worker-агенты не нужны, завершать задачу прямым полезным ответом через FINISH.

Главная цель:
- Дать максимально полный и профессиональный результат по запросу пользователя.
- Использовать worker-агентов только тогда, когда они действительно нужны.
- Не вызывать одного и того же агента повторно для одной и той же задачи.
- Не запрашивать лишние данные, если можно ответить по существу без них.

Доступные узлы:
{allowed_nodes}

Worker-агенты и условия готовности:
1) StructureAnalyzer
- Выбирай только если есть непустой `target_molecule` (SMILES).
- Используй его для анализа конкретной молекулы и её свойств.
- Не выбирай, если `target_molecule` пуст.

2) SynthesisProtocolSearchAgent
- Выбирай, когда пользователь просит:
  - найти методики или маршруты синтеза;
  - подобрать литературные процедуры получения вещества;
  - собрать несколько вариантов синтеза;
  - сравнить условия реакций, выходы, реагенты, катализаторы, растворители;
  - получить набор возможных синтетических путей;
  - выбрать наиболее практичный / удобный маршрут синтеза.
- Этот агент возвращает не менее 3 различных маршрутов/методик, если они существуют.
- Если найдено меньше 3, он возвращает все найденные.
- После поиска он также выбирает лучший маршрут по мнению модели
  с учётом не только корректности, но и практичности/удобства.
- Он сохраняет структуру взаимодействия: trace, ranking, best_route, raw/meta.
- Используй его, когда нужен сбор и структурирование экспериментальных процедур
  или когда нужно выбрать наиболее удобный путь среди найденных.

3) LiteratureRAGAgent
- Выбирай, когда запрос пользователя является справочным, фактологическим, литературным,
  требует точного ответа, ссылки на источники, поиска по статьям, книгам, обзорам,
  патентам, базам знаний или внутреннему индексу документов.
- Используй его для вопросов в стиле:
  "что известно о...",
  "найди данные по литературе...",
  "какие есть методы...",
  "что пишут статьи...",
  "дай точный ответ по источникам..."
- Выбирай его также тогда, когда для ответа нужна повышенная точность и опора на retrieval,
  даже если вопрос выглядит как общий теоретический.
- Не выбирай его только в случае, если вопрос тривиален и не требует внешнего поиска.

Критически важные правила:
- Если пользователь просит именно найти протоколы синтеза, условия реакции, сравнить выходы,
  реагенты, растворители, катализаторы или выбрать лучшую методику синтеза,
  предпочитай SynthesisProtocolSearchAgent.
- Если нужен общий литературный обзор, фактологическая справка, ответы по источникам,
  но без акцента на экспериментальные протоколы, предпочитай LiteratureRAGAgent.
- Если запрос пользователя общий теоретический, справочный, фактологический,
  образовательный или литературный вопрос по химии, сначала проверь,
  нужен ли retrieval через LiteratureRAGAgent.
- Не запрашивай SMILES для общих справочных вопросов.
- Не запрашивай лишние данные, если LiteratureRAGAgent или SynthesisProtocolSearchAgent
  уже могут начать работу по `task`.

Политика маршрутизации:
- Сначала определи, можно ли ответить напрямую без worker-агентов.
- Для справочных, фактологических и литературных вопросов по умолчанию предпочитай
  LiteratureRAGAgent вместо прямого FINISH, если он доступен.
- Для запросов о протоколах синтеза, условиях реакций, сравнении методик и выборе
  наиболее удобного синтетического пути по умолчанию предпочитай
  SynthesisProtocolSearchAgent, если он доступен.
- Если можно ответить напрямую без потери точности — выбери FINISH.
- Если нужен специализированный анализ, определи релевантных агентов.
- Если релевантный агент еще не вызывался и входные данные достаточны — выбери его.
- Если релевантных невызывавшихся агентов с достаточными данными не осталось — выбери FINISH.
- Если данных действительно недостаточно для релевантного агента — выбери FINISH и кратко перечисли, чего именно не хватает.
- Если в истории есть ошибки инициализации или выполнения агента, не выбирай этот агент повторно.

Критерии FINISH:
- Все релевантные агенты уже отработали, или
- Нет достаточных входных данных для дальнейших релевантных вызовов, или
- На вопрос можно качественно ответить без вызова worker-агентов, или
- Задача уже решена по history и/или agent_interactions.

Требования к формату ответа:
- Верни только валидный JSON-объект.
- Без markdown, без комментариев, без текста вокруг JSON.
- Ровно два поля:
{{
  "next_node": "<ИМЯ_УЗЛА_ИЛИ_FINISH>",
  "user_message": "<краткий профессиональный текст для пользователя>"
}}

Требования к user_message:
- Если next_node != FINISH: коротко (1 предложение) сообщи, какой следующий шаг выполняется.
- Если next_node == FINISH:
  - если вопрос общий, дай прямой ответ по существу;
  - если были результаты history, дай сжатый профессиональный итог;
  - если у SynthesisProtocolSearchAgent есть best_route, учти его в итоговом сообщении;
  - если данных действительно не хватило, явно перечисли только недостающие данные.
"""


def _parse_supervisor_decision(result: Any) -> tuple[str, str]:
    if not isinstance(result, dict):
        return "FINISH", ""

    payload = result.get("data", result)
    if not isinstance(payload, dict):
        return "FINISH", ""

    next_node = str(payload.get("next_node", "FINISH"))
    user_message = str(payload.get("user_message", "")).strip()
    return next_node, user_message


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

        working_state: Dict[str, Any] = dict(state)
        working_state["history"] = list(state.get("history", []))
        working_state["agent_interactions"] = _safe_copy_agent_interactions(
            state.get("agent_interactions", {})
        )

        try:
            updated_state = node_fn(working_state)
        except Exception as exc:
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

        logger.info(
            "{}: ответ за {:.2f} c: {}",
            agent_name,
            elapsed,
            _to_log_text(last_event.get("output")),
        )

        incoming_interactions = _safe_copy_agent_interactions(state.get("agent_interactions", {}))
        updated_interactions = _safe_copy_agent_interactions(
            updated_state.get("agent_interactions", {})
        )

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

        return updates

    return wrapped


structure_agent_instance = StructurePropertiesAgent(temperature=0.01)
structure_node_legacy = structure_agent_instance.as_node()

try:
    synthesis_protocol_agent_instance = SynthesisProtocolSearchAgent(
        temperature=0.01,
    )
    synthesis_protocol_node_legacy = synthesis_protocol_agent_instance.as_node()
except Exception as exc:
    synthesis_protocol_node_legacy = _error_node("SynthesisProtocolSearchAgent", exc)

try:
    literature_rag_agent_instance = LiteratureRAGAgent(
        model=MODEL_AGENT,
        temperature=0.0,
    )
    literature_rag_node_legacy = literature_rag_agent_instance.as_node()
except Exception as exc:
    literature_rag_node_legacy = _error_node("LiteratureRAGAgent", exc)


def supervisor_node(state: TeamState) -> Dict[str, Any]:
    started_total = perf_counter()
    history = list(state.get("history", []))
    agent_interactions = _safe_copy_agent_interactions(state.get("agent_interactions", {}))

    logger.info("Supervisor: анализирую историю и принимаю решение...")

    worker_steps = sum(1 for item in history if str(item.get("agent")) in ALL_WORKER_NODES)
    if worker_steps >= MAX_WORKER_STEPS:
        message = (
            "Завершаю работу: достигнут лимит шагов. "
            "Уточните входные данные, чтобы продолжить."
        )
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
    except Exception as exc:
        llm_elapsed = perf_counter() - started_llm
        logger.error("Supervisor: ошибка вызова LLM за {:.2f} c: {}", llm_elapsed, exc)

        called = _called_workers(history)
        failed = _failed_init_workers(history)
        fallback_next = "FINISH" if called else _pick_next_available_worker(called, failed)

        if fallback_next == "FINISH":
            last_worker = _extract_latest_worker_event(history)
            if last_worker:
                message = _format_worker_summary(last_worker)
                event = _build_supervisor_event(
                    state.get("task", ""),
                    message,
                    source_agent=str(last_worker.get("agent", "Supervisor")),
                )
            else:
                message = "Не удалось получить ответ от модели Supervisor. Попробуйте повторить запрос."
                event = _build_supervisor_event(state.get("task", ""), message)

            logger.info("Supervisor: завершил шаг за {:.2f} c", perf_counter() - started_total)
            return {
                "next_worker": "FINISH",
                "history": [event],
                "supervisor_trace": [
                    {
                        "decision": "FINISH",
                        "reason": "supervisor_llm_error",
                        "error": str(exc),
                        "message": message,
                    }
                ],
            }

        logger.warning("Supervisor: fallback-маршрут -> {}", fallback_next)
        logger.info("Supervisor: завершил шаг за {:.2f} c", perf_counter() - started_total)
        return {
            "next_worker": fallback_next,
            "supervisor_trace": [
                {
                    "decision": fallback_next,
                    "reason": "supervisor_llm_error_fallback_route",
                    "error": str(exc),
                }
            ],
        }

    llm_elapsed = perf_counter() - started_llm
    logger.info("Supervisor: ответ LLM за {:.2f} c: {}", llm_elapsed, _to_log_text(result))

    next_worker, user_message = _parse_supervisor_decision(result)
    if next_worker not in AVAILABLE_AGENTS:
        logger.warning("Supervisor: узел '{}' отключен или неизвестен -> FINISH", next_worker)
        next_worker = "FINISH"

    called = _called_workers(history)
    failed = _failed_init_workers(history)

    if next_worker != "FINISH" and (next_worker in called or next_worker in failed):
        logger.warning(
            "Supervisor: агент '{}' уже вызывался или недоступен. Ищу альтернативу.",
            next_worker,
        )
        next_worker = _pick_next_available_worker(
            called,
            failed,
            exclude={next_worker},
            prefer_not_called=True,
        )

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

        event = _build_supervisor_event(
            state.get("task", ""),
            user_message,
            source_agent=source_agent,
        )
        logger.info("Supervisor: завершил шаг за {:.2f} c", perf_counter() - started_total)
        return {
            "next_worker": "FINISH",
            "history": [event],
            "supervisor_trace": [
                {
                    "decision": "FINISH",
                    "reason": "llm_finish",
                    "message": user_message,
                }
            ],
        }

    logger.info("Supervisor: завершил шаг за {:.2f} c", perf_counter() - started_total)
    return {
        "next_worker": next_worker,
        "supervisor_trace": [
            {
                "decision": next_worker,
                "reason": "llm_route",
                "message": user_message,
            }
        ],
    }


workflow = StateGraph(TeamState)
workflow.add_node("Supervisor", supervisor_node)
workflow.add_node(
    "StructureAnalyzer",
    _timed_worker_node("StructureAnalyzer", structure_node_legacy, "properties"),
)  # type: ignore[arg-type]
workflow.add_node(
    "SynthesisProtocolSearchAgent",
    _timed_worker_node(
        "SynthesisProtocolSearchAgent",
        synthesis_protocol_node_legacy,
        "synthesis_protocol_result",
    ),
)  # type: ignore[arg-type]
workflow.add_node(
    "LiteratureRAGAgent",
    _timed_worker_node("LiteratureRAGAgent", literature_rag_node_legacy, "literature_result"),
)  # type: ignore[arg-type]

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