"""Supervisor-оркестратор для MAS.

Модуль собирает граф из Supervisor и рабочих агентов.
Supervisor принимает решение о следующем шаге, получает ответы агентов обратно
в историю и завершает работу узлом FINISH.
"""

from __future__ import annotations

import json
import os
from time import perf_counter
from typing import Any, Callable, Dict, List, TypedDict

from langgraph.graph import END, START, StateGraph
from loguru import logger

from app.llm_client import VseGPTConfig, VseGPTWrapper
from app.MAS.agents.decomposition_products_agent import SeparationMethodsAgent
from app.MAS.agents.properties_agent import StructurePropertiesAgent
from app.MAS.agents.reaction_products_agent import MixtureReactionAgent


API_KEY = os.getenv("VSEGPT_API_KEY", "")
BASE_URL = os.getenv("URL", "https://api.vsegpt.ru/v1")
MODEL_ORCHESTOR = os.getenv("MODEL_ORCHESTOR", "openai/gpt-5.4-nano-thinking-xhigh")
MODEL_AGENT = os.getenv("MODEL_AGENT", "openai/gpt-5.4-nano-thinking-xhigh")
MAX_WORKER_STEPS = int(os.getenv("MAX_WORKER_STEPS", "8"))
SUPERVISOR_TIMEOUT_SECONDS = float(os.getenv("SUPERVISOR_TIMEOUT_SECONDS", "45"))
LOG_VALUE_MAX_CHARS = int(os.getenv("LOG_VALUE_MAX_CHARS", "1200"))

ALL_WORKER_NODES = [
    "StructureAnalyzer",
    "MixtureReactionAgent",
    "SeparationMethodsAgent",
]
KNOWN_NODES = set(ALL_WORKER_NODES + ["FINISH"])


def _to_log_text(value: Any, max_chars: int = LOG_VALUE_MAX_CHARS) -> str:
    """Безопасно сериализует объект для логирования с ограничением длины."""
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False)
    else:
        text = str(value)

    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}... [truncated, total={len(text)} chars]"


def _parse_available_agents(raw_value: str) -> List[str]:
    """Парсит список доступных узлов из env `AVAILABLE_AGENTS`."""
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


config = VseGPTConfig(
    base_url=BASE_URL,
    api_key=API_KEY,
    model=MODEL_ORCHESTOR,
)
llm = VseGPTWrapper(config)


class TeamState(TypedDict, total=False):
    """Состояние графа оркестратора."""

    task: str
    target_molecule: str
    mixture_input: Dict[str, Any]
    separation_task: str
    history: List[Any]
    properties: Dict[str, Any]
    mixture_reaction: Dict[str, Any]
    separation_result: Dict[str, Any]
    next_worker: str


def _history_as_text(history: List[Any]) -> str:
    """Преобразует историю в безопасный текст для Supervisor prompt."""
    lines: List[str] = []
    for item in history:
        if isinstance(item, str):
            lines.append(item)
            continue

        if isinstance(item, dict):
            lines.append(json.dumps(item, ensure_ascii=False))

    return "\n".join(lines) if lines else "История пуста (начало работы)."


def _count_worker_steps(state: TeamState) -> int:
    """Считает число запусков рабочих узлов в истории."""
    history = state.get("history", [])
    steps = 0
    for item in history:
        if isinstance(item, dict) and str(item.get("agent")) in ALL_WORKER_NODES:
            steps += 1
    return steps


def _worker_already_called(state: TeamState, agent_name: str) -> bool:
    """Проверяет, был ли агент уже вызван хотя бы раз."""
    history = state.get("history", [])
    for item in history:
        if isinstance(item, dict) and str(item.get("agent")) == agent_name:
            return True
    return False


def _worker_failed_init(state: TeamState, agent_name: str) -> bool:
    """Проверяет, встречалась ли ошибка инициализации агента."""
    history = state.get("history", [])
    for item in reversed(history):
        if not isinstance(item, dict):
            continue
        if str(item.get("agent")) != agent_name:
            continue

        output = item.get("output")
        if isinstance(output, dict) and output.get("initialization_error"):
            return True

    return False


def _pick_next_available_worker(
    state: TeamState,
    exclude: set[str] | None = None,
    prefer_not_called: bool = False,
) -> str:
    """Выбирает следующего доступного агента или FINISH.

    Args:
        state: Текущее состояние графа.
        exclude: Узлы, которые нужно исключить из выбора.
        prefer_not_called: Если True, не выбирает уже вызванных агентов.
    """
    excluded = exclude or set()

    for node in ALL_WORKER_NODES:
        if node not in AVAILABLE_AGENTS or node in excluded:
            continue
        if _worker_failed_init(state, node):
            continue
        if prefer_not_called and _worker_already_called(state, node):
            continue
        return node

    return "FINISH"


def _last_agent_output(state: TeamState, agent_name: str) -> Any:
    """Возвращает последний output конкретного агента из history."""
    for item in reversed(state.get("history", [])):
        if not isinstance(item, dict):
            continue
        if str(item.get("agent")) == agent_name:
            return item.get("output")
    return None


def _format_supervisor_summary(last_event: Dict[str, Any]) -> str:
    """Формирует краткую сводку Supervisor по ответу последнего агента."""
    agent_name = str(last_event.get("agent", "UnknownAgent"))
    output = last_event.get("output")

    if isinstance(output, dict):
        prediction = output.get("prediction")
        if prediction is not None:
            return (
                f"Получен ответ от {agent_name}. "
                f"Ключевая часть результата (prediction): {prediction}"
            )
        return f"Получен ответ от {agent_name}: {json.dumps(output, ensure_ascii=False)}"

    return f"Получен ответ от {agent_name}: {output}"


def _append_supervisor_reply_if_needed(state: TeamState) -> None:
    """Добавляет ответ Supervisor после ответа рабочего агента."""
    history = state.setdefault("history", [])
    if not history:
        return

    last_item = history[-1]
    if not isinstance(last_item, dict):
        return

    if str(last_item.get("agent")) == "Supervisor":
        return

    summary = _format_supervisor_summary(last_item)
    supervisor_output = {
        "summary": summary,
        "prediction": summary,
        "source_agent": last_item.get("agent"),
    }
    history.append(
        {
            "agent": "Supervisor",
            "input": state.get("task", ""),
            "output": supervisor_output,
        }
    )

    logger.info(
        "Supervisor: получен ответ от агента {}, добавляю свой ответ в history: {}",
        last_item.get("agent"),
        _to_log_text(supervisor_output),
    )


def _append_supervisor_finish_reply_if_needed(
    state: TeamState,
    reason: str = "",
) -> None:
    """Гарантирует, что перед FINISH последний ответ в history — от Supervisor."""
    history = state.setdefault("history", [])

    if history and isinstance(history[-1], dict) and str(history[-1].get("agent")) == "Supervisor":
        return

    if history and isinstance(history[-1], dict):
        summary = _format_supervisor_summary(history[-1])
        source_agent = history[-1].get("agent")
    else:
        summary = (
            "Я — Supervisor. Завершаю работу без вызова агентов. "
            "Уточните задачу или добавьте входные данные."
        )
        source_agent = "Supervisor"

    if reason:
        summary = f"{summary} Причина завершения: {reason}."

    supervisor_output = {
        "summary": summary,
        "prediction": summary,
        "source_agent": source_agent,
    }

    history.append(
        {
            "agent": "Supervisor",
            "input": state.get("task", ""),
            "output": supervisor_output,
        }
    )

    logger.info(
        "Supervisor: финальный ответ перед FINISH добавлен: {}",
        _to_log_text(supervisor_output),
    )


def _build_supervisor_system_prompt() -> str:
    """Собирает system prompt для выбора следующего узла."""
    allowed_nodes = json.dumps(AVAILABLE_AGENTS, ensure_ascii=False)
    return f"""Ты — Главный Супервизор (координатор) мультиагентной химической лаборатории.
Твоя цель: проанализировать задачу пользователя и историю работы, чтобы выбрать следующий шаг.

ПРАВИЛА МАРШРУТИЗАЦИИ:
1. Изучи задачу и историю.
2. Если в истории есть ответ, решающий задачу пользователя, ИЛИ если все подходящие агенты вернули ошибку — выбери FINISH.
3. Если нужен анализ структуры молекулы И в данных указан параметр SMILES — выбери StructureAnalyzer. Если SMILES нет, НЕ выбирай этого агента.
4. Если нужен анализ продуктов реакции И в данных есть смесь (mixture_input) — выбери MixtureReactionAgent. Если данных о смеси нет, НЕ выбирай его.
5. Если нужны методы разделения смеси — выбери SeparationMethodsAgent.
6. ВАЖНО: Если агент вернул ошибку (например, "error": ...), НЕ ВЫЗЫВАЙ ЕГО СНОВА. Переходи к FINISH или другому агенту, если задача состоит из нескольких частей.

СТРОГИЕ ОГРАНИЧЕНИЯ:
- Не вызывай одного и того же агента повторно для одной и той же подзадачи.
- Ответ должен быть строго валидным JSON-объектом.
- Без Markdown, комментариев и пояснений.

Формат ответа:
{{"next_node": "<ИМЯ_УЗЛА>"}}

Допустимые значения для <ИМЯ_УЗЛА>: {allowed_nodes}
"""


def _error_node(agent_name: str, error: Exception):
    """Возвращает fallback-узел, если агент не удалось инициализировать."""

    def node(state: TeamState):
        logger.error("{} не инициализирован: {}", agent_name, error)
        state.setdefault("history", []).append(
            {
                "agent": agent_name,
                "output": {
                    "error": str(error),
                    "initialization_error": True,
                },
            }
        )
        return state

    return node


def _timed_worker_node(agent_name: str, node_fn: Callable[[TeamState], TeamState]):
    """Обертка для логирования времени и ответа рабочего агента."""

    def wrapped(state: TeamState) -> TeamState:
        started = perf_counter()
        logger.info("{}: запуск...", agent_name)

        try:
            updated_state = node_fn(state)
        except Exception as exc:
            elapsed = perf_counter() - started
            logger.exception(
                "{}: ошибка за {:.2f} c: {}",
                agent_name,
                elapsed,
                exc,
            )
            state.setdefault("history", []).append(
                {
                    "agent": agent_name,
                    "output": {
                        "error": str(exc),
                        "runtime_error": True,
                    },
                }
            )
            return state

        elapsed = perf_counter() - started
        output = _last_agent_output(updated_state, agent_name)
        logger.info(
            "{}: ответ за {:.2f} c: {}",
            agent_name,
            elapsed,
            _to_log_text(output),
        )
        return updated_state

    return wrapped


structure_agent_instance = StructurePropertiesAgent(llm=llm, temperature=0.3)
structure_analyzer_node = structure_agent_instance.as_node()

try:
    mixture_agent_instance = MixtureReactionAgent(model=MODEL_AGENT, temperature=0.0)
    mixture_reaction_node = mixture_agent_instance.as_node()
except Exception as exc:
    mixture_reaction_node = _error_node("MixtureReactionAgent", exc)

try:
    separation_agent_instance = SeparationMethodsAgent(
        model=MODEL_AGENT,
        temperature=0.0,
    )
    separation_methods_node = separation_agent_instance.as_node()
except Exception as exc:
    separation_methods_node = _error_node("SeparationMethodsAgent", exc)


def supervisor_node(state: TeamState):
    """Основной узел Supervisor: выбирает следующий шаг графа."""
    started_total = perf_counter()
    logger.info("Supervisor: анализирую историю и принимаю решение...")
    _append_supervisor_reply_if_needed(state)

    worker_steps = _count_worker_steps(state)
    if worker_steps >= MAX_WORKER_STEPS:
        logger.warning(
            "Supervisor: достигнут лимит шагов worker ({}) -> FINISH",
            MAX_WORKER_STEPS,
        )
        state["next_worker"] = "FINISH"
        _append_supervisor_finish_reply_if_needed(
            state,
            reason="достигнут лимит шагов",
        )
        logger.info("Supervisor: завершил шаг за {:.2f} c", perf_counter() - started_total)
        return state

    history_text = _history_as_text(state.get("history", []))
    system_prompt = _build_supervisor_system_prompt()

    prompt = (
        f"Задача: {state.get('task', '')}\n"
        f"SMILES: {state.get('target_molecule', 'Не указан')}\n\n"
        f"Текущая история работы:\n{history_text}"
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
        logger.error(
            "Supervisor: ошибка вызова LLM за {:.2f} c: {}",
            llm_elapsed,
            exc,
        )

        fallback_next = (
            "FINISH"
            if _count_worker_steps(state) > 0
            else _pick_next_available_worker(state)
        )
        logger.warning(
            "Supervisor: fallback-маршрут -> {}",
            fallback_next,
        )
        state["next_worker"] = fallback_next
        if fallback_next == "FINISH":
            _append_supervisor_finish_reply_if_needed(
                state,
                reason="ошибка вызова Supervisor LLM",
            )
        logger.info("Supervisor: завершил шаг за {:.2f} c", perf_counter() - started_total)
        return state

    llm_elapsed = perf_counter() - started_llm
    logger.info(
        "Supervisor: ответ LLM за {:.2f} c: {}",
        llm_elapsed,
        _to_log_text(result),
    )

    if isinstance(result, dict) and "next_node" in result:
        next_worker = str(result["next_node"])
    else:
        logger.warning("Supervisor: ошибка парсинга решения, завершаю работу.")
        next_worker = "FINISH"

    if next_worker not in AVAILABLE_AGENTS:
        logger.warning(
            "Supervisor: узел '{}' отключен или неизвестен -> FINISH",
            next_worker,
        )
        next_worker = "FINISH"

    # Не допускаем повторных вызовов одного и того же рабочего агента.
    if next_worker != "FINISH" and _worker_already_called(state, next_worker):
        logger.warning(
            "Supervisor: агент '{}' уже вызывался. Ищу альтернативу.",
            next_worker,
        )
        next_worker = _pick_next_available_worker(
            state,
            exclude={next_worker},
            prefer_not_called=True,
        )

    if next_worker != "FINISH" and _worker_failed_init(state, next_worker):
        logger.warning(
            "Supervisor: агент '{}' недоступен (ошибка инициализации). Ищу альтернативу.",
            next_worker,
        )
        next_worker = _pick_next_available_worker(
            state,
            exclude={next_worker},
            prefer_not_called=True,
        )

    logger.info("Supervisor решил: передаю задачу -> {}", next_worker)
    state["next_worker"] = next_worker

    if next_worker == "FINISH":
        _append_supervisor_finish_reply_if_needed(state)

    logger.info("Supervisor: завершил шаг за {:.2f} c", perf_counter() - started_total)
    return state


workflow = StateGraph(TeamState)
workflow.add_node("Supervisor", supervisor_node)
workflow.add_node(
    "StructureAnalyzer",
    _timed_worker_node("StructureAnalyzer", structure_analyzer_node),
)  # type: ignore[arg-type]
workflow.add_node(
    "MixtureReactionAgent",
    _timed_worker_node("MixtureReactionAgent", mixture_reaction_node),
)  # type: ignore[arg-type]
workflow.add_node(
    "SeparationMethodsAgent",
    _timed_worker_node("SeparationMethodsAgent", separation_methods_node),
)  # type: ignore[arg-type]

workflow.add_edge(START, "Supervisor")

workflow.add_edge("StructureAnalyzer", "Supervisor")
workflow.add_edge("MixtureReactionAgent", "Supervisor")
workflow.add_edge("SeparationMethodsAgent", "Supervisor")


def route_supervisor(state: TeamState):
    """Роутер для условных ребер Supervisor."""
    return state.get("next_worker", "FINISH")


workflow.add_conditional_edges(
    "Supervisor",
    route_supervisor,
    {
        "StructureAnalyzer": "StructureAnalyzer",
        "MixtureReactionAgent": "MixtureReactionAgent",
        "SeparationMethodsAgent": "SeparationMethodsAgent",
        "FINISH": END,
    },
)

app = workflow.compile()
