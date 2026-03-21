"""Модуль Supervisor-оркестратора для MAS.

Этот файл собирает и компилирует граф LangGraph по паттерну Supervisor:
1. Пользовательская задача поступает в узел `Supervisor`.
2. Supervisor анализирует историю и выбирает следующего worker-агента.
3. Worker возвращает структурированный результат в `history`.
4. Управление всегда возвращается в Supervisor.
5. Supervisor либо выбирает следующий шаг, либо завершает работу узлом `FINISH`.

Ключевая идея реализации:
- `history` хранится как редуцируемое поле (`Annotated[..., operator.add]`),
  поэтому узлы возвращают только дельту событий, а не мутируют общий список вручную.
- Supervisor является единственной точкой принятия решения о маршруте (`next_worker`).
- Worker-узлы обернуты тайминг-логированием и безопасной обработкой исключений.
"""

from __future__ import annotations

import json
import operator
import os
from time import perf_counter
from typing import Annotated, Any, Callable, Dict, List, TypedDict

from langgraph.graph import END, START, StateGraph
from loguru import logger

from src.llm_client import VseGPTConfig, VseGPTWrapper
from src.MAS.agents.decomposition_products_agent import SeparationMethodsAgent
from src.MAS.agents.properties_agent import StructurePropertiesAgent
from src.MAS.agents.reaction_products_agent import MixtureReactionAgent


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


class TeamState(TypedDict, total=False):
    """Тип состояния графа оркестратора.

    Поля:
        task: Исходный текст задачи пользователя.
        target_molecule: SMILES-строка для задач анализа структуры.
        mixture_input: Структурированные входные данные по смеси/реакции.
        separation_task: Текст подзадачи для агента разделения.
        history: Журнал событий агентов и Supervisor.
            Важно: поле настроено через reducer (`operator.add`), поэтому
            каждый узел возвращает только новые события списка.
        properties: Результат `StructureAnalyzer`.
        mixture_reaction: Результат `MixtureReactionAgent`.
        separation_result: Результат `SeparationMethodsAgent`.
        next_worker: Имя следующего узла, выбранного Supervisor.
    """

    task: str
    target_molecule: str
    mixture_input: Dict[str, Any]
    separation_task: str
    history: Annotated[List[Dict[str, Any]], operator.add]
    properties: Dict[str, Any]
    mixture_reaction: Dict[str, Any]
    separation_result: Dict[str, Any]
    next_worker: str


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
    """Парсит переменную окружения `AVAILABLE_AGENTS`.

    Args:
        raw_value: Строка вида `"AgentA,AgentB,FINISH"`.

    Returns:
        Список валидных узлов. Если env пустая/некорректная, возвращает
        дефолтный набор из всех worker-узлов и `FINISH`.
    """
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
    """Преобразует историю событий в текст для prompt Supervisor.

    Args:
        history: Список событий в формате словарей.

    Returns:
        Единая строка с сериализованной историей, пригодная для передачи в LLM.
    """
    if not history:
        return "История пуста (начало работы)."

    lines: List[str] = []
    for item in history:
        lines.append(json.dumps(item, ensure_ascii=False))
    return "\n".join(lines)


def _extract_latest_agent_event(
    history: List[Dict[str, Any]],
    agent_name: str,
) -> Dict[str, Any] | None:
    """Ищет последнее событие конкретного агента.

    Args:
        history: История шагов графа.
        agent_name: Имя агента (`StructureAnalyzer`, `Supervisor` и т.д.).

    Returns:
        Последний словарь события агента или `None`, если событие не найдено.
    """
    for item in reversed(history):
        if str(item.get("agent")) == agent_name:
            return item
    return None


def _extract_latest_worker_event(history: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    """Ищет последнее событие любого worker-агента.

    Args:
        history: История шагов графа.

    Returns:
        Последнее событие worker-агента (не Supervisor) или `None`.
    """
    for item in reversed(history):
        if str(item.get("agent")) in ALL_WORKER_NODES:
            return item
    return None


def _format_worker_summary(event: Dict[str, Any]) -> str:
    """Готовит короткую текстовую сводку по результату worker-агента.

    Args:
        event: Событие worker-узла из истории.

    Returns:
        Человекочитаемое предложение для финального ответа Supervisor.
    """
    agent_name = str(event.get("agent", "UnknownAgent"))
    output = event.get("output")

    if isinstance(output, dict) and output.get("prediction") is not None:
        return (
            f"Получен ответ от {agent_name}. "
            f"Ключевая часть результата (prediction): {output.get('prediction')}"
        )

    return f"Получен ответ от {agent_name}: {json.dumps(output, ensure_ascii=False)}"


def _build_supervisor_event(
    task: str,
    message: str,
    source_agent: str = "Supervisor",
) -> Dict[str, Any]:
    """Создает унифицированное событие Supervisor для записи в history.

    Args:
        task: Исходный запрос пользователя.
        message: Текст ответа Supervisor.
        source_agent: Агент-источник данных, на основе которого сформирован ответ.

    Returns:
        Словарь события в формате, совместимом с текущим state/history.
    """
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
    """Возвращает множество worker-узлов, которые уже вызывались."""
    return {
        str(item.get("agent"))
        for item in history
        if str(item.get("agent")) in ALL_WORKER_NODES
    }


def _failed_init_workers(history: List[Dict[str, Any]]) -> set[str]:
    """Возвращает множество worker-узлов с ошибкой инициализации.

    Args:
        history: История шагов графа.

    Returns:
        Набор имен агентов, у которых был `initialization_error=True`.
    """
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
    """Выбирает следующего доступного worker-агента.

    Args:
        called: Множество уже вызванных worker-узлов.
        failed: Множество узлов с ошибкой инициализации.
        exclude: Дополнительный набор узлов для исключения из выбора.
        prefer_not_called: Если `True`, не возвращает уже вызванные узлы.

    Returns:
        Имя выбранного worker-узла или `FINISH`, если подходящих кандидатов нет.
    """
    excluded = exclude or set()

    for node in ALL_WORKER_NODES:
        if node not in AVAILABLE_AGENTS or node in excluded or node in failed:
            continue
        if prefer_not_called and node in called:
            continue
        return node

    return "FINISH"


def _build_supervisor_system_prompt() -> str:
    """Формирует системный prompt для LLM-Supervisor."""
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

2) MixtureReactionAgent
- Выбирай только если в `mixture_input` есть непустой список веществ (`substances`).
- Используй его для анализа смеси, совместимости, возможных реакций, продуктов.
- Не выбирай, если данные по смеси отсутствуют.

3) SeparationMethodsAgent
- Выбирай, когда запрос связан с разделением/очисткой/выделением компонентов,
  либо когда нужен практический план разделения.
- Не выбирай его для общих теоретических вопросов, не связанных с разделением.

Критически важное правило:
- Если запрос пользователя является общим теоретическим, справочным или образовательным вопросом по химии,
  и для качественного ответа не требуется анализ конкретной молекулы, смеси или процесса разделения,
  НЕ выбирай worker-агента.
- В таком случае выбери FINISH и дай прямой полезный ответ пользователю.
- Не запрашивай SMILES для общих вопросов.
- Не запрашивай состав смеси, если пользователь не просит анализ смеси.

Политика маршрутизации:
- Сначала определи, можно ли ответить напрямую без worker-агентов.
- Если можно — выбери FINISH и дай содержательный ответ.
- Если нужен специализированный анализ, определи релевантных агентов.
- Если релевантный агент еще не вызывался и входные данные достаточны — выбери его.
- Если релевантных невызывавшихся агентов с достаточными данными не осталось — выбери FINISH.
- Если данных действительно недостаточно для релевантного агента — выбери FINISH и кратко перечисли, чего именно не хватает.
- Если в истории есть ошибки инициализации или выполнения агента, не выбирай этот агент повторно.

Критерии FINISH:
- Все релевантные агенты уже отработали, или
- Нет достаточных входных данных для дальнейших релевантных вызовов, или
- На вопрос можно качественно ответить без вызова worker-агентов, или
- Задача уже решена по истории.

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
  - если данных действительно не хватило, явно перечисли только недостающие данные.
"""


def _parse_supervisor_decision(result: Any) -> tuple[str, str]:
    """Извлекает `next_node` и `user_message` из ответа Supervisor LLM."""
    if not isinstance(result, dict):
        return "FINISH", ""

    payload = result.get("data", result)
    if not isinstance(payload, dict):
        return "FINISH", ""

    next_node = str(payload.get("next_node", "FINISH"))
    user_message = str(payload.get("user_message", "")).strip()
    return next_node, user_message


def _error_node(agent_name: str, error: Exception):
    """Создает fallback-узел для неинициализированного агента.

    Args:
        agent_name: Имя проблемного узла.
        error: Исключение, полученное при инициализации.

    Returns:
        Callable-узел, который возвращает событие ошибки в `history`.
    """

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
            ]
        }

    return node


def _timed_worker_node(
    agent_name: str,
    node_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    result_key: str,
):
    """Оборачивает worker-узел таймингом и унифицированным возвратом delta-state.

    Args:
        agent_name: Имя worker-агента для логирования.
        node_fn: Legacy-функция узла агента.
        result_key: Ключ в состоянии, куда агент пишет основной результат
            (`properties`, `mixture_reaction`, `separation_result`).

    Returns:
        Callable-узел, который:
        - замеряет время выполнения,
        - обрабатывает исключения без падения графа,
        - возвращает только новые изменения (`history` + `result_key`).
    """

    def wrapped(state: TeamState) -> Dict[str, Any]:
        started = perf_counter()
        logger.info("{}: запуск...", agent_name)

        working_state: Dict[str, Any] = dict(state)
        working_state["history"] = list(state.get("history", []))

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
                ]
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

        updates: Dict[str, Any] = {"history": [last_event]}
        if result_key in updated_state:
            updates[result_key] = updated_state[result_key]
        return updates

    return wrapped


structure_agent_instance = StructurePropertiesAgent(temperature=0.01)
structure_node_legacy = structure_agent_instance.as_node()

try:
    mixture_agent_instance = MixtureReactionAgent(model=MODEL_AGENT, temperature=0.0)
    mixture_node_legacy = mixture_agent_instance.as_node()
except Exception as exc:
    mixture_node_legacy = _error_node("MixtureReactionAgent", exc)

try:
    separation_agent_instance = SeparationMethodsAgent(
        temperature=0.01,
    )
    separation_node_legacy = separation_agent_instance.as_node()
except Exception as exc:
    separation_node_legacy = _error_node("SeparationMethodsAgent", exc)


def supervisor_node(state: TeamState) -> Dict[str, Any]:
    """Центральный узел Supervisor.

    Полный алгоритм:
    1. Читает текущую историю и проверяет лимиты шагов.
    2. Запрашивает LLM-решение (`next_node`, `user_message`).
    3. Валидирует решение и корректирует маршрут при повторных/невалидных узлах.
    4. Если выбран `FINISH`, формирует финальный ответ Supervisor в `history`.

    Args:
        state: Текущее состояние графа.

    Returns:
        Delta-обновление состояния:
        - всегда `next_worker`,
        - при завершении также финальное событие Supervisor в `history`.
    """
    started_total = perf_counter()
    history = list(state.get("history", []))

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
        return {"next_worker": "FINISH", "history": [event]}

    system_prompt = _build_supervisor_system_prompt()
    prompt = (
        f"Задача: {state.get('task', '')}\n"
        f"SMILES: {state.get('target_molecule', 'Не указан')}\n"
        f"mixture_input: {json.dumps(state.get('mixture_input', {}), ensure_ascii=False)}\n\n"
        f"Текущая история:\n{_history_as_text(history)}"
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
            return {"next_worker": "FINISH", "history": [event]}

        logger.warning("Supervisor: fallback-маршрут -> {}", fallback_next)
        logger.info("Supervisor: завершил шаг за {:.2f} c", perf_counter() - started_total)
        return {"next_worker": fallback_next}

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
                    "(SMILES, состав смеси, ограничения)."
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
        return {"next_worker": "FINISH", "history": [event]}

    logger.info("Supervisor: завершил шаг за {:.2f} c", perf_counter() - started_total)
    return {"next_worker": next_worker}


workflow = StateGraph(TeamState)
workflow.add_node("Supervisor", supervisor_node)
workflow.add_node(
    "StructureAnalyzer",
    _timed_worker_node("StructureAnalyzer", structure_node_legacy, "properties"),
)  # type: ignore[arg-type]
workflow.add_node(
    "MixtureReactionAgent",
    _timed_worker_node("MixtureReactionAgent", mixture_node_legacy, "mixture_reaction"),
)  # type: ignore[arg-type]
workflow.add_node(
    "SeparationMethodsAgent",
    _timed_worker_node("SeparationMethodsAgent", separation_node_legacy, "separation_result"),
)  # type: ignore[arg-type]

workflow.add_edge(START, "Supervisor")
workflow.add_edge("StructureAnalyzer", "Supervisor")
workflow.add_edge("MixtureReactionAgent", "Supervisor")
workflow.add_edge("SeparationMethodsAgent", "Supervisor")


def route_supervisor(state: TeamState):
    """Роутер условных ребер из Supervisor.

    Args:
        state: Текущее состояние графа.

    Returns:
        Имя следующего узла из `state.next_worker` или `FINISH` по умолчанию.
    """
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
