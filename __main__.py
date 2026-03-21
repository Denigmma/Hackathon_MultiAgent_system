"""Простая CLI-точка входа для MAS.

Сценарий работы:
1. Прочитать пользовательский запрос из терминала.
2. Запустить граф оркестратора с начальным состоянием.
3. Достать финальный ответ Supervisor из истории.
4. Вывести ответ в консоль.
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Dict, List

from dotenv import load_dotenv
from loguru import logger

load_dotenv()
warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.",
)

from app.MAS.orchestrator.agent_orchestrator import app

GRAPH_RECURSION_LIMIT = int(os.getenv("GRAPH_RECURSION_LIMIT", "25"))


def _extract_supervisor_answer(state: Dict[str, Any]) -> str:
    """Возвращает последний финальный ответ Supervisor из `state.history`.

    Args:
        state: Финальное состояние графа LangGraph.

    Returns:
        Текст ответа Supervisor. Если ответа нет, возвращает понятное fallback-сообщение.
    """
    history = state.get("history", [])
    if not isinstance(history, list):
        return "Supervisor не вернул корректную историю ответов."

    for event in reversed(history):
        if not isinstance(event, dict):
            continue
        if event.get("agent") != "Supervisor":
            continue

        output = event.get("output")
        if isinstance(output, dict):
            summary = output.get("summary") or output.get("prediction")
            if summary:
                return str(summary)
            return str(output)

        if output is not None:
            return str(output)

    return "Supervisor не вернул финальный ответ."


def _build_initial_state(user_input: str) -> Dict[str, Any]:
    """Собирает начальное состояние графа для одного пользовательского запроса."""
    return {
        "task": user_input,
        "target_molecule": "",
        "mixture_input": {},
        "separation_task": user_input,
        "history": [],
        "properties": {},
        "mixture_reaction": {},
        "separation_result": {},
        "next_worker": "",
    }


def main() -> int:
    """Запускает CLI-режим оркестратора.

    Returns:
        Код завершения процесса:
        - 0: успешный запуск,
        - 1: ошибка исполнения или пустой запрос.
    """
    try:
        print("--- ЗАПУСК СИСТЕМЫ ---\n")
        user_input = input("Введите запрос: ").strip()

        if not user_input:
            print("Пустой запрос.")
            return 1

        initial_state = _build_initial_state(user_input)
        result = app.invoke(  # type: ignore
            initial_state,
            {"recursion_limit": GRAPH_RECURSION_LIMIT},
        )

        answer = _extract_supervisor_answer(result)
        print(f"\nОтвет модели:\n{answer}")
        return 0

    except Exception as err:
        logger.error("Ошибка при запуске приложения: {}", err)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
