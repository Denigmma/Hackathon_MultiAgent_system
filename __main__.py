"""Корневая точка входа приложения.

Файл запускает оркестратор и выводит пользователю только итоговый ответ модели.
"""

from __future__ import annotations

import os
import warnings

from dotenv import load_dotenv
from loguru import logger

load_dotenv()
warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.",
)

from app.MAS.orchestrator.agent_orchestrator import app

GRAPH_RECURSION_LIMIT = int(os.getenv("GRAPH_RECURSION_LIMIT", "25"))


def _extract_model_answer(result: dict) -> str:
    """Достает финальный ответ модели из state.history.

    Приоритет:
    1. Последний ответ Supervisor (output.summary / output.prediction).
    2. Последний output любого агента.
    3. Сообщение-заглушка.
    """

    history = result.get("history", [])
    if not isinstance(history, list):
        return "Модель не вернула корректную историю ответов."

    # 1) Предпочитаем ответ Supervisor
    for item in reversed(history):
        if not isinstance(item, dict):
            continue
        if item.get("agent") != "Supervisor":
            continue

        output = item.get("output")
        if isinstance(output, dict):
            summary = output.get("summary") or output.get("prediction")
            if summary:
                return str(summary)
            return str(output)
        if output is not None:
            return str(output)

    return "Supervisor не вернул финальный ответ."


if __name__ == "__main__":
    try:
        print("--- ЗАПУСК СИСТЕМЫ ---\n")
        user_input = input("Введите запрос: ").strip()

        if not user_input:
            print("Пустой запрос.")
            raise SystemExit(1)

        initial_state = {
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

        result = app.invoke(  # type: ignore
            initial_state,
            {"recursion_limit": GRAPH_RECURSION_LIMIT},
        )
        answer = _extract_model_answer(result)
        print(f"\nОтвет модели:\n{answer}")

    except Exception as err:
        logger.error(f"Ошибка при запуске приложения: {err}")
        raise SystemExit(1)
