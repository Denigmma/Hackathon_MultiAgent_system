"""Простая CLI-точка входа для MAS.

Сценарий работы:
1. Прочитать пользовательский запрос из терминала.
2. Подготовить начальное состояние графа, включая попытку извлечь SMILES из текста.
3. Запустить граф оркестратора с начальным состоянием.
4. Достать финальный ответ Supervisor из истории.
5. Вывести ответ в консоль.
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Dict

from dotenv import load_dotenv
from loguru import logger

load_dotenv()
warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.",
)

from src.MAS.orchestrator.agent_orchestrator import app
from src.mas_runtime import (
    GRAPH_RECURSION_LIMIT,
    build_initial_state,
    extract_supervisor_answer,
)


def main() -> int:
    """Запускает CLI-режим оркестратора."""
    try:
        print("--- ЗАПУСК СИСТЕМЫ ---\n")
        user_input = input("Введите запрос: ").strip()

        if not user_input:
            print("Пустой запрос.")
            return 1

        initial_state = build_initial_state(user_input)
        result = app.invoke(  # type: ignore[arg-type]
            initial_state,
            {"recursion_limit": GRAPH_RECURSION_LIMIT},
        )

        answer = extract_supervisor_answer(result)
        print(f"\nОтвет модели:\n{answer}")
        return 0

    except Exception as err:  # pragma: no cover - CLI safety net
        logger.exception("Ошибка при запуске приложения: {}", err)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
