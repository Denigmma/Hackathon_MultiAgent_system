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
import re
import warnings
from typing import Any, Dict

from dotenv import load_dotenv
from loguru import logger
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.error")

load_dotenv()
warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.",
)

from src.MAS.orchestrator.agent_orchestrator import app

GRAPH_RECURSION_LIMIT = int(os.getenv("GRAPH_RECURSION_LIMIT", "25"))
SMILES_CANDIDATE_RE = re.compile(r"[A-Za-z0-9@+\-\[\]\(\)=#$\\/%.]+")


def _extract_supervisor_answer(state: Dict[str, Any]) -> str:
    """Возвращает последний финальный ответ Supervisor из ``state.history``."""
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

import re
from rdkit import Chem

SMILES_CANDIDATE_RE = re.compile(r"[A-Za-z0-9@+\-\[\]\(\)=#$\\/%.]+")
REACTION_SYMBOLS_RE = re.compile(r"(->|→|⇄|=)")
FORMULA_LIKE_RE = re.compile(r"^\d*[A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)*$")


def _looks_like_smiles(token: str) -> bool:
    token = token.strip(".,;:!?\"'")

    if not token:
        return False

    # Явные куски уравнений не трогаем
    if token in {"+", "->", "→", "=", "⇄"}:
        return False

    # Обычные химические формулы вида H2O, CO2, CH4, 2H2O и т.п. пропускаем
    if FORMULA_LIKE_RE.fullmatch(token):
        return False

    # Для SMILES обычно характерны спецсимволы/строчные атомы/скобки
    has_smiles_markers = any(ch in token for ch in "[]=#()/\\@") or any(ch in token for ch in "cnops")
    if not has_smiles_markers:
        return False

    return True

def _extract_smiles_from_text(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""

    # Если это похоже на уравнение реакции, лучше вообще не искать SMILES
    if REACTION_SYMBOLS_RE.search(text):
        return ""

    for token in SMILES_CANDIDATE_RE.findall(text):
        candidate = token.strip(".,;:!?\"'")
        if not _looks_like_smiles(candidate):
            continue

        mol = Chem.MolFromSmiles(candidate)
        if mol is None:
            continue

        return Chem.MolToSmiles(mol, canonical=True)

    return ""


def _build_initial_state(user_input: str) -> Dict[str, Any]:
    """Собирает начальное состояние графа для одного пользовательского запроса."""
    smiles = _extract_smiles_from_text(user_input)
    return {
        "task": user_input,
        "target_molecule": smiles,
        "synthesis_protocol_task": user_input,
        "literature_query": user_input,
        "history": [],
        "properties": {},
        "synthesis_protocol_result": {},
        "literature_result": {},
        "agent_interactions": {},
        "supervisor_trace": [],
        "next_worker": "",
    }


def main() -> int:
    """Запускает CLI-режим оркестратора."""
    try:
        print("--- ЗАПУСК СИСТЕМЫ ---\n")
        user_input = input("Введите запрос: ").strip()

        if not user_input:
            print("Пустой запрос.")
            return 1

        initial_state = _build_initial_state(user_input)
        result = app.invoke(  # type: ignore[arg-type]
            initial_state,
            {"recursion_limit": GRAPH_RECURSION_LIMIT},
        )

        answer = _extract_supervisor_answer(result)
        print(f"\nОтвет модели:\n{answer}")
        return 0

    except Exception as err:  # pragma: no cover - CLI safety net
        logger.exception("Ошибка при запуске приложения: {}", err)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
