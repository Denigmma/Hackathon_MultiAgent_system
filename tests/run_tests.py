from __future__ import annotations

import json
import os
import re
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from loguru import logger
from rdkit import Chem

load_dotenv()
warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.",
)

from src.MAS.orchestrator.agent_orchestrator import app

GRAPH_RECURSION_LIMIT = int(os.getenv("GRAPH_RECURSION_LIMIT", "25"))
SMILES_CANDIDATE_RE = re.compile(r"[A-Za-z0-9@+\-\[\]\(\)=#$\\/%.]+")


def extract_supervisor_answer(state: Dict[str, Any]) -> str:
    """Возвращает последний финальный ответ Supervisor из state.history."""
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


def extract_smiles_from_text(text: str) -> str:
    """Пытается достать первый валидный SMILES из текста."""
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


def build_initial_state(user_input: str) -> Dict[str, Any]:
    """Собирает начальное состояние графа для одного пользовательского запроса."""
    smiles = extract_smiles_from_text(user_input)
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


def run_single_query(question: str) -> Dict[str, Any]:
    """Запускает один вопрос через оркестратор и возвращает результат."""
    started_at = time.time()
    initial_state = build_initial_state(question)

    result_state = app.invoke(  # type: ignore[arg-type]
        initial_state,
        {"recursion_limit": GRAPH_RECURSION_LIMIT},
    )

    prediction = extract_supervisor_answer(result_state)
    finished_at = time.time()

    return {
        "prediction": prediction,
        "target_molecule": initial_state.get("target_molecule", ""),
        "latency_sec": round(finished_at - started_at, 3),
        "raw_state": result_state,  # можно убрать, если JSON становится слишком большим
    }


def evaluate_dataset(
    input_path: str = "test_data.json",
    output_path: str = "test_results.json",
    limit: int | None = None,
    save_raw_state: bool = False,
) -> None:
    """Прогоняет весь датасет через систему и сохраняет результаты в JSON."""
    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.exists():
        raise FileNotFoundError(f"Файл не найден: {input_file}")

    with input_file.open("r", encoding="utf-8") as f:
        dataset: List[Dict[str, Any]] = json.load(f)

    if limit is not None:
        dataset = dataset[:limit]

    results: List[Dict[str, Any]] = []
    total = len(dataset)

    logger.info("Запускаю тестирование на {} примерах", total)

    for idx, sample in enumerate(dataset, start=1):
        category = sample.get("category", "")
        question = sample.get("question", "")
        expected_answer = sample.get("answer", "")

        logger.info("[{}/{}] Обработка вопроса: {}", idx, total, question)

        record: Dict[str, Any] = {
            "id": idx,
            "category": category,
            "question": question,
            "expected_answer": expected_answer,
        }

        try:
            result = run_single_query(question)
            record["prediction"] = result["prediction"]
            record["target_molecule"] = result["target_molecule"]
            record["latency_sec"] = result["latency_sec"]
            record["status"] = "ok"

            if save_raw_state:
                record["raw_state"] = result["raw_state"]

        except Exception as err:
            logger.exception("Ошибка на примере {}: {}", idx, err)
            record["prediction"] = ""
            record["target_molecule"] = ""
            record["latency_sec"] = None
            record["status"] = "error"
            record["error"] = str(err)

        results.append(record)

        # сохраняем промежуточно после каждого примера
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info("Готово. Результаты сохранены в {}", output_file.resolve())


if __name__ == "__main__":
    evaluate_dataset(
        input_path="test_data.json",
        output_path="test_results.json",
        limit=None,           # например, 10 для быстрого теста
        save_raw_state=False, # True если нужен полный state
    )