from __future__ import annotations

import json
import random
import os
from typing import Any, Dict, List

from dotenv import load_dotenv

load_dotenv()

from src.MAS.orchestrator.agent_orchestrator import app

INPUT_JSON = "test_data.json"
OUTPUT_JSON = "MAS_answers.json"
N_QUESTIONS = 1
GRAPH_RECURSION_LIMIT = int(os.getenv("GRAPH_RECURSION_LIMIT", "25"))


def _extract_supervisor_answer(state: Dict[str, Any]) -> str:
    history = state.get("history", [])

    if not isinstance(history, list):
        return "Ошибка: нет истории"

    for event in reversed(history):
        if not isinstance(event, dict):
            continue
        if event.get("agent") != "Supervisor":
            continue

        output = event.get("output")

        if isinstance(output, dict):
            return str(output.get("summary") or output.get("prediction") or output)

        if output is not None:
            return str(output)

    return "Нет ответа от Supervisor"


def _build_initial_state(user_input: str) -> Dict[str, Any]:
    return {
        "task": user_input,
        "target_molecule": "",
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


def main():
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    sample = random.sample(data, min(N_QUESTIONS, len(data)))

    results = []

    for i, item in enumerate(sample, 1):
        question = item.get("question", "")

        print(f"\n[{i}/15] Вопрос: {question}")

        state = _build_initial_state(question)

        try:
            result = app.invoke(
                state,
                {"recursion_limit": GRAPH_RECURSION_LIMIT},
            )

            model_answer = _extract_supervisor_answer(result)

        except Exception as e:
            model_answer = f"ERROR: {str(e)}"

        new_item = dict(item)
        new_item["model_answer"] = model_answer

        results.append(new_item)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nГотово. Сохранено в {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
