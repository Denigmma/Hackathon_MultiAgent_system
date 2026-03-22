from __future__ import annotations

import json
import os
import random
import re
from typing import Any, Dict, List

from dotenv import load_dotenv
from rdkit import Chem

load_dotenv()

from src.MAS.orchestrator.agent_orchestrator import app

INPUT_JSON = "test_data.json"
OUTPUT_JSON = "MAS_answers.json"
N_QUESTIONS = int(os.getenv("N_QUESTIONS", "15"))
GRAPH_RECURSION_LIMIT = int(os.getenv("GRAPH_RECURSION_LIMIT", "25"))
SMILES_CANDIDATE_RE = re.compile(r"[A-Za-z0-9@+\-\[\]\(\)=#$\\/%.]+")


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


def _build_initial_state(user_input: str) -> Dict[str, Any]:
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


def main() -> None:
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    sample = random.sample(data, min(N_QUESTIONS, len(data)))
    results = []

    for i, item in enumerate(sample, 1):
        question = str(item.get("question", ""))
        print(f"\n[{i}/{len(sample)}] Вопрос: {question}")

        state = _build_initial_state(question)

        try:
            result = app.invoke(
                state,
                {"recursion_limit": GRAPH_RECURSION_LIMIT},
            )
            model_answer = _extract_supervisor_answer(result)
        except Exception as exc:
            model_answer = f"ERROR: {exc}"

        new_item = dict(item)
        new_item["model_answer"] = model_answer
        results.append(new_item)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nГотово. Сохранено в {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
