from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

from dotenv import load_dotenv
from rdkit import Chem

load_dotenv()

from src.MAS.orchestrator.agent_orchestrator import app, _format_worker_summary

GRAPH_RECURSION_LIMIT = int(os.getenv("GRAPH_RECURSION_LIMIT", "25"))
SMILES_CANDIDATE_RE = re.compile(r"[A-Za-z0-9@+\-\[\]\(\)=#$\\/%.]+")
AGENT_DISPLAY_NAMES = {
    "Supervisor": "Главный агент",
    "StructureAnalyzer": "агента по анализу структуры",
    "SynthesisProtocolSearchAgent": "агента по поиску методик синтеза",
    "LiteratureRAGAgent": "агента по литературному поиску",
}
WORKER_RESPONSE_NAMES = {
    "StructureAnalyzer": "Агент по анализу структуры",
    "SynthesisProtocolSearchAgent": "Агент по поиску методик синтеза",
    "LiteratureRAGAgent": "Агент по литературному поиску",
}


@dataclass(frozen=True)
class MASProgressEvent:
    agent: str
    content: str
    kind: str
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MASRunResult:
    answer: str
    final_state: Dict[str, Any]
    events: List[MASProgressEvent]


def extract_supervisor_answer(state: Dict[str, Any]) -> str:
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


def extract_smiles_from_text(text: str) -> str:
    """Пытается достать первый валидный SMILES из пользовательского текста."""
    text = (text or "").strip()
    if not text:
        return ""

    for token in SMILES_CANDIDATE_RE.findall(text):
        candidate = token.strip(".,;:!?\"'")
        if not candidate:
            continue
        if len(candidate) < 2:
            continue
        if candidate[0] in "-()":
            continue
        if not re.search(r"[A-Za-z]", candidate):
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


def _display_agent_name(agent_name: str) -> str:
    return AGENT_DISPLAY_NAMES.get(agent_name, agent_name)


def _worker_response_name(agent_name: str) -> str:
    return WORKER_RESPONSE_NAMES.get(agent_name, agent_name)


def _worker_event_to_text(event: Dict[str, Any]) -> str:
    agent_name = str(event.get("agent", "") or "").strip()
    output = event.get("output")
    if not agent_name:
        return "Система получила промежуточный результат."

    if isinstance(output, dict):
        if agent_name == "LiteratureRAGAgent":
            search_query = str(output.get("search_query") or "").strip()
            answer = output.get("answer")
            if search_query and isinstance(answer, str) and answer.strip():
                return (
                    "Агент по литературному поиску сформировал веб-запрос: "
                    f"{search_query}. Итог поиска: {answer.strip()}"
                )

        summary = output.get("summary")
        if isinstance(summary, str) and summary.strip():
            return summary.strip()

        prediction = output.get("prediction")
        if isinstance(prediction, str) and prediction.strip():
            return prediction.strip()

        answer = output.get("answer")
        if isinstance(answer, str) and answer.strip():
            return f"{_worker_response_name(agent_name)} ответил: {answer.strip()}"

    return f"{_worker_response_name(agent_name)} ответил: {_format_worker_summary(event)}"


def _normalize_supervisor_trace_messages(entry: Dict[str, Any]) -> List[str]:
    decision = str(entry.get("decision") or "").strip()
    message = str(entry.get("message") or "").strip()

    if decision == "FINISH":
        return ["Главный агент завершил координацию и формирует итоговый ответ."]

    invoked_agent = _display_agent_name(decision)
    messages = [f"Главный агент вызвал {invoked_agent}."]
    if message:
        messages.append(f"Постановка задачи от главного агента: {message}")
    return messages


def run_mas_query(
    user_input: str,
    recursion_limit: int | None = None,
    on_event: Callable[[MASProgressEvent], None] | None = None,
) -> MASRunResult:
    events: List[MASProgressEvent] = []
    final_state: Dict[str, Any] = build_initial_state(user_input)
    seen_trace = 0
    seen_history = 0
    limit = GRAPH_RECURSION_LIMIT if recursion_limit is None else int(recursion_limit)

    def emit(event: MASProgressEvent) -> None:
        events.append(event)
        if on_event is not None:
            on_event(event)

    for current_state in app.stream(
        final_state,
        {"recursion_limit": limit},
        stream_mode="values",
    ):
        if isinstance(current_state, dict):
            final_state = current_state
        else:
            continue

        trace = final_state.get("supervisor_trace", [])
        if isinstance(trace, list) and len(trace) > seen_trace:
            for item in trace[seen_trace:]:
                if not isinstance(item, dict):
                    continue
                normalized_messages = _normalize_supervisor_trace_messages(item)
                for content in normalized_messages:
                    if not content:
                        continue
                    emit(
                        MASProgressEvent(
                            agent="Supervisor",
                            content=content,
                            kind=str(item.get("reason") or "supervisor_update"),
                            payload=item,
                        )
                    )
            seen_trace = len(trace)

        history = final_state.get("history", [])
        if isinstance(history, list) and len(history) > seen_history:
            for item in history[seen_history:]:
                if not isinstance(item, dict):
                    continue
                agent_name = str(item.get("agent") or "").strip()
                if not agent_name or agent_name == "Supervisor":
                    continue
                emit(
                    MASProgressEvent(
                        agent=agent_name,
                        content=_worker_event_to_text(item),
                        kind="worker_result",
                        payload=item,
                    )
                )
            seen_history = len(history)

    return MASRunResult(
        answer=extract_supervisor_answer(final_state),
        final_state=final_state,
        events=events,
    )
