from __future__ import annotations

import json
import operator
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Sequence, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from loguru import logger
from pydantic import BaseModel, Field
from rdkit import Chem, RDLogger

# Убираем шум RDKit parse ошибок в консоли
RDLogger.DisableLog("rdApp.error")

# Поддержка запуска файла напрямую из папки orchestrator
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.MAS.agents.properties_agent import (  # noqa: E402
    StructurePropertiesAgent as PropertiesAgentImpl,
)
from app.MAS.agents.reagent_selection_agent import (  # noqa: E402
    ReagentSelectionAgent,
)
from app.MAS.agents.structure_agent import (  # noqa: E402
    StructurePropertiesAgent as StructureAgentImpl,
)

load_dotenv()


@dataclass(frozen=True)
class ModelPromptBinding:
    role: str
    description: str
    model: str
    temperature: float
    system_prompt_key: str
    user_prompt_key: str | None = None


class AgentState(TypedDict, total=False):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str


class Route(BaseModel):
    next: str = Field(
        description=(
            "Имя следующего агента из списка или FINISH, если финальный ответ уже готов."
        )
    )


AGENTS = [
    "Structure_Agent",
    "Properties_Agent",
    "Reagent_Selection_Agent",
]

SMILES_TOKEN_RE = re.compile(r"[A-Za-z0-9@+\-\[\]\(\)=#$\\/.]{2,160}")
PROMPTS_PATH = PROJECT_ROOT / "prompts.json"


def _safe_json(payload: object) -> str:
    try:
        return json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    except Exception:
        return str(payload)


def _content_to_text(content: object) -> str:
    return content if isinstance(content, str) else str(content)


def _latest_human_text(messages: Sequence[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return _content_to_text(msg.content)
    return ""


def _latest_context(messages: Sequence[BaseMessage], limit: int = 6) -> str:
    chunks: list[str] = []
    for msg in messages[-limit:]:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        chunks.append(f"{role}: {_content_to_text(msg.content)}")
    return "\n".join(chunks)


def _looks_like_smiles_token(token: str) -> bool:
    has_symbol_or_digit = any(ch.isdigit() for ch in token) or any(
        ch in token for ch in "[]()=#@+-/\\."
    )
    uppercase_word_like = token.isupper() and 2 <= len(token) <= 40
    return has_symbol_or_digit or uppercase_word_like


def _extract_valid_smiles(text: str) -> str | None:
    if not text:
        return None

    stripped = text.strip()
    if " " not in stripped and len(stripped) <= 160:
        if Chem.MolFromSmiles(stripped) is not None:
            return stripped

    for token in SMILES_TOKEN_RE.findall(text):
        if not _looks_like_smiles_token(token):
            continue
        if Chem.MolFromSmiles(token) is not None:
            return token

    return None


def _extract_smiles_from_messages(messages: Sequence[BaseMessage]) -> str | None:
    for msg in reversed(messages):
        candidate = _extract_valid_smiles(_content_to_text(msg.content))
        if candidate:
            return candidate
    return None


def _agent_has_run(messages: Sequence[BaseMessage], agent_name: str) -> bool:
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.name == agent_name:
            return True
    return False


def _is_synthesis_request(text: str) -> bool:
    query = (text or "").lower()
    keywords = [
        "синтез",
        "маршрут",
        "реагент",
        "реакц",
        "прекурсор",
        "катализ",
        "synthesis",
        "reaction",
        "reagent",
        "route",
    ]
    return any(k in query for k in keywords)


def _cleanup_agent_text(text: str) -> str:
    cleaned = text.strip()
    if " result:\n" in cleaned:
        cleaned = cleaned.split(" result:\n", 1)[1].strip()

    try:
        obj = json.loads(cleaned)
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        return cleaned


def _load_prompt_catalog(path: Path) -> tuple[dict[str, str], dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Не найден файл промптов: {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    prompts = raw.get("prompts")
    models = raw.get("models")

    if not isinstance(prompts, dict) or not isinstance(models, dict):
        raise ValueError("prompts.json должен содержать объекты 'prompts' и 'models'.")

    return prompts, models


def _build_binding(role: str, models_map: dict[str, object]) -> ModelPromptBinding:
    config = models_map.get(role)
    if not isinstance(config, dict):
        raise ValueError(f"В prompts.json отсутствует описание модели '{role}'.")

    env_model_key = str(config.get("env_model", "")).strip()
    default_model = str(config.get("default_model", "")).strip()
    model = os.getenv(env_model_key, default_model) if env_model_key else default_model

    if not model:
        raise ValueError(f"Для роли '{role}' не задана модель.")

    system_prompt_key = str(config.get("system_prompt", "")).strip()
    user_prompt_key = str(config.get("user_prompt", "")).strip() or None

    if not system_prompt_key:
        raise ValueError(f"Для роли '{role}' не задан system_prompt.")

    return ModelPromptBinding(
        role=role,
        description=str(config.get("description", role)),
        model=model,
        temperature=float(config.get("temperature", 0.0)),
        system_prompt_key=system_prompt_key,
        user_prompt_key=user_prompt_key,
    )


def _build_llm(model_name: str, temperature: float) -> ChatOpenAI:
    api_key = os.getenv("OPEN_ROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Не найден API ключ. Установите OPEN_ROUTER_API_KEY или OPENAI_API_KEY."
        )

    return ChatOpenAI(
        api_key=api_key,
        base_url=os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"),
        model=model_name,
        temperature=temperature,
    )


PROMPTS, MODELS_MAP = _load_prompt_catalog(PROMPTS_PATH)
SUPERVISOR_BINDING = _build_binding("supervisor", MODELS_MAP)
FINAL_BINDING = _build_binding("final_answer", MODELS_MAP)

for binding in (SUPERVISOR_BINDING, FINAL_BINDING):
    if binding.system_prompt_key not in PROMPTS:
        raise ValueError(
            f"В prompts.json отсутствует промпт '{binding.system_prompt_key}' для роли '{binding.role}'."
        )
    if binding.user_prompt_key and binding.user_prompt_key not in PROMPTS:
        raise ValueError(
            f"В prompts.json отсутствует промпт '{binding.user_prompt_key}' для роли '{binding.role}'."
        )

supervisor_llm = _build_llm(SUPERVISOR_BINDING.model, SUPERVISOR_BINDING.temperature)
final_answer_llm = _build_llm(FINAL_BINDING.model, FINAL_BINDING.temperature)

SUPERVISOR_PROMPT = PROMPTS[SUPERVISOR_BINDING.system_prompt_key]
FINAL_SYSTEM_PROMPT = PROMPTS[FINAL_BINDING.system_prompt_key]
FINAL_USER_PROMPT = PROMPTS[FINAL_BINDING.user_prompt_key] if FINAL_BINDING.user_prompt_key else ""

# Инициализируем существующих агентов из app/MAS/agents
structure_agent = StructureAgentImpl(model=supervisor_llm)
properties_agent = PropertiesAgentImpl(llm=supervisor_llm)
reagent_selection_agent = ReagentSelectionAgent(model=supervisor_llm)


def describe_runtime_bindings() -> str:
    return "\n".join(
        [
            "Конфигурация модель -> промпт:",
            (
                f"- {SUPERVISOR_BINDING.role}: model='{SUPERVISOR_BINDING.model}', "
                f"prompt='{SUPERVISOR_BINDING.system_prompt_key}'"
            ),
            (
                f"- {FINAL_BINDING.role}: model='{FINAL_BINDING.model}', "
                f"prompts='{FINAL_BINDING.system_prompt_key}'"
                + (
                    f" + '{FINAL_BINDING.user_prompt_key}'"
                    if FINAL_BINDING.user_prompt_key
                    else ""
                )
            ),
        ]
    )


def supervisor_node(state: AgentState) -> dict:
    logger.info("Supervisor: анализирую историю и выбираю следующего агента.")
    messages = list(state.get("messages", []))
    latest_user = _latest_human_text(messages)

    smiles = _extract_smiles_from_messages(messages)
    if not smiles:
        if _agent_has_run(messages, "Reagent_Selection_Agent"):
            logger.info(
                "Supervisor: валидный SMILES не найден, Reagent_Selection_Agent уже выполнен -> FINISH"
            )
            return {"next": "FINISH"}

        if _is_synthesis_request(latest_user):
            logger.info(
                "Supervisor: SMILES не найден, но запрос похож на синтетический -> Reagent_Selection_Agent"
            )
            return {"next": "Reagent_Selection_Agent"}

        logger.info(
            "Supervisor: запрос не требует спец-агентов по синтезу -> FINISH"
        )
        return {"next": "FINISH"}

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SUPERVISOR_PROMPT),
            MessagesPlaceholder(variable_name="messages"),
        ]
    ).partial(agents=", ".join(AGENTS))

    chain = prompt | supervisor_llm.with_structured_output(Route)
    result = chain.invoke({"messages": messages})

    candidate = result.next.strip() if isinstance(result.next, str) else ""
    if candidate not in {*AGENTS, "FINISH"}:
        logger.warning(
            "Supervisor вернул неизвестный узел '%s'. Принудительно выбираю FINISH.",
            candidate,
        )
        candidate = "FINISH"

    logger.info(f"Supervisor выбрал: {candidate}")
    return {"next": candidate}


def structure_node(state: AgentState) -> dict:
    messages = list(state.get("messages", []))
    logger.info("Structure_Agent: запускаю анализ структуры.")

    smiles = _extract_smiles_from_messages(messages)
    if not smiles:
        result = {
            "status": "NO_VALID_SMILES",
            "error": "В контексте нет валидного SMILES. Агент пропущен.",
            "hint": "Передайте валидный SMILES, например: CCO",
        }
    else:
        try:
            result = structure_agent.run(smiles)
        except Exception as exc:
            result = {"error": f"Structure_Agent failed: {exc}"}

    return {
        "messages": [
            AIMessage(
                name="Structure_Agent",
                content=f"Structure_Agent result:\n{_safe_json(result)}",
            )
        ]
    }


def properties_node(state: AgentState) -> dict:
    messages = list(state.get("messages", []))
    logger.info("Properties_Agent: запускаю оценку свойств.")

    smiles = _extract_smiles_from_messages(messages)
    if not smiles:
        result = {
            "status": "NO_VALID_SMILES",
            "error": "В контексте нет валидного SMILES. Агент пропущен.",
            "hint": "Передайте валидный SMILES, например: CCO",
        }
    else:
        try:
            result = properties_agent.run(smiles)
        except Exception as exc:
            result = {"error": f"Properties_Agent failed: {exc}"}

    return {
        "messages": [
            AIMessage(
                name="Properties_Agent",
                content=f"Properties_Agent result:\n{_safe_json(result)}",
            )
        ]
    }


def reagent_selection_node(state: AgentState) -> dict:
    messages = list(state.get("messages", []))
    reaction_description = _latest_human_text(messages) or _latest_context(messages)
    logger.info("Reagent_Selection_Agent: подбираю реагенты и проверяю доступность.")

    try:
        result = reagent_selection_agent.run(reaction_description)
    except Exception as exc:
        result = {"error": f"Reagent_Selection_Agent failed: {exc}"}

    return {
        "messages": [
            AIMessage(
                name="Reagent_Selection_Agent",
                content=f"Reagent_Selection_Agent result:\n{_safe_json(result)}",
            )
        ]
    }


def _format_agent_trace(agent_trace: Sequence[tuple[str, str]]) -> str:
    if not agent_trace:
        return "Агенты не вызывались: Supervisor завершил работу напрямую."

    lines: list[str] = []
    for idx, (node_name, content) in enumerate(agent_trace, start=1):
        lines.append(f"{idx}. {node_name}:\n{content}")
    return "\n\n".join(lines)


def _generate_final_answer(user_query: str, agent_trace: Sequence[tuple[str, str]]) -> str:
    trace_text = _format_agent_trace(agent_trace)

    if not FINAL_USER_PROMPT:
        return trace_text

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", FINAL_SYSTEM_PROMPT),
            ("user", FINAL_USER_PROMPT),
        ]
    )

    chain = prompt | final_answer_llm

    try:
        response = chain.invoke({"user_query": user_query, "agent_trace": trace_text})
        answer = _content_to_text(response.content).strip()
        return answer or "Не удалось сформировать финальный ответ."
    except Exception as exc:
        logger.error(f"Ошибка генерации финального ответа LLM: {exc}")
        return (
            "Не удалось сгенерировать финальный ответ LLM. "
            "Ниже рабочий контекст от агентов:\n\n"
            + trace_text
        )


def run_query(user_query: str, recursion_limit: int = 15) -> str:
    initial_input = {"messages": [HumanMessage(content=user_query)]}
    agent_trace: list[tuple[str, str]] = []

    for event in app.stream(initial_input, {"recursion_limit": recursion_limit}):  # type: ignore[arg-type]
        if "__end__" in event:
            continue

        supervisor_payload = event.get("Supervisor")
        if isinstance(supervisor_payload, dict):
            next_node = supervisor_payload.get("next")
            if isinstance(next_node, str):
                print(f"[Supervisor] -> {next_node}")

        for node_name, payload in event.items():
            if node_name in {"Supervisor", "__end__"}:
                continue
            if not isinstance(payload, dict):
                continue

            messages = payload.get("messages")
            if not isinstance(messages, list) or not messages:
                continue

            last_message = messages[-1]
            if isinstance(last_message, BaseMessage):
                raw_text = _content_to_text(last_message.content)
            else:
                raw_text = str(last_message)

            cleaned = _cleanup_agent_text(raw_text)
            agent_trace.append((node_name, cleaned))
            print(f"[{node_name}] выполнен")

    return _generate_final_answer(user_query, agent_trace)


def interactive_cli() -> None:
    print("Запуск Supervisor-оркестратора (интерактивный режим).")
    print(describe_runtime_bindings())
    print("Введите запрос и нажмите Enter. Для выхода: exit/quit/выход")

    while True:
        try:
            user_query = input("\nВы> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nВыход.")
            break

        if not user_query:
            continue

        if user_query.lower() in {"exit", "quit", "q", "выход"}:
            print("Выход.")
            break

        print("\nМаршрут:")
        answer = run_query(user_query)
        print("\nОтвет:")
        print(answer)


def main_cli(argv: Sequence[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])

    if args:
        query = " ".join(args).strip()
        if not query:
            print("Пустой запрос. Передайте текст после имени скрипта.")
            return 1

        print(describe_runtime_bindings())
        print("\nМаршрут:")
        answer = run_query(query)
        print("\nОтвет:")
        print(answer)
        return 0

    interactive_cli()
    return 0


workflow = StateGraph(AgentState)

workflow.add_node("Supervisor", supervisor_node)
workflow.add_node("Structure_Agent", structure_node)
workflow.add_node("Properties_Agent", properties_node)
workflow.add_node("Reagent_Selection_Agent", reagent_selection_node)

for agent_name in AGENTS:
    workflow.add_edge(agent_name, "Supervisor")

route_map = {name: name for name in AGENTS}
route_map["FINISH"] = END

workflow.add_conditional_edges("Supervisor", lambda x: x["next"], route_map)  # type: ignore
workflow.set_entry_point("Supervisor")

app = workflow.compile()


if __name__ == "__main__":
    print("Запуск перенесен в корневой __main__.py")
    print("Используйте: poetry run python3 __main__.py")
