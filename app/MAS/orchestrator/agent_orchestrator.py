import os
import json
from typing import TypedDict, List, Any
from langgraph.graph import StateGraph, START, END
from loguru import logger
from app.llm_client import VseGPTConfig, VseGPTWrapper
from app.MAS.agents.properties_agent import StructurePropertiesAgent
from app.MAS.agents.reaction_products_agent import MixtureReactionAgent
from app.MAS.agents.decomposition_products_agent import SeparationMethodsAgent

API_KEY = os.getenv("VSEGPT_API_KEY", "")
MODEL_ORCHESTOR = os.getenv("MODEL_ORCHESTOR", "openai/gpt-5.4-nano-thinking-xhigh")
MODEL_AGENT = os.getenv("MODEL_AGENT", "openai/gpt-5.4-nano-thinking-xhigh")
BASE_URL = os.getenv("URL", "https://api.vsegpt.ru/v1")


config = VseGPTConfig(
    base_url=BASE_URL,
    api_key=API_KEY,
    model=MODEL_ORCHESTOR,
)
llm = VseGPTWrapper(config)

# Список доступных агентов для маршрутизации
AVAILABLE_AGENTS = [
    agent.strip()
    for agent in str(
        os.getenv(
            "AVAILABLE_AGENTS",
            "MixtureReactionAgent,SeparationMethodsAgent,StructureAnalyzer,FINISH",
        )
    ).split(",")
    if agent.strip()
]


# --- 1. СТРУКТУРА СОСТОЯНИЯ ---
class TeamState(TypedDict):
    task: str
    target_molecule: str = None  # type: ignore # SMILES-строка, которую ждет агент
    mixture_input: dict
    separation_task: str
    history: List[Any]  # Меняем на Any, так как агент пишет сюда dict, а не str
    properties: dict  # Сюда агент запишет результат
    mixture_reaction: dict
    separation_result: dict
    next_worker: str


# --- 2. ИНИЦИАЛИЗАЦИЯ УЗЛОВ-АГЕНТОВ ---
# Создаем экземпляры агентов
structure_agent_instance = StructurePropertiesAgent(llm=llm, temperature=0.3)
structure_analyzer_node = structure_agent_instance.as_node()


def _error_node(agent_name: str, error: Exception):
    """
    Возвращает fallback-узел, если конкретный агент не удалось инициализировать.
    """

    def node(state: TeamState):
        logger.error(f"{agent_name} не инициализирован: {error}")
        state.setdefault("history", []).append(
            {"agent": agent_name, "output": {"error": str(error)}}
        )
        return state

    return node


try:
    mixture_agent_instance = MixtureReactionAgent(model=MODEL_AGENT, temperature=0.0)
    mixture_reaction_node = mixture_agent_instance.as_node()
except Exception as exc:
    mixture_reaction_node = _error_node("MixtureReactionAgent", exc)

try:
    separation_agent_instance = SeparationMethodsAgent(
        model=MODEL_AGENT, temperature=0.0
    )
    separation_methods_node = separation_agent_instance.as_node()
except Exception as exc:
    separation_methods_node = _error_node("SeparationMethodsAgent", exc)


# --- 3. УЗЕЛ-СУПЕРВИЗОР ---
def supervisor_node(state: TeamState):
    logger.info("Supervisor: Анализирую историю и принимаю решение...")

    # Безопасная сборка истории (так как теперь там могут быть и строки, и словари)
    history_lines = []
    for item in state.get("history", []):
        if isinstance(item, str):
            history_lines.append(item)
        elif isinstance(item, dict):
            # Если это словарь от нашего StructurePropertiesAgent
            history_lines.append(json.dumps(item, ensure_ascii=False))

    history_text = (
        "\n".join(history_lines) if history_lines else "История пуста (начало работы)."
    )

    # Настраиваем Супервизора на возврат строгого JSON
    system_prompt = """Ты — координатор химической лаборатории.
Проанализируй задачу и историю работы, чтобы решить, что делать дальше.
Верни строго JSON-объект с одним ключом "next_node".

Доступные узлы (next_node):
1. "StructureAnalyzer" — если нужно проанализировать молекулу (SMILES) и получить её свойства, а в истории этих данных еще нет.
2. "MixtureReactionAgent" — если нужно оценить возможные продукты реакции в смеси веществ.
3. "SeparationMethodsAgent" — если нужно предложить методы разделения смеси.
4. "FINISH" — если задача полностью выполнена.
"""

    prompt = f"Задача: {state['task']}\nSMILES: {state.get('target_molecule', 'Не указан')}\n\nТекущая история работы:\n{history_text}"

    try:
        result = llm.ask(
            prompt=prompt, system_prompt=system_prompt, json_mode=True, temperature=0.0
        )
    except Exception as exc:
        logger.error(f"Ошибка вызова Supervisor LLM: {exc}")
        has_properties = bool(state.get("properties"))
        fallback_next = "FINISH" if has_properties else "StructureAnalyzer"
        logger.warning(
            f"Переход в fallback-маршрут из-за ошибки LLM: next_worker={fallback_next}"
        )
        return {"next_worker": fallback_next}

    if isinstance(result, dict) and "next_node" in result:
        next_worker = result["next_node"]
    else:
        logger.warning("Ошибка парсинга решения Супервизора, завершаю работу.")
        next_worker = "FINISH"

    if next_worker not in AVAILABLE_AGENTS:
        logger.warning(
            f"Супервизор вернул неизвестного агента '{next_worker}', меняю на FINISH."
        )
        next_worker = "FINISH"

    logger.info(f"Supervisor решил: передаю задачу -> {next_worker}")
    return {"next_worker": next_worker}


# --- 4. СБОРКА И ЗАПУСК ГРАФА ---

workflow = StateGraph(TeamState)

workflow.add_node("Supervisor", supervisor_node)
workflow.add_node("StructureAnalyzer", structure_analyzer_node)  # type: ignore
workflow.add_node("MixtureReactionAgent", mixture_reaction_node)  # type: ignore
workflow.add_node("SeparationMethodsAgent", separation_methods_node)  # type: ignore

# Точка входа
workflow.add_edge(START, "Supervisor")

# Агенты всегда возвращает результат Супервизору
workflow.add_edge("StructureAnalyzer", "Supervisor")
workflow.add_edge("MixtureReactionAgent", "Supervisor")
workflow.add_edge("SeparationMethodsAgent", "Supervisor")


def route_supervisor(state: TeamState):
    return state["next_worker"]


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
