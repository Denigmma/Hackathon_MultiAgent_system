import os
import json
from typing import TypedDict, List, Any
from langgraph.graph import StateGraph, START, END
from loguru import logger
from app.llm_client import VseGPTConfig, VseGPTWrapper
from app.Agents.agents.properties_agent import StructurePropertiesAgent

API_KEY = os.getenv("VSEGPT_API_KEY", "")
MODEL_ORCHESTOR = os.getenv("MODEL_ORCHESTOR", "openai/gpt-5.4-nano-thinking-xhigh")
BASE_URL = os.getenv("URL", "https://api.vsegpt.ru/v1")


config = VseGPTConfig(
    base_url=BASE_URL,
    api_key=API_KEY,
    model=MODEL_ORCHESTOR,
)
llm = VseGPTWrapper(config)

# Список доступных агентов для маршрутизации
AVAILABLE_AGENTS = [agent.strip() for agent in str(os.getenv("AVAILABLE_AGENTS", "")).split(",")]


# --- 1. СТРУКТУРА СОСТОЯНИЯ ---
class TeamState(TypedDict):
    task: str
    target_molecule: str = None  # type: ignore # SMILES-строка, которую ждет агент
    history: List[Any]  # Меняем на Any, так как агент пишет сюда dict, а не str
    properties: dict  # Сюда агент запишет результат
    next_worker: str


# --- 2. ИНИЦИАЛИЗАЦИЯ УЗЛОВ-АГЕНТОВ ---
# Создаем экземпляры агентов
agent_instance = StructurePropertiesAgent(llm=llm, temperature=0.0)

structure_analyzer_node = agent_instance.as_node()

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
2. "FINISH" — если свойства молекулы уже получены или задача полностью выполнена.
"""

    prompt = f"Задача: {state['task']}\nSMILES: {state.get('target_molecule', 'Не указан')}\n\nТекущая история работы:\n{history_text}"

    # Используем json_mode
    result = llm.ask(
        prompt=prompt, system_prompt=system_prompt, json_mode=True, temperature=0.0
    )

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
workflow.add_node("StructureAnalyzer", structure_analyzer_node) # type: ignore

# Точка входа
workflow.add_edge(START, "Supervisor")

# Агенты всегда возвращает результат Супервизору
workflow.add_edge("StructureAnalyzer", "Supervisor")


def route_supervisor(state: TeamState):
    return state["next_worker"]


workflow.add_conditional_edges(
    "Supervisor",
    route_supervisor,
    {"StructureAnalyzer": "StructureAnalyzer", "FINISH": END},
)

app = workflow.compile()
