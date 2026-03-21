from typing import Annotated, Sequence, TypedDict
import operator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from loguru import logger
import os


# Подключаем VseGPT через стандартный клиент LangChain
llm = ChatOpenAI(
    api_key=os.getenv("VSEGPT_API_KEY"), # type: ignore
    base_url="https://api.vsegpt.ru/v1",
    model="openai/gpt-4o-mini",
    temperature=0.01
)

# --- 2. ОПРЕДЕЛЕНИЕ СОСТОЯНИЯ ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str

# --- 3. СПИСОК АГЕНТОВ ---
agents = [
    "Structure_Property_Agent", 
    "Retrosynthesis_Agent", 
    "Protocol_Search_Agent", 
    "Reagents_Agent", 
    "Literature_RAG_Agent", 
    "Validator_Agent"
]

# Pydantic-модель для структурированного ответа Оркестратора
class Route(BaseModel):
    next: str = Field(
        description="Выбери следующего агента или 'FINISH', если итоговый маршрут синтеза готов и проверен."
    )

# --- 4. УЗЕЛ ОРКЕСТРАТОРА ---
def supervisor_node(state: AgentState) -> dict:
    logger.info("Оркестратор: Анализирую текущее состояние...")
    
    system_prompt = (
        "Ты — Главный химик-супервайзер. Твоя задача управлять планированием синтеза органических молекул.\n"
        "В твоем распоряжении команда специализированных агентов: {agents}.\n\n"
        "ПРАВИЛА МАРШРУТИЗАЦИИ:\n"
        "1. Анализируй историю сообщений. Определи, на каком этапе находится процесс.\n"
        "2. Вызови нужного агента для следующего шага.\n"
        "3. ВАЖНО: Соблюдай итеративную логику! Если Protocol_Search_Agent или Reagents_Agent сообщают о тупике "
        "(нет методик или недоступны реагенты), ты ДОЛЖЕН вернуть задачу обратно к Retrosynthesis_Agent для поиска альтернативного маршрута.\n"
        "4. В конце всегда вызывай Validator_Agent для проверки консистентности.\n"
        "5. Если Validator_Agent подтвердил маршрут, и всё готово, верни 'FINISH'."
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ]).partial(agents=", ".join(agents))
    
    # Используем with_structured_output. Большинство умных моделей во VseGPT (особенно с префиксом openai/) 
    # отлично поддерживают этот метод для возврата строгого JSON.
    supervisor_chain = prompt | llm.with_structured_output(Route)
    
    result = supervisor_chain.invoke(state) # type: ignore
    logger.warning(f"Оркестратор принял решение. Следующий узел: {result.next}") # type: ignore
    
    return {"next": result.next} # type: ignore

# --- 5. ЗАГЛУШКИ ДЛЯ АГЕНТОВ КОЛЛЕГ ---
def mock_agent(state: AgentState, name: str, response_mock: str) -> dict:
    logger.debug(f"{name} начал работу.")
    result_message = AIMessage(content=response_mock, name=name)
    logger.success(f"{name} завершил работу.")
    return {"messages": [result_message]}

def structure_node(state: AgentState): return mock_agent(state, "Structure_Property_Agent", "Свойства молекулы посчитаны. logP=2.5")
def retro_node(state: AgentState): return mock_agent(state, "Retrosynthesis_Agent", "Предлагаю маршрут: Прекурсор А + Прекурсор Б -> Целевая молекула")
def protocol_node(state: AgentState): return mock_agent(state, "Protocol_Search_Agent", "Методика найдена: реакция Сузуки.")
def reagents_node(state: AgentState): return mock_agent(state, "Reagents_Agent", "Все реагенты доступны на складе.")
def literature_node(state: AgentState): return mock_agent(state, "Literature_RAG_Agent", "Выход реакции составит около 85%.")
def validator_node(state: AgentState): return mock_agent(state, "Validator_Agent", "Проверка пройдена: маршрут консистентен.")

# --- 6. СБОРКА ГРАФА ---
workflow = StateGraph(AgentState)

workflow.add_node("Supervisor", supervisor_node)
workflow.add_node("Structure_Property_Agent", structure_node)
workflow.add_node("Retrosynthesis_Agent", retro_node)
workflow.add_node("Protocol_Search_Agent", protocol_node)
workflow.add_node("Reagents_Agent", reagents_node)
workflow.add_node("Literature_RAG_Agent", literature_node)
workflow.add_node("Validator_Agent", validator_node)

for agent in agents:
    workflow.add_edge(agent, "Supervisor")

conditional_map = {k: k for k in agents}
conditional_map["FINISH"] = END

workflow.add_conditional_edges("Supervisor", lambda x: x["next"], conditional_map) # type: ignore
workflow.set_entry_point("Supervisor")
app = workflow.compile()

# --- 7. ТЕСТОВЫЙ ЗАПУСК ---
if __name__ == "__main__":
    logger.info("Запуск системы планирования синтеза...")
    
    # Тестовый запрос
    initial_input = {"messages": [HumanMessage(content="Спланируй синтез ибупрофена.")]}
    
    for event in app.stream(initial_input, {"recursion_limit": 15}): # type: ignore
        if "__end__" not in event:
            print("---")
