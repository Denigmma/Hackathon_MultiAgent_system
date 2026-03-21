"""Корневая точка входа приложения."""

...
from loguru import logger
from dotenv import load_dotenv
load_dotenv()
from app.Agents.Orchestrator.agent_orchestrator import app


if __name__ == "__main__":
    try:
        print("--- ЗАПУСК СИСТЕМЫ ---\n")

        user_input = input("Введите запрос: ")
        # Исходное состояние: задаем задачу и сразу передаем SMILES в target_molecule
        initial_state = {
            "task": "Проанализируй молекулу этанола",
            "target_molecule": "CCO",  # Этанол
            "history": [],
            "properties": {},
        }

        result = app.invoke(initial_state) # type: ignore

        print("\n--- ИТОГОВЫЙ ОТЧЕТ ИСТОРИИ ---")
        for event in result["history"]:
            if isinstance(event, dict):
                # Красиво выводим словари от агента
                print(f"[{event.get('agent')}] Input: {event.get('input')}")
                print(f"Prediction: {event.get('output', {}).get('prediction')}")
            else:
                print(event)
            print("-" * 30)
    except Exception as err:
        logger.error(f"Ошибка при запуске приложения: {err}")
        raise SystemExit()
