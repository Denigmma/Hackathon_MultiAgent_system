"""Корневая точка входа приложения.

Файл запускает оркестратор и выводит пользователю только красиво отформатированный 
итоговый ответ модели.
"""

from __future__ import annotations

import os
import json
import warnings

from dotenv import load_dotenv
from loguru import logger

load_dotenv()
warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.",
)

from app.MAS.orchestrator.agent_orchestrator import app

GRAPH_RECURSION_LIMIT = int(os.getenv("GRAPH_RECURSION_LIMIT", "25"))


def _extract_model_answer(result: dict) -> str:
    """Достает и красиво форматирует финальный ответ из состояния графа."""
    
    # 1. Если отработал SeparationMethodsAgent
    sep_res = result.get("separation_result", {})
    if sep_res and not sep_res.get("error"):
        lines = [f"🧪 План разделения: {sep_res.get('target_name', 'Смесь')}"]
        
        for i, rec in enumerate(sep_res.get("suggestions", []), 1):
            lines.append(f"\n{i}. Метод: {rec.get('method', 'Не указан')} (Оценка: {rec.get('score', 0)})")
            lines.append(f"   Обоснование: {rec.get('rationale', '')}")
            
            if rec.get("prerequisites"):
                lines.append("   Требования:")
                for p in rec["prerequisites"]:
                    lines.append(f"    - {p}")
                    
            if rec.get("limitations"):
                lines.append("   Ограничения:")
                for limit in rec["limitations"]:
                    lines.append(f"    - {limit}")
                    
        if sep_res.get("warnings"):
            lines.append("\n⚠️ Предупреждения:")
            for w in sep_res["warnings"]:
                lines.append(f"  - {w}")
                
        return "\n".join(lines)

    # 2. Если отработал StructureAnalyzer (пока выводим базово)
    prop_res = result.get("properties", {})
    if prop_res and not prop_res.get("error"):
        return f"🔬 Свойства молекулы:\n{json.dumps(prop_res, ensure_ascii=False, indent=2)}"

    # 3. Если отработал MixtureReactionAgent (пока выводим базово)
    mix_res = result.get("mixture_reaction", {})
    if mix_res and not mix_res.get("error"):
        return f"⚗️ Продукты реакции:\n{json.dumps(mix_res, ensure_ascii=False, indent=2)}"

    # 4. Fallback: если структурированных ответов нет или была ошибка, 
    # ищем ответ в истории (как было раньше)
    history = result.get("history", [])
    if not isinstance(history, list):
        return "Модель не вернула корректную историю ответов."

    for item in reversed(history):
        if not isinstance(item, dict):
            continue
        
        # Если агент вернул ошибку, выводим её
        output = item.get("output")
        if isinstance(output, dict) and output.get("error"):
            return f"❌ Ошибка от агента {item.get('agent')}: {output.get('error')}"

        if item.get("agent") == "Supervisor":
            if isinstance(output, dict):
                summary = output.get("summary") or output.get("prediction")
                if summary:
                    return str(summary)
            if output is not None:
                return str(output)

    return "Supervisor не вернул финальный ответ или графу не хватило данных."


if __name__ == "__main__":
    try:
        print("--- ЗАПУСК СИСТЕМЫ ---\n")
        user_input = input("Введите запрос: ").strip()

        if not user_input:
            print("Пустой запрос.")
            raise SystemExit(1)

        initial_state = {
            "task": user_input,
            "target_molecule": "",
            "mixture_input": {},
            "separation_task": user_input, # Передаем запрос сюда для SeparationMethodsAgent
            "history": [],
            "properties": {},
            "mixture_reaction": {},
            "separation_result": {},
            "next_worker": "",
        }

        result = app.invoke(  # type: ignore
            initial_state, # type: ignore
            {"recursion_limit": GRAPH_RECURSION_LIMIT},
        )
        
        answer = _extract_model_answer(result)
        print("\n" + "="*50)
        print("ОТВЕТ МОДЕЛИ:")
        print("="*50)
        print(answer)
        print("="*50 + "\n")

    except Exception as err:
        logger.error(f"Ошибка при запуске приложения: {err}")
        raise SystemExit(1)