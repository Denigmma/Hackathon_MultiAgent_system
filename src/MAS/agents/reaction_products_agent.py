from __future__ import annotations

import ast
import json
import operator as op
import os
from typing import Any, Dict, List, Optional, Union

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool

from src.NeuralSearch.main import main as neural_search_main
# from src.RAG.main import main as rag_main

MODEL_AGENT = os.getenv("MODEL_AGENT", "openai/gpt-5.4-nano-thinking-xhigh")
MODEL_PROVIDER_AGENT = os.getenv("MODEL_PROVIDER_AGENT", "openai")
VSEGPT_API_KEY = os.getenv("VSEGPT_API_KEY", "")
BASE_URL = os.getenv("URL", "https://api.vsegpt.ru/v1")
AGENT_TIMEOUT_SECONDS = float(os.getenv("AGENT_TIMEOUT_SECONDS", "60"))


class MixtureReactionAgent:
    """
    Агент для анализа смеси веществ.

    Tools:
    - rag_search
    - neural_search
    - evaluate_expression
    """

    def __init__(
            self,
            model: str = MODEL_AGENT,
            temperature: float = 0.0,
            max_substances: int = 20,
    ):
        if not VSEGPT_API_KEY:
            raise ValueError("VSEGPT_API_KEY не задан в окружении.")

        self.max_substances = max_substances

        self.model = init_chat_model(
            model,
            model_provider=MODEL_PROVIDER_AGENT,
            temperature=temperature,
            api_key=VSEGPT_API_KEY,
            base_url=BASE_URL,
            timeout=AGENT_TIMEOUT_SECONDS,
        )

        self.agent = create_agent(
            model=self.model,
            tools=self._build_tools(),
            system_prompt=(
                "Ты химический ассистент. Анализируй смесь веществ и условия смешения. "
                "При необходимости используй инструменты: "
                "rag_search, neural_search, evaluate_expression. "
                "Верни финальный ответ строго как JSON с полями: "
                "reaction_likely, summary, products, mixture_composition, warnings."
            ),
            name="mixture_reaction_agent",
        )

    @staticmethod
    def _to_dict(obj: Any) -> Dict[str, Any]:
        if isinstance(obj, dict):
            return obj
        if hasattr(obj, "__dict__"):
            return dict(obj.__dict__)
        return dict(obj)

    @staticmethod
    def _loads_json(text: str) -> Dict[str, Any]:
        text = (text or "").strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) >= 3 and lines[-1].strip().startswith("```"):
                text = "\n".join(lines[1:-1]).strip()

        try:
            data = json.loads(text)
            return data if isinstance(data, dict) else {"error": "Non-dict JSON", "raw": data}
        except Exception:
            return {"error": "Invalid JSON", "raw": text}

    @staticmethod
    def _extract_final_text(agent_state: Any) -> str:
        if isinstance(agent_state, str):
            return agent_state

        if isinstance(agent_state, dict):
            if "output" in agent_state and isinstance(agent_state["output"], str):
                return agent_state["output"]

            messages = agent_state.get("messages") or []
            if messages:
                last = messages[-1]
                content = getattr(last, "content", last if isinstance(last, str) else "")
                if isinstance(content, list):
                    parts = []
                    for block in content:
                        if isinstance(block, dict) and "text" in block:
                            parts.append(str(block["text"]))
                        else:
                            parts.append(str(block))
                    return "".join(parts)
                return str(content)

        if hasattr(agent_state, "content"):
            return str(getattr(agent_state, "content", ""))

        return str(agent_state)

    @staticmethod
    def _mass_fractions_local(substances: List[Dict[str, Any]]) -> Dict[str, Any]:
        items = []
        total_mass = 0.0

        for s in substances:
            if s.get("amount_unit") != "g" or s.get("amount") is None:
                continue

            mass = float(s["amount"])
            if s.get("purity") is not None:
                mass *= float(s["purity"])

            if mass < 0:
                continue

            items.append({"name": s.get("name", ""), "mass_g": round(mass, 6)})
            total_mass += mass

        if total_mass <= 0:
            return {"total_mass_g": 0.0, "items": []}

        for item in items:
            item["mass_fraction_percent"] = round(item["mass_g"] / total_mass * 100, 3)

        return {"total_mass_g": round(total_mass, 6), "items": items}

    @staticmethod
    def _validate_inputs(
            substances: List[Dict[str, Any]],
            conditions: Optional[Dict[str, Any]],
            max_substances: int,
    ) -> Optional[Dict[str, Any]]:
        if not substances:
            return {"error": "Список веществ не должен быть пустым."}

        if len(substances) > max_substances:
            return {"error": f"Слишком много веществ (max {max_substances})."}

        for s in substances:
            name = str(s.get("name", "")).strip()
            if not name:
                return {"error": "У каждого вещества должно быть имя."}

            if s.get("amount") is not None and float(s["amount"]) < 0:
                return {"error": "Количество не может быть отрицательным."}

            if s.get("purity") is not None:
                purity = float(s["purity"])
                if not (0 <= purity <= 1):
                    return {"error": "Чистота должна быть в диапазоне 0..1."}

        if conditions and conditions.get("pH") is not None:
            pH = float(conditions["pH"])
            if not (0 <= pH <= 14):
                return {"error": "pH должен быть в диапазоне 0..14."}

        return None

    _OPS = {
        ast.Add: op.add,
        ast.Sub: op.sub,
        ast.Mult: op.mul,
        ast.Div: op.truediv,
        ast.FloorDiv: op.floordiv,
        ast.Mod: op.mod,
        ast.Pow: op.pow,
        ast.USub: op.neg,
        ast.UAdd: op.pos,
    }

    @classmethod
    def _safe_eval(cls, expr: str) -> float:
        def _eval(node: ast.AST) -> float:
            if isinstance(node, ast.Expression):
                return _eval(node.body)

            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return float(node.value)

            if isinstance(node, ast.BinOp) and type(node.op) in cls._OPS:
                return cls._OPS[type(node.op)](_eval(node.left), _eval(node.right))

            if isinstance(node, ast.UnaryOp) and type(node.op) in cls._OPS:
                return cls._OPS[type(node.op)](_eval(node.operand))

            raise ValueError("Unsupported expression")

        tree = ast.parse(expr, mode="eval")
        return _eval(tree)

    def _build_tools(self):
        agent_self = self

        # @tool("rag_search")
        # def rag_search(query: str) -> str:
        #     """Поиск по локальной RAG системе."""
        #     try:
        #         return str(rag_main(query))
        #     except Exception as e:
        #         return f"RAG_ERROR: {e}"

        @tool("neural_search")
        def neural_search(query: str) -> str:
            """Поиск через NeuralSearch."""
            try:
                return str(neural_search_main(query))
            except Exception as e:
                return f"NEURAL_ERROR: {e}"

        @tool("evaluate_expression")
        def evaluate_expression(expr: str) -> str:
            """Вычисляет математическое выражение безопасным способом."""
            try:
                return str(agent_self._safe_eval(expr))
            except Exception as e:
                return f"EVAL_ERROR: {e}"

        # return [rag_search, neural_search, evaluate_expression]
        return [neural_search, evaluate_expression]

    def run(
            self,
            substances: List[Union[Dict[str, Any], Any]],
            conditions: Optional[Union[Dict[str, Any], Any]] = None,
            context: Optional[str] = None,
    ) -> Dict[str, Any]:
        substances_dict = [self._to_dict(s) for s in substances]
        conditions_dict = self._to_dict(conditions) if conditions is not None else None

        validation_error = self._validate_inputs(
            substances_dict,
            conditions_dict,
            self.max_substances,
        )
        if validation_error:
            return validation_error

        payload = {
            "substances": substances_dict,
            "conditions": conditions_dict,
            "context": context,
            "mixture_composition": self._mass_fractions_local(substances_dict),
        }

        agent_state = self.agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": json.dumps(payload, ensure_ascii=False, indent=2),
                    }
                ]
            }
        )

        final_text = self._extract_final_text(agent_state)
        final_json = self._loads_json(final_text)

        if "mixture_composition" not in final_json or not final_json.get("mixture_composition"):
            final_json["mixture_composition"] = payload["mixture_composition"]

        return {
            "input": payload,
            "analysis": final_json,
        }

    def as_tool(self):
        @tool("analyze_mixture")
        def analyze_mixture(payload_json: str) -> dict:
            try:
                payload = json.loads(payload_json)
            except Exception:
                return {"error": "Invalid JSON payload."}

            return self.run(
                payload.get("substances", []),
                conditions=payload.get("conditions"),
                context=payload.get("context"),
            )

        return analyze_mixture

    def as_node(self):
        def node(state: dict) -> dict:
            payload = state.get("mixture_input", state)
            result = self.run(
                payload.get("substances", []),
                conditions=payload.get("conditions"),
                context=payload.get("context"),
            )
            state["mixture_reaction"] = result
            state.setdefault("history", []).append(
                {
                    "agent": "MixtureReactionAgent",
                    "input": payload,
                    "output": result,
                }
            )
            return state

        return node
