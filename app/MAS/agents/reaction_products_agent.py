import json
from typing import Any, Dict, List, Optional, Union

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool


class MixtureReactionAgent:
    """
    Агент для анализа смеси веществ на базе LangChain v1.

    Идея:
    - LLM сама решает, какие инструменты вызвать;
    - инструменты дают ей химические эвристики и массовые доли;
    - итог возвращается как dict;
    - есть as_tool() и as_node() для интеграции в LangChain / графы.
    """

    def __init__(
            self,
            model: str = "openai:gpt-5.2",
            temperature: float = 0.0,
            max_substances: int = 20,
    ):
        self.max_substances = max_substances
        self.model = init_chat_model(model, temperature=temperature)

        self.agent = create_agent(
            model=self.model,
            tools=self._build_tools(),
            system_prompt=(
                "Ты химический ассистент. Анализируй смесь веществ и условия смешения. "
                "При необходимости вызывай инструменты. "
                "Верни финальный ответ строго как JSON с полями: "
                "reaction_likely, summary, products, mixture_composition, warnings. "
                "Будь консервативен и не выдумывай реакцию без оснований."
            ),
            name="mixture_reaction_agent",
        )

    @staticmethod
    def _norm(x: Any) -> str:
        return str(x or "").strip().lower().replace(" ", "").replace("-", "").replace("·", ".")

    @staticmethod
    def _to_dict(obj: Any) -> Dict[str, Any]:
        if isinstance(obj, dict):
            return obj
        if hasattr(obj, "__dict__"):
            return dict(obj.__dict__)
        return dict(obj)

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        text = (text or "").strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) >= 3 and lines[-1].strip().startswith("```"):
                return "\n".join(lines[1:-1]).strip()
        return text

    @classmethod
    def _loads_json(cls, text: str) -> Dict[str, Any]:
        text = cls._strip_code_fences(text)
        try:
            return json.loads(text)
        except Exception:
            return {"error": "Invalid JSON", "raw": text}

    def _build_tools(self):
        agent_self = self

        @tool("mass_fractions")
        def mass_fractions(payload_json: str) -> str:
            """
            Возвращает массовые доли веществ.
            Ожидает JSON вида: {"substances": [...]}
            """
            try:
                payload = json.loads(payload_json)
            except Exception:
                return json.dumps({"error": "Invalid JSON payload."}, ensure_ascii=False)

            substances = payload.get("substances", [])
            items = []
            total_mass = 0.0

            for s in substances:
                s = agent_self._to_dict(s)
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
                return json.dumps({"total_mass_g": 0.0, "items": []}, ensure_ascii=False)

            for item in items:
                item["mass_fraction_percent"] = round(item["mass_g"] / total_mass * 100, 3)

            return json.dumps(
                {"total_mass_g": round(total_mass, 6), "items": items},
                ensure_ascii=False,
            )

        @tool("chemistry_rules")
        def chemistry_rules(payload_json: str) -> str:
            """
            Возвращает химические эвристики:
            - нейтрализация
            - кислота + карбонат
            - аммонийная соль + щёлочь
            - типовые осадки

            Ожидает JSON вида: {"substances": [...], "conditions": {...}}
            """
            try:
                payload = json.loads(payload_json)
            except Exception:
                return json.dumps({"error": "Invalid JSON payload."}, ensure_ascii=False)

            substances = [agent_self._to_dict(s) for s in payload.get("substances", [])]
            names = {agent_self._norm(s.get("name")) for s in substances}

            acids = {"hcl", "h2so4", "hno3", "h3po4", "ch3cooh"}
            bases = {"naoh", "koh", "nh3", "ca(oh)2", "ba(oh)2"}
            carbonates = {"na2co3", "k2co3", "caco3", "mgco3", "nahco3", "khco3"}
            ammoniums = {"nh4cl", "nh4no3", "(nh4)2so4", "nh4br", "nh4i", "nh4oh"}

            precipitation_pairs = {
                frozenset({"agno3", "nacl"}): "agcl",
                frozenset({"agno3", "nabr"}): "agbr",
                frozenset({"agno3", "nai"}): "agi",
                frozenset({"bacl2", "na2so4"}): "baso4",
                frozenset({"cacl2", "na2co3"}): "caco3",
                frozenset({"cacl2", "nahco3"}): "caco3",
                frozenset({"pb(no3)2", "ki"}): "pbi2",
                frozenset({"feso4", "naoh"}): "fe(oh)2",
                frozenset({"fecl3", "naoh"}): "fe(oh)3",
                frozenset({"cuso4", "naoh"}): "cu(oh)2",
                frozenset({"alcl3", "naoh"}): "al(oh)3",
            }

            matches: List[Dict[str, Any]] = []

            if (names & acids) and (names & bases):
                acid = next((s["name"] for s in substances if agent_self._norm(s.get("name")) in acids), "acid")
                base = next((s["name"] for s in substances if agent_self._norm(s.get("name")) in bases), "base")
                matches.append(
                    {
                        "type": "acid_base",
                        "reaction_likely": True,
                        "confidence": 0.95,
                        "summary": "Кислота и основание: вероятна нейтрализация.",
                        "equation": f"{acid} + {base} -> salt + H2O",
                        "products": [{"name": "salt + water", "confidence": 0.95}],
                        "warnings": [],
                    }
                )

            if (names & acids) and (names & carbonates):
                acid = next((s["name"] for s in substances if agent_self._norm(s.get("name")) in acids), "acid")
                carbonate = next((s["name"] for s in substances if agent_self._norm(s.get("name")) in carbonates),
                                 "carbonate")
                matches.append(
                    {
                        "type": "acid_carbonate",
                        "reaction_likely": True,
                        "confidence": 0.92,
                        "summary": "Кислота и карбонат/гидрокарбонат: вероятно выделение CO2.",
                        "equation": f"{carbonate} + {acid} -> salt + H2O + CO2",
                        "products": [{"name": "salt + water + carbon dioxide", "confidence": 0.92}],
                        "warnings": ["Возможное газовыделение."],
                    }
                )

            if (names & bases) and (names & ammoniums):
                base = next((s["name"] for s in substances if agent_self._norm(s.get("name")) in bases), "base")
                ammonium = next((s["name"] for s in substances if agent_self._norm(s.get("name")) in ammoniums),
                                "ammonium salt")
                matches.append(
                    {
                        "type": "ammonium_base",
                        "reaction_likely": True,
                        "confidence": 0.9,
                        "summary": "Аммонийная соль и щёлочь: вероятно выделение аммиака.",
                        "equation": f"{ammonium} + {base} -> NH3 + H2O + salt",
                        "products": [{"name": "ammonia + water + salt", "confidence": 0.9}],
                        "warnings": ["Возможное выделение аммиака."],
                    }
                )

            for pair, precipitate in precipitation_pairs.items():
                if pair.issubset(names):
                    a, b = sorted(pair)
                    matches.append(
                        {
                            "type": "precipitation",
                            "reaction_likely": True,
                            "confidence": 0.88,
                            "summary": "Вероятно образование осадка.",
                            "equation": f"{a} + {b} -> {precipitate}↓ + byproducts",
                            "products": [{"name": f"precipitate: {precipitate}", "confidence": 0.88}],
                            "warnings": ["Возможное образование осадка."],
                        }
                    )
                    break

            return json.dumps(
                {
                    "matches": matches,
                    "best_match": max(matches, key=lambda x: x.get("confidence", 0.0), default=None),
                },
                ensure_ascii=False,
            )

        return [mass_fractions, chemistry_rules]

    def _validate_inputs(self, substances: List[Dict[str, Any]], conditions: Optional[Dict[str, Any]]) -> Optional[
        Dict[str, Any]]:
        if not substances:
            return {"error": "Список веществ не должен быть пустым."}

        if len(substances) > self.max_substances:
            return {"error": f"Слишком много веществ (max {self.max_substances})."}

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

    def _extract_final_json(self, agent_state: Dict[str, Any]) -> Dict[str, Any]:
        messages = agent_state.get("messages", [])
        if not messages:
            return {"error": "Agent returned no messages.", "raw_state": agent_state}

        last = messages[-1]
        content = getattr(last, "content", last if isinstance(last, str) else None)

        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    parts.append(str(block["text"]))
                else:
                    parts.append(str(block))
            content = "".join(parts)
        elif not isinstance(content, str):
            content = str(content)

        parsed = self._loads_json(content)
        if "error" in parsed:
            parsed["raw_content"] = content
        return parsed

    def _mass_fractions_local(self, substances: List[Dict[str, Any]]) -> Dict[str, Any]:
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

    def run(
            self,
            substances: List[Union[Dict[str, Any], Any]],
            conditions: Optional[Union[Dict[str, Any], Any]] = None,
            context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Основной метод анализа смеси.
        Возвращает dict.
        """
        substances_dict = [self._to_dict(s) for s in substances]
        conditions_dict = self._to_dict(conditions) if conditions is not None else None

        validation_error = self._validate_inputs(substances_dict, conditions_dict)
        if validation_error:
            return validation_error

        payload = {
            "substances": substances_dict,
            "conditions": conditions_dict,
            "context": context,
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

        final_json = self._extract_final_json(agent_state)

        if not isinstance(final_json, dict):
            final_json = {"error": "Unexpected agent output.", "raw": final_json}

        if "mixture_composition" not in final_json or not final_json.get("mixture_composition"):
            final_json["mixture_composition"] = self._mass_fractions_local(substances_dict)

        return {
            "input": payload,
            "analysis": final_json,
        }

    def as_tool(self):
        """
        Преобразует агента в LangChain tool.
        Tool принимает JSON-строку с ключами substances, conditions, context.
        """

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
        """
        Преобразует агента в callable-узел для графовых пайплайнов.
        """

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
                    "agent": "mixture_reaction",
                    "input": payload,
                    "output": result,
                }
            )
            return state

        return node
