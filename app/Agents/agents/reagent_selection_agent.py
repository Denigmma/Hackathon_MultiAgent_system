import json
from typing import Dict, Any, List

from langchain.tools import tool
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate


class ReagentSelectionAgent:
    """
    Агент для подбора реагентов и проверки их доступности
    """

    def __init__(self, model=None, temperature: float = 0.0):
        self.llm = model
        self.tools = [self._build_tool()]
        self.agent = self._build_agent()

    # -------------------------
    # Mock / backend data layer
    # -------------------------
    @staticmethod
    def _reagent_catalog() -> List[Dict[str, Any]]:
        """
        Заглушка каталога реагентов.
        В реальной системе здесь может быть:
        - API поставщиков
        - внутренняя БД
        - PubChem / ChEMBL интеграции
        """
        return [
            {"name": "NaBH4", "available": True, "price": 50},
            {"name": "LiAlH4", "available": True, "price": 120},
            {"name": "Pd/C", "available": True, "price": 200},
            {"name": "H2", "available": True, "price": 10},
            {"name": "DCC", "available": False, "price": None},
        ]

    @staticmethod
    def _search_catalog(query: str) -> List[Dict[str, Any]]:
        catalog = ReagentSelectionAgent._reagent_catalog()
        return [r for r in catalog if query.lower() in r["name"].lower()]

    # -------------------------
    # Tool
    # -------------------------
    def _build_tool(self):
        @tool
        def select_and_check_reagents(reaction_description: str) -> str:
            """
            Suggests reagents for a given chemical reaction and checks their availability.
            Input: reaction description (text).
            Output: JSON with proposed reagents and availability info.
            """

            try:
                # простая эвристика подбора (можно заменить LLM-логикой)
                reaction_lower = reaction_description.lower()

                candidates = []

                if "reduction" in reaction_lower or "reduce" in reaction_lower:
                    candidates += ["NaBH4", "LiAlH4", "H2"]

                if "hydrogenation" in reaction_lower:
                    candidates += ["H2", "Pd/C"]

                if "coupling" in reaction_lower:
                    candidates += ["Pd/C", "DCC"]

                # fallback
                if not candidates:
                    candidates = ["NaBH4", "H2"]

                # проверка доступности
                results = []
                for reagent in candidates:
                    matches = self._search_catalog(reagent)

                    if matches:
                        results.extend(matches)
                    else:
                        results.append({
                            "name": reagent,
                            "available": False,
                            "price": None
                        })

                # ранжирование
                ranked = sorted(
                    results,
                    key=lambda r: (not r.get("available", False), r.get("price") or 1e9)
                )

                return json.dumps({
                    "reaction": reaction_description,
                    "proposed_reagents": candidates,
                    "checked_reagents": results,
                    "ranked_reagents": ranked
                }, indent=2)

            except Exception as e:
                return json.dumps({"error": str(e)})

        return select_and_check_reagents

    # -------------------------
    # Agent
    # -------------------------
    def _build_agent(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a chemistry assistant specialized in reagent selection."),
            ("user", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])

        agent = create_openai_functions_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )

        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=False
        )

    # -------------------------
    # Run
    # -------------------------
    def run(self, reaction_description: str) -> Dict[str, Any]:
        response = self.agent.run(
            f"Select suitable reagents and check availability for: {reaction_description}"
        )

        try:
            return json.loads(response)
        except:
            return {"raw_output": response}

    # -------------------------
    # LangGraph node
    # -------------------------
    def as_node(self):
        def node(state: dict) -> dict:
            reaction = state.get("reaction_description")

            result = self.run(reaction)

            state["reagents"] = result

            state.setdefault("history", []).append({
                "agent": "reagent_selection",
                "input": reaction,
                "output": result
            })

            return state

        return node


if __name__ == "__main__":
    agent = ReagentSelectionAgent()

    result = agent.run("reduction of a ketone to alcohol")
    print(json.dumps(result, indent=2))
