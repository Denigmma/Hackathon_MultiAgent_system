# import json
# from typing import Any, Dict, List
#
#
# class ReagentSelectionAgent:
#     """
#     Агент для подбора реагентов и проверки доступности.
#     """
#
#     def __init__(self, model=None, temperature: float = 0.0):
#         self.llm = model
#         self.temperature = temperature
#
#     @staticmethod
#     def _reagent_catalog() -> List[Dict[str, Any]]:
#         return [
#             {"name": "NaBH4", "available": True, "price": 50},
#             {"name": "LiAlH4", "available": True, "price": 120},
#             {"name": "Pd/C", "available": True, "price": 200},
#             {"name": "H2", "available": True, "price": 10},
#             {"name": "DCC", "available": False, "price": None},
#         ]
#
#     @staticmethod
#     def _search_catalog(query: str) -> List[Dict[str, Any]]:
#         catalog = ReagentSelectionAgent._reagent_catalog()
#         return [r for r in catalog if query.lower() in r["name"].lower()]
#
#     def run(self, reaction_description: str) -> Dict[str, Any]:
#         try:
#             reaction_description = (reaction_description or "").strip()
#             if not reaction_description:
#                 return {
#                     "error": "Empty input. Provide reaction description.",
#                 }
#
#             reaction_lower = reaction_description.lower()
#             candidates: List[str] = []
#
#             if "reduction" in reaction_lower or "reduce" in reaction_lower:
#                 candidates += ["NaBH4", "LiAlH4", "H2"]
#
#             if "hydrogenation" in reaction_lower:
#                 candidates += ["H2", "Pd/C"]
#
#             if "coupling" in reaction_lower:
#                 candidates += ["Pd/C", "DCC"]
#
#             if not candidates:
#                 candidates = ["NaBH4", "H2"]
#
#             results: List[Dict[str, Any]] = []
#             for reagent in candidates:
#                 matches = self._search_catalog(reagent)
#                 if matches:
#                     results.extend(matches)
#                 else:
#                     results.append(
#                         {
#                             "name": reagent,
#                             "available": False,
#                             "price": None,
#                         }
#                     )
#
#             ranked = sorted(
#                 results,
#                 key=lambda r: (not r.get("available", False), r.get("price") or 10**9),
#             )
#
#             return {
#                 "reaction": reaction_description,
#                 "proposed_reagents": candidates,
#                 "checked_reagents": results,
#                 "ranked_reagents": ranked,
#             }
#         except Exception as exc:
#             return {"error": str(exc)}
#
#     def as_node(self):
#         def node(state: dict) -> dict:
#             reaction = state.get("reaction_description", "")
#             result = self.run(reaction)
#             state["reagents"] = result
#             state.setdefault("history", []).append(
#                 {
#                     "agent": "reagent_selection",
#                     "input": reaction,
#                     "output": result,
#                 }
#             )
#             return state
#
#         return node
#
