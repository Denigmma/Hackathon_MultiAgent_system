import json
from typing import Any, Dict

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski


class StructurePropertiesAgent:
    """
    Агент для анализа структуры молекулы и предсказания базовых свойств.

    Интерфейс оставлен простым и стабильным:
    - run(smiles) -> Dict[str, Any]
    - as_node() -> callable для LangGraph
    """

    def __init__(self, model=None, temperature: float = 0.0):
        self.llm = model
        self.temperature = temperature

    @staticmethod
    def compute_rdkit_properties(smiles: str) -> Dict[str, Any]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        return {
            "molecular_weight": round(Descriptors.MolWt(mol), 4),
            "logP": round(Crippen.MolLogP(mol), 4),
            "num_h_donors": Lipinski.NumHDonors(mol),
            "num_h_acceptors": Lipinski.NumHAcceptors(mol),
            "num_rotatable_bonds": Lipinski.NumRotatableBonds(mol),
            "tpsa": round(Descriptors.TPSA(mol), 4),
        }

    def run(self, smiles: str) -> Dict[str, Any]:
        smiles = (smiles or "").strip()
        if not smiles:
            return {"error": "Empty input. Please provide a SMILES string."}

        try:
            props = self.compute_rdkit_properties(smiles)

            if props["logP"] < 1:
                solubility = "high"
            elif props["logP"] < 3:
                solubility = "moderate"
            else:
                solubility = "low"

            toxicity = "unknown"
            if props["logP"] > 5:
                toxicity = "potentially toxic"

            props["solubility_estimate"] = solubility
            props["toxicity_estimate"] = toxicity
            return props

        except Exception as exc:
            return {
                "error": str(exc),
                "hint": "Передайте валидный SMILES, например: CCO",
            }

    def as_node(self):
        def node(state: dict) -> dict:
            smiles = state.get("target_molecule", "")
            result = self.run(smiles)
            state["properties"] = result
            state.setdefault("history", []).append(
                {
                    "agent": "structure_properties",
                    "input": smiles,
                    "output": result,
                }
            )
            return state

        return node


if __name__ == "__main__":
    agent = StructurePropertiesAgent()
    result = agent.run("CCO")
    print(json.dumps(result, indent=2, ensure_ascii=False))
