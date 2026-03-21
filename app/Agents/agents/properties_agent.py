import json
from typing import Any, Dict

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski


class StructurePropertiesAgent:
    """
    Агент для анализа молекулы по SMILES и прогнозирования свойств.
    Интерфейс сохранён: run(smiles) -> Dict[str, Any]
    """

    def __init__(self, llm=None):
        self.llm = llm

    @staticmethod
    def compute_descriptors(smiles: str) -> Dict[str, Any]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"error": "Invalid SMILES"}

        return {
            "MolWt": round(Descriptors.MolWt(mol), 4),
            "LogP": round(Crippen.MolLogP(mol), 4),
            "NumHDonors": Lipinski.NumHDonors(mol),
            "NumHAcceptors": Lipinski.NumHAcceptors(mol),
            "TPSA": round(Descriptors.TPSA(mol), 4),
            "NumRotatableBonds": Lipinski.NumRotatableBonds(mol),
            "NumHeavyAtoms": Descriptors.HeavyAtomCount(mol),
            "RingCount": Descriptors.RingCount(mol),
        }

    @staticmethod
    def _predict_from_descriptors(desc: Dict[str, Any]) -> Dict[str, Any]:
        if "error" in desc:
            return {"error": desc["error"]}

        logp = float(desc["LogP"])
        tpsa = float(desc["TPSA"])
        molwt = float(desc["MolWt"])
        hbd = int(desc["NumHDonors"])
        hba = int(desc["NumHAcceptors"])

        if logp < 1 and tpsa > 90:
            solubility = "high"
        elif logp < 3:
            solubility = "medium"
        else:
            solubility = "low"

        if logp > 5 or molwt > 550:
            toxicity = "high"
        elif logp > 3.5:
            toxicity = "medium"
        else:
            toxicity = "low"

        lipinski_violations = 0
        lipinski_violations += int(molwt > 500)
        lipinski_violations += int(logp > 5)
        lipinski_violations += int(hbd > 5)
        lipinski_violations += int(hba > 10)

        if lipinski_violations == 0:
            drug_likeness = "high"
        elif lipinski_violations <= 2:
            drug_likeness = "medium"
        else:
            drug_likeness = "low"

        return {
            "solubility": solubility,
            "toxicity": toxicity,
            "drug_likeness": drug_likeness,
            "comments": (
                "Heuristic estimate based on RDKit descriptors and Lipinski-style rules."
            ),
            "lipinski_violations": lipinski_violations,
        }

    def run(self, smiles: str) -> Dict[str, Any]:
        smiles = (smiles or "").strip()
        if not smiles:
            return {"error": "Empty input. Please provide a SMILES string."}

        descriptors = self.compute_descriptors(smiles)
        prediction = self._predict_from_descriptors(descriptors)

        return {
            "input": smiles,
            "descriptors": descriptors,
            "prediction": prediction,
        }

    def as_node(self):
        def node(state: dict) -> dict:
            smiles = state.get("target_molecule", "")
            result = self.run(smiles)
            state["properties"] = result
            state.setdefault("history", []).append(
                {"agent": "properties_agent", "input": smiles, "output": result}
            )
            return state

        return node


if __name__ == "__main__":
    agent = StructurePropertiesAgent()
    result = agent.run("CCO")
    print(json.dumps(result, indent=2, ensure_ascii=False))
