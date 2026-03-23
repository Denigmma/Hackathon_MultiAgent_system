from __future__ import annotations

import json
import re
from typing import Any, Dict

from langchain.tools import tool
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

SMILES_CANDIDATE_RE = re.compile(r"[A-Za-z0-9@+\-\[\]\(\)=#$\\/%.]+")


class StructurePropertiesAgent:
    """Детерминированный агент анализа молекулы по SMILES.

    В отличие от предыдущей версии, этот агент не зависит от LLM для базового
    расчёта дескрипторов и предсказания простых эвристических свойств.
    Это делает его стабильным и воспроизводимым.
    """

    def __init__(self, model: str | None = None, temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature

    @staticmethod
    def _extract_smiles_from_text(text: str) -> str:
        text = (text or "").strip()
        if not text:
            return ""

        for token in SMILES_CANDIDATE_RE.findall(text):
            candidate = token.strip(".,;:!?\"'")
            if not candidate:
                continue

            mol = Chem.MolFromSmiles(candidate)
            if mol is None:
                continue

            return Chem.MolToSmiles(mol, canonical=True)

        return ""

    @staticmethod
    def compute_descriptors(smiles: str) -> Dict[str, Any]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"error": "Invalid SMILES"}

        return {
            "Formula": rdMolDescriptors.CalcMolFormula(mol),
            "MolWt": round(Descriptors.MolWt(mol), 4),
            "LogP": round(Crippen.MolLogP(mol), 4),
            "TPSA": round(rdMolDescriptors.CalcTPSA(mol), 4),
            "NumHeavyAtoms": int(mol.GetNumHeavyAtoms()),
            "NumAromaticRings": int(rdMolDescriptors.CalcNumAromaticRings(mol)),
            "NumHDonors": int(Lipinski.NumHDonors(mol)),
            "NumHAcceptors": int(Lipinski.NumHAcceptors(mol)),
            "NumRotatableBonds": int(Lipinski.NumRotatableBonds(mol)),
            "FractionCSP3": round(rdMolDescriptors.CalcFractionCSP3(mol), 4),
        }

    @staticmethod
    def _predict_properties(desc: Dict[str, Any]) -> Dict[str, Any]:
        if "error" in desc:
            return {"error": desc["error"]}

        logp = float(desc.get("LogP", 0))
        tpsa = float(desc.get("TPSA", 0))
        molwt = float(desc.get("MolWt", 0))
        hbd = int(desc.get("NumHDonors", 0))
        hba = int(desc.get("NumHAcceptors", 0))
        rot_bonds = int(desc.get("NumRotatableBonds", 0))
        frac_csp3 = float(desc.get("FractionCSP3", 0))

        if logp < 1 and tpsa > 90:
            solubility = "high"
        elif logp < 3:
            solubility = "medium"
        else:
            solubility = "low"

        if logp > 5 or molwt > 550 or frac_csp3 < 0.2:
            toxicity = "high"
        elif logp > 3.5:
            toxicity = "medium"
        else:
            toxicity = "low"

        violations = 0
        violations += int(molwt > 500)
        violations += int(logp > 5)
        violations += int(hbd > 5)
        violations += int(hba > 10)
        violations += int(rot_bonds > 10)

        if violations == 0:
            drug_likeness = "high"
        elif violations <= 2:
            drug_likeness = "medium"
        else:
            drug_likeness = "low"

        if rot_bonds < 3:
            rigidity = "rigid"
        elif rot_bonds < 8:
            rigidity = "moderate"
        else:
            rigidity = "flexible"

        if tpsa <= 90 and logp <= 5:
            permeability = "likely_good"
        elif tpsa <= 120:
            permeability = "borderline"
        else:
            permeability = "likely_low"

        return {
            "solubility": solubility,
            "toxicity": toxicity,
            "drug_likeness": drug_likeness,
            "rigidity": rigidity,
            "permeability": permeability,
            "lipinski_violations": violations,
        }

    @staticmethod
    def _build_summary(canonical_smiles: str, desc: Dict[str, Any], prediction: Dict[str, Any]) -> str:
        if "error" in desc:
            return "Не удалось вычислить дескрипторы: SMILES некорректен."

        return (
            f"Молекула {canonical_smiles} валидна; формула {desc['Formula']}, "
            f"MolWt={desc['MolWt']}, LogP={desc['LogP']}, TPSA={desc['TPSA']}. "
            f"По эвристической оценке: растворимость {prediction['solubility']}, "
            f"drug-likeness {prediction['drug_likeness']}, токсикологический риск {prediction['toxicity']}."
        )

    def run(self, smiles_or_text: str) -> Dict[str, Any]:
        raw_input = (smiles_or_text or "").strip()
        if not raw_input:
            return {
                "input": "",
                "valid": False,
                "descriptors": {},
                "prediction": {"error": "Empty SMILES"},
                "summary": "SMILES не передан.",
            }

        smiles = raw_input
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            extracted = self._extract_smiles_from_text(raw_input)
            if extracted:
                smiles = extracted
                mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            return {
                "input": raw_input,
                "valid": False,
                "descriptors": {},
                "prediction": {"error": "Invalid SMILES"},
                "summary": "Из запроса не удалось извлечь корректный SMILES.",
            }

        canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
        descriptors = self.compute_descriptors(canonical_smiles)
        prediction = self._predict_properties(descriptors)
        summary = self._build_summary(canonical_smiles, descriptors, prediction)

        return {
            "input": canonical_smiles,
            "valid": True,
            "descriptors": descriptors,
            "prediction": prediction,
            "summary": summary,
        }

    def as_tool(self):
        @tool("analyze_structure")
        def analyze_structure(smiles: str) -> dict:
            """Анализ молекулы по SMILES."""
            return self.run(smiles)

        return analyze_structure

    def as_node(self):
        def node(state: Dict[str, Any]) -> Dict[str, Any]:
            smiles_or_text = (
                state.get("target_molecule")
                or state.get("smiles")
                or state.get("task")
                or ""
            )
            result = self.run(str(smiles_or_text))

            if result.get("valid"):
                state["target_molecule"] = str(result.get("input", ""))

            state["properties"] = result
            state.setdefault("history", []).append(
                {
                    "agent": "StructureAnalyzer",
                    "input": smiles_or_text,
                    "output": result,
                }
            )
            return state

        return node
