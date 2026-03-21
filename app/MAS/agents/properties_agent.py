from __future__ import annotations

from typing import Any, Dict, Optional

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
from langchain.tools import tool


class StructurePropertiesAgent:
    """
    Агент для анализа молекул на основе SMILES.

    Класс вычисляет молекулярные дескрипторы с помощью RDKit
    и на их основе формирует эвристические оценки свойств молекулы:
    растворимость, токсичность, drug-likeness и гибкость структуры.

    Основной интерфейс:
    - run(smiles) -> Dict[str, Any]
    - as_tool() -> LangChain tool
    - as_node() -> callable для графовых пайплайнов
    """

    def __init__(self, llm: Optional[Any] = None, temperature: float = 0.0):
        """
        Инициализация агента.

        Args:
            llm: Опциональная LLM-модель. В данной реализации не используется,
                но оставлена для совместимости с LangChain-пайплайнами.
            temperature: Параметр температуры для возможного дальнейшего использования LLM.
        """
        self.llm = llm
        self.temperature = temperature

    @staticmethod
    def compute_descriptors(smiles: str) -> Dict[str, Any]:
        """
        Вычисляет набор молекулярных дескрипторов для входной SMILES-строки.

        Args:
            smiles: Строка SMILES, описывающая молекулу.

        Returns:
            Словарь дескрипторов молекулы или словарь с ключом "error",
            если SMILES некорректен.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"error": "Invalid SMILES"}

        return {
            "MolWt": round(Descriptors.MolWt(mol), 4),
            "ExactMolWt": round(Descriptors.ExactMolWt(mol), 4),
            "LogP": round(Crippen.MolLogP(mol), 4),
            "TPSA": round(rdMolDescriptors.CalcTPSA(mol), 4),
            "NumHDonors": Lipinski.NumHDonors(mol),
            "NumHAcceptors": Lipinski.NumHAcceptors(mol),
            "NumRotatableBonds": Lipinski.NumRotatableBonds(mol),
            "NumHeavyAtoms": Lipinski.HeavyAtomCount(mol),
            "RingCount": Lipinski.RingCount(mol),
            "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),
            "NumAliphaticRings": rdMolDescriptors.CalcNumAliphaticRings(mol),
            "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
            "LabuteASA": rdMolDescriptors.CalcLabuteASA(mol),
            "Chi0": rdMolDescriptors.CalcChi0(mol),
            "Chi1": rdMolDescriptors.CalcChi1(mol),
            "Kappa1": rdMolDescriptors.CalcKappa1(mol),
            "Kappa2": rdMolDescriptors.CalcKappa2(mol),
            "Kappa3": rdMolDescriptors.CalcKappa3(mol),
        }

    @staticmethod
    def _predict_properties(desc: Dict[str, Any]) -> Dict[str, Any]:
        """
        Выполняет эвристическую оценку свойств молекулы на основе дескрипторов.
        """
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

        return {
            "solubility": solubility,
            "toxicity": toxicity,
            "drug_likeness": drug_likeness,
            "rigidity": rigidity,
            "lipinski_violations": violations,
            "comments": "Heuristic estimates based on extended RDKit descriptors.",
        }

    def run(self, smiles: str) -> Dict[str, Any]:
        """
        Основной метод выполнения анализа молекулы.

        Args:
            smiles: Строка SMILES.

        Returns:
            Словарь с исходным SMILES, дескрипторами и предсказанием.
        """
        smiles = (smiles or "").strip()
        if not smiles:
            return {"error": "Empty input. Please provide a SMILES string."}

        descriptors = self.compute_descriptors(smiles)
        prediction = self._predict_properties(descriptors)

        return {
            "input": smiles,
            "descriptors": descriptors,
            "prediction": prediction,
        }

    def as_tool(self):
        """
        Преобразует агента в LangChain tool.

        Tool можно передать в список tools для агента LangChain.
        Возвращает структурированный dict, который модель сможет читать как результат.
        """

        @tool("analyze_structure")
        def analyze_structure(smiles: str) -> dict:
            """
            Анализирует молекулярные свойства по SMILES.

            Args:
                smiles: SMILES-строка молекулы.

            Returns:
                Structured result with descriptors and heuristic property predictions.
            """
            return self.run(smiles)

        return analyze_structure

    def as_node(self):
        """
        Преобразует агента в функцию-узел для использования в графовых системах.
        """

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
