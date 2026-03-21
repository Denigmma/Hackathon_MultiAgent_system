from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

MODEL_AGENT = os.getenv("MODEL_AGENT", "openai/gpt-5.4-nano-thinking-xhigh")
MODEL_PROVIDER_AGENT = os.getenv("MODEL_PROVIDER_AGENT", "openai")
VSEGPT_API_KEY = os.getenv("VSEGPT_API_KEY", "")
BASE_URL = os.getenv("URL", "https://api.vsegpt.ru/v1")
AGENT_TIMEOUT_SECONDS = float(os.getenv("AGENT_TIMEOUT_SECONDS", "60"))


class StructurePropertiesAgent:
    """
    Агент для анализа молекул по SMILES.
    Tools:
    - validate_smiles
    - compute_descriptors
    - predict_properties
    """

    def __init__(
            self,
            model: str = MODEL_AGENT,
            temperature: float = 0.0,
    ) -> None:
        if not VSEGPT_API_KEY:
            raise ValueError("VSEGPT_API_KEY не задан в окружении.")

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
                "Ты химический ассистент. Анализируй SMILES и свойства молекулы. "
                "Используй tools по необходимости: validate_smiles, compute_descriptors, predict_properties. "
                "Верни только валидный JSON без markdown и без пояснений вокруг него. "
                "Формат ответа: "
                "{"
                '"input": string, '
                '"valid": boolean, '
                '"descriptors": object, '
                '"prediction": object'
                "}"
            ),
            name="structure_properties_agent",
        )

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
    def compute_descriptors(smiles: str) -> Dict[str, Any]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"error": "Invalid SMILES"}

        return {
            "MolWt": round(Descriptors.MolWt(mol), 4),
            "LogP": round(Crippen.MolLogP(mol), 4),
            "TPSA": round(rdMolDescriptors.CalcTPSA(mol), 4),
            "NumHDonors": Lipinski.NumHDonors(mol),
            "NumHAcceptors": Lipinski.NumHAcceptors(mol),
            "NumRotatableBonds": Lipinski.NumRotatableBonds(mol),
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

        return {
            "solubility": solubility,
            "toxicity": toxicity,
            "drug_likeness": drug_likeness,
            "rigidity": rigidity,
            "lipinski_violations": violations,
        }

    def _build_tools(self):
        agent_self = self

        @tool("validate_smiles")
        def validate_smiles(smiles: str) -> str:
            """Проверяет SMILES и возвращает канонический SMILES."""
            smiles = (smiles or "").strip()
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return json.dumps(
                    {"valid": False, "error": "Invalid SMILES"},
                    ensure_ascii=False,
                )

            return json.dumps(
                {
                    "valid": True,
                    "smiles": Chem.MolToSmiles(mol, canonical=True),
                },
                ensure_ascii=False,
            )

        @tool("compute_descriptors")
        def compute_descriptors(smiles: str) -> str:
            """Вычисляет молекулярные дескрипторы по SMILES."""
            result = agent_self.compute_descriptors(smiles)
            return json.dumps(result, ensure_ascii=False)

        @tool("predict_properties")
        def predict_properties(descriptors_json: str) -> str:
            """Оценивает свойства молекулы по дескрипторам."""
            try:
                desc = json.loads(descriptors_json)
            except Exception:
                return json.dumps(
                    {"error": "Invalid JSON payload"},
                    ensure_ascii=False,
                )

            result = agent_self._predict_properties(desc)
            return json.dumps(result, ensure_ascii=False)

        return [validate_smiles, compute_descriptors, predict_properties]

    def run(self, smiles: str) -> Dict[str, Any]:
        smiles = (smiles or "").strip()
        if not smiles:
            return {"error": "Empty SMILES"}

        state = self.agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": smiles,
                    }
                ]
            }
        )

        final_text = self._extract_final_text(state)
        parsed = self._loads_json(final_text)

        if "input" not in parsed:
            parsed["input"] = smiles

        return parsed

    def as_tool(self):
        @tool("analyze_structure")
        def analyze_structure(smiles: str) -> dict:
            return self.run(smiles)

        return analyze_structure

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
