import json
from typing import Dict, Any

from langchain.tools import tool
from langchain.agents import create_openai_functions_agent, AgentExecutor, AgentType
from langchain.prompts import ChatPromptTemplate

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski


class StructurePropertiesAgent:
    """
    Агент для анализа структуры молекулы и предсказания свойств
    """

    def __init__(self, model=None, temperature: float = 0.0):
        self.llm = model

        self.tools = [self._build_tool()]
        self.agent = self._build_agent()

    @staticmethod
    def compute_rdkit_properties(smiles: str) -> Dict[str, Any]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        props = {
            "molecular_weight": Descriptors.MolWt(mol),
            "logP": Crippen.MolLogP(mol),
            "num_h_donors": Lipinski.NumHDonors(mol),
            "num_h_acceptors": Lipinski.NumHAcceptors(mol),
            "num_rotatable_bonds": Lipinski.NumRotatableBonds(mol),
            "tpsa": Descriptors.TPSA(mol),
        }

        return props

    def _build_tool(self):
        @tool
        def predict_structure_properties(smiles: str) -> str:
            """
            Predicts molecular properties from SMILES (Simplified Molecular-Input Line-Entry System) using RDKit.
            """
            try:
                props = self.compute_rdkit_properties(smiles)

                # эвристики
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

                return json.dumps(props, indent=2)

            except Exception as e:
                return json.dumps({"error": str(e)})

        return predict_structure_properties

    def _build_agent(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a chemistry assistant."),
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

    def run(self, smiles: str) -> Dict[str, Any]:
        response = self.agent.run(
            f"Calculate molecular properties for this SMILES: {smiles}"
        )

        try:
            return json.loads(response)
        except:
            return {"raw_output": response}

    def as_node(self):
        def node(state: dict) -> dict:
            smiles = state.get("target_molecule")

            result = self.run(smiles)

            state["properties"] = result

            state.setdefault("history", []).append({
                "agent": "structure_properties",
                "input": smiles,
                "output": result
            })

            return state

        return node


if __name__ == "__main__":
    agent = StructurePropertiesAgent()

    result = agent.run("CCO")
    print(json.dumps(result, indent=2))
