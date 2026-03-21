import json
from typing import Dict, Any

from langchain.tools import tool
from langchain.prompts import ChatPromptTemplate
from langchain.agents import create_openai_functions_agent, AgentExecutor

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski


class StructurePropertiesAgent:
    """
    Агент для анализа структуры молекулы (SMILES)
    и предсказания базовых физико-химических свойств.
    """

    def __init__(self, llm):
        self.llm = llm
        self.tools = self._build_tools()
        self.agent = self._build_agent()

    def _build_tools(self):
        @tool
        def compute_descriptors(smiles: str) -> Dict[str, Any]:
            """
            Вычисляет базовые дескрипторы молекулы по SMILES.
            """
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {"error": "Invalid SMILES"}

            descriptors = {
                "MolWt": Descriptors.MolWt(mol),
                "LogP": Crippen.MolLogP(mol),
                "NumHDonors": Lipinski.NumHDonors(mol),
                "NumHAcceptors": Lipinski.NumHAcceptors(mol),
                "TPSA": Descriptors.TPSA(mol),
                "NumRotatableBonds": Lipinski.NumRotatableBonds(mol),
                "NumHeavyAtoms": Descriptors.HeavyAtomCount(mol),
                "RingCount": Descriptors.RingCount(mol),
            }

            return descriptors

        @tool
        def predict_properties(descriptors: Dict[str, Any]) -> Dict[str, Any]:
            """
            Использует LLM для интерпретации дескрипторов
            и прогнозирования свойств молекулы.
            """
            prompt = ChatPromptTemplate.from_messages([
                ("system",
                 "Ты химический эксперт. По набору дескрипторов молекулы оцени её свойства."),
                ("user",
                 "Дескрипторы:\n{descriptors}\n\n"
                 "Верни JSON со следующими полями:\n"
                 "- solubility (low/medium/high)\n"
                 "- toxicity (low/medium/high)\n"
                 "- drug_likeness (low/medium/high)\n"
                 "- comments")
            ])

            chain = prompt | self.llm

            response = chain.invoke({
                "descriptors": json.dumps(descriptors, indent=2)
            })

            try:
                return json.loads(response.content)
            except Exception:
                return {"raw_output": response.content}

        return [compute_descriptors, predict_properties]

    def _build_agent(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Ты агент для анализа молекул. "
             "Сначала вычисляешь дескрипторы, затем прогнозируешь свойства."),
            ("user", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])

        agent = create_openai_functions_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )

        return AgentExecutor(agent=agent, tools=self.tools, verbose=True)

    def run(self, smiles: str) -> Dict[str, Any]:
        """
        Запуск агента на входном SMILES.
        """
        input_query = f"SMILES: {smiles}"
        result = self.agent.invoke({"input": input_query})
        return result


if __name__ == "__main__":
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    agent = StructurePropertiesAgent(llm)

    result = agent.run("CCO")
    print(result)
