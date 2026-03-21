from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Protocol, Union
import json


@dataclass
class Substance:
    """
    Описание одного вещества.
    """
    name: str  # Название вещества
    formula: Optional[str] = None  # Химическая формула, если известна
    amount: Optional[float] = None  # Количество вещества
    amount_unit: Optional[str] = None  # "g", "mol", "ml", "L"
    purity: Optional[float] = None  # Чистота в долях, например 0.98
    role: Optional[str] = None  # "reactant", "solvent", "catalyst", "impurity"


@dataclass
class MixConditions:
    """
    Условия смешения.
    """
    temperature_c: Optional[float] = None
    pressure_atm: Optional[float] = None
    solvent: Optional[str] = None
    pH: Optional[float] = None
    time_min: Optional[float] = None
    stirring: Optional[bool] = None


@dataclass
class AgentResult:
    """
    Итог ответа агента.
    """
    reaction_likely: bool
    summary: str
    products: List[Dict[str, Any]] = field(default_factory=list)
    mixture_composition: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    raw: Optional[Dict[str, Any]] = None


# ----------------------------
# Интерфейс LLM
# ----------------------------

class LLMClient(Protocol):
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        ...


# ----------------------------
# Агрегатор масс/составов
# ----------------------------

class CompositionCalculator:
    """
    Считает массовые доли, если известны массы веществ.
    """

    @staticmethod
    def mass_fractions(substances: List[Substance]) -> List[Dict[str, Any]]:
        masses = []
        for s in substances:
            if s.amount is None or s.amount_unit != "g":
                continue
            if s.purity is not None:
                masses.append((s.name, s.amount * s.purity))
            else:
                masses.append((s.name, s.amount))

        total_mass = sum(m for _, m in masses)
        if total_mass <= 0:
            return []

        result = []
        for name, mass in masses:
            result.append({
                "name": name,
                "mass_g": round(mass, 6),
                "mass_fraction_percent": round((mass / total_mass) * 100, 3)
            })
        return result


# ----------------------------
# Основной агент
# ----------------------------

class MixtureReactionAgent:
    """
    Агент, который:
    - принимает список веществ;
    - учитывает условия;
    - пытается оценить, будет ли реакция;
    - возвращает вероятные продукты и/или состав смеси.

    Важно:
    - точный прогноз возможен только при наличии формул, количеств, условий и, желательно, базы реакций;
    - если данных мало, агент должен честно сказать, что ответ вероятностный.
    """

    def __init__(
            self,
            llm: Optional[LLMClient] = None,
            max_substances: int = 20,
    ):
        self.llm = llm
        self.max_substances = max_substances
        self.calculator = CompositionCalculator()

    def analyze(
            self,
            substances: List[Substance],
            conditions: Optional[MixConditions] = None,
            context: Optional[str] = None,
    ) -> AgentResult:
        """
        Основной метод анализа.
        """
        self._validate_inputs(substances, conditions)

        # Если LLM подключена — просим сделать химический вывод в структурированном виде.
        if self.llm is not None:
            llm_result = self._analyze_with_llm(substances, conditions, context)
            if llm_result is not None:
                return llm_result

        # Если LLM нет — хотя бы считаем состав по массе.
        mixture = self.calculator.mass_fractions(substances)

        return AgentResult(
            reaction_likely=False,
            summary=(
                "Недостаточно данных для уверенного предсказания реакции. "
                "Возвращён только массовый состав входной смеси."
            ),
            products=[],
            mixture_composition=mixture,
            warnings=[
                "Для точного прогноза нужны формулы, количества, растворитель, температура и pH.",
                "Без базы реакций или модели предсказания агент может только оценивать вероятный состав."
            ],
            raw=None
        )

    def _validate_inputs(
            self,
            substances: List[Substance],
            conditions: Optional[MixConditions],
    ) -> None:
        if not substances:
            raise ValueError("Список веществ не должен быть пустым.")
        if len(substances) > self.max_substances:
            raise ValueError(f"Слишком много веществ: максимум {self.max_substances}.")

        for s in substances:
            if not s.name or not s.name.strip():
                raise ValueError("У каждого вещества должно быть имя.")
            if s.amount is not None and s.amount < 0:
                raise ValueError(f"Количество вещества '{s.name}' не может быть отрицательным.")
            if s.purity is not None and not (0 <= s.purity <= 1):
                raise ValueError(f"Чистота '{s.name}' должна быть в диапазоне 0..1.")

        if conditions is not None:
            if conditions.pH is not None and not (0 <= conditions.pH <= 14):
                raise ValueError("pH должен быть в диапазоне 0..14.")

    def _build_prompt(
            self,
            substances: List[Substance],
            conditions: Optional[MixConditions],
            context: Optional[str],
    ) -> List[Dict[str, str]]:
        payload = {
            "substances": [s.__dict__ for s in substances],
            "conditions": conditions.__dict__ if conditions else None,
            "context": context,
        }

        system = (
            "Ты помощник для анализа химических смесей. "
            "Нужно отвечать осторожно и структурированно. "
            "Если данных недостаточно, прямо скажи об этом. "
            "Не выдумывай точный состав без оснований. "
            "Верни только JSON с полями: reaction_likely, summary, products, mixture_composition, warnings."
        )

        user = (
            "Проанализируй смесь веществ и оцени, что получится после смешения.\n"
            "Верни вероятные продукты, если реакция возможна, иначе опиши состав смеси.\n"
            f"Данные:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
        )

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def _analyze_with_llm(
            self,
            substances: List[Substance],
            conditions: Optional[MixConditions],
            context: Optional[str],
    ) -> Optional[AgentResult]:
        messages = self._build_prompt(substances, conditions, context)
        response_text = self.llm.chat(messages)

        try:
            data = json.loads(response_text)
        except Exception:
            # Если модель ответила не JSON, возвращаем fallback.
            mixture = self.calculator.mass_fractions(substances)
            return AgentResult(
                reaction_likely=False,
                summary="LLM вернула невалидный JSON. Возвращён состав смеси по массе.",
                products=[],
                mixture_composition=mixture,
                warnings=["Ответ модели не удалось распарсить как JSON."],
                raw={"llm_response": response_text},
            )

        return AgentResult(
            reaction_likely=bool(data.get("reaction_likely", False)),
            summary=str(data.get("summary", "")),
            products=data.get("products", []) or [],
            mixture_composition=data.get("mixture_composition", []) or [],
            warnings=data.get("warnings", []) or [],
            raw=data,
        )


if __name__ == "__main__":
    substances = [
        Substance(name="NaCl", formula="NaCl", amount=10, amount_unit="g", purity=1.0, role="reactant"),
        Substance(name="H2O", formula="H2O", amount=90, amount_unit="g", purity=1.0, role="solvent"),
    ]

    conditions = MixConditions(
        temperature_c=25,
        pressure_atm=1,
        solvent="water",
        pH=7,
        time_min=10,
        stirring=True,
    )

    agent = MixtureReactionAgent(llm=None)
    result = agent.analyze(substances, conditions)

    print(result.summary)
    print(result.mixture_composition)
