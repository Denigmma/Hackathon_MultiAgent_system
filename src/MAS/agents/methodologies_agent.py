"""Совместимость со старым импортом.

Раньше в проекте существовал отдельный файл methodologies_agent.py c собственной
реализацией SynthesisProtocolSearchAgent. Это приводило к дублированию логики и
рассинхрону поведения. Теперь единственный источник истины — solver_agent.py.
"""

from .solver_agent import SynthesisProtocolSearchAgent

__all__ = ["SynthesisProtocolSearchAgent"]
