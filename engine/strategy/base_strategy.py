"""
Abstract base class for all investment and bonus strategies.

Strategies are always injected into Fund at construction — they are never
hardcoded inside Fund or any asset/liability model.  This is a hard
architectural rule (CLAUDE.md).

BaseStrategy is a minimal marker class.  The two concrete strategies have
different signatures and return types, so the ABC imposes no abstract methods
beyond a human-readable name.  Shared behaviour (logging, config validation)
can be added here as the model grows.
"""
from __future__ import annotations

from abc import ABC


class BaseStrategy(ABC):
    """
    Abstract base class for all strategy objects.

    Subclasses:
        InvestmentStrategy — SAA rebalancing (AC bond constraint enforced).
        BonusStrategy      — Policyholder bonus crediting.
    """

    @property
    def name(self) -> str:
        """Human-readable strategy name for logging and result attribution."""
        return self.__class__.__name__
