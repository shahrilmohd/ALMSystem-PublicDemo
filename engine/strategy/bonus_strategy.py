"""
engine/strategy/bonus_strategy.py — With-profits PAR bonus crediting strategy.

Proprietary implementation — stubbed in public demo.
BonusStrategy manages asset share tracking, smoothed declared reversionary
bonus crediting, and terminal bonus computation for UK with-profits products.
SmoothedBonusStrategy implements a corridor-based smoothing algorithm.
"""
from __future__ import annotations
from abc import abstractmethod
from typing import Any
from engine.strategy.base_strategy import BaseStrategy


class BonusStrategy(BaseStrategy):
    """
    Abstract bonus strategy for with-profits (PAR) liability products.

    Manages asset share tracking, reversionary bonus declaration,
    and terminal bonus computation. Injected into DeterministicRun
    and StochasticRun when projecting with-profits funds.
    """

    @abstractmethod
    def update_smoothed_returns(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def declare_reversionary(self, *args: Any, **kwargs: Any) -> float:
        raise NotImplementedError

    @abstractmethod
    def compute_terminal_bonus_rate(self, *args: Any, **kwargs: Any) -> float:
        raise NotImplementedError


class SmoothedBonusStrategy(BonusStrategy):
    """
    Smoothed declared bonus strategy using an asset share corridor.

    Proprietary implementation — stubbed in public demo.
    Maintains a rolling window of earned returns, applies a smoothing
    corridor to determine the declared bonus rate, and computes
    terminal bonus at maturity or surrender.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Proprietary implementation — not available in public demo.")

    def update_smoothed_returns(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Proprietary implementation — not available in public demo.")

    def declare_reversionary(self, *args: Any, **kwargs: Any) -> float:
        raise NotImplementedError("Proprietary implementation — not available in public demo.")

    def compute_terminal_bonus_rate(self, *args: Any, **kwargs: Any) -> float:
        raise NotImplementedError("Proprietary implementation — not available in public demo.")
