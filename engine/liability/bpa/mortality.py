"""
engine/liability/bpa/mortality.py — BPA mortality basis and decrement functions.

Proprietary implementation — stubbed in public demo.
MortalityBasis encapsulates the selected mortality table and CMI improvement
factors used for BPA liability projections. The production implementation
supports S3PXA/S3PNA base tables with CMI 2023 improvement factors.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class MortalityBasis:
    """
    Encapsulates a mortality table and improvement basis for BPA projections.

    Attributes
    ----------
    table_name : str
        Name of the base mortality table (e.g. 'S3PNA', 'S3PXA').
    improvement_model : str
        Name of the CMI improvement model (e.g. 'CMI_2023').
    base_year : int
        Calibration year for the improvement factors.
    long_term_rate : float
        Long-term annual improvement rate applied beyond the CMI projection period.
    """
    table_name: str = ""
    improvement_model: str = ""
    base_year: int = 2023
    long_term_rate: float = 0.015
    _q_table: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    _improvement_grid: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)


def q_x(*args: Any, **kwargs: Any) -> float:
    """Return the annual mortality rate q(x, t) for a given age and calendar year."""
    raise NotImplementedError("Proprietary implementation — not available in public demo.")


def survival_probs_variable_dt(*args: Any, **kwargs: Any) -> np.ndarray:
    """Compute survival probabilities over a variable-length timestep grid."""
    raise NotImplementedError("Proprietary implementation — not available in public demo.")


def improvement_factor(*args: Any, **kwargs: Any) -> float:
    """Return the cumulative CMI improvement factor from base year to target year."""
    raise NotImplementedError("Proprietary implementation — not available in public demo.")
