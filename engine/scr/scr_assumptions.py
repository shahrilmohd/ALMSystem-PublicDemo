"""
engine/scr/scr_assumptions.py
==============================
SCRStressAssumptions — central table of all SII standard formula stress parameters.

Design (DECISIONS.md §63)
--------------------------
No stress engine carries a hardcoded default. Every calibrated parameter lives here.
Use SCRStressAssumptions.sii_standard_formula() for the SII Delegated Regulation
2015/35 prescribed defaults. Override individual fields with dataclasses.replace()
for sensitivity testing.

Correlation matrices are stored as tuple-of-tuples (immutable, hashable). Engines
convert them to np.ndarray once at construction.

Matrix dimensions
-----------------
market_corr  [3×3]: rows/cols = [interest, spread, currency]
life_corr    [4×4]: rows/cols = [mortality, longevity, lapse, expense]
  BPA:          scr_mortality=0.0 — longevity drives the life sub-SCR
  Conventional: scr_longevity=0.0 — mortality drives the life sub-SCR
  Same BSCRAggregator serves both without code changes (DECISIONS.md §63).
module_corr  [3×3]: rows/cols = [market, life, counterparty_default]
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SCRStressAssumptions:
    """
    Central table of all SII standard formula stress parameters for this model.

    Use SCRStressAssumptions.sii_standard_formula() for the regulatory defaults.
    Override individual fields with dataclasses.replace() for sensitivity testing.
    """

    # ---- Spread stress (SII DR Art 176–180) ----
    spread_up_bps: float
    spread_down_bps: float

    # ---- Interest rate stress (SII DR Art 166–169) ----
    rate_up_bps: float
    rate_down_bps: float

    # ---- Longevity stress (SII DR Art 137–138) ----
    longevity_mortality_stress_factor: float  # 0.20 = 20% permanent mortality improvement

    # ---- Lapse stress (SII DR Art 142–144) ----
    lapse_permanent_shock_factor: float  # 0.50 = ±50% permanent rate change
    lapse_mass_shock_factor: float       # 0.30 = 30% immediate TV/SV payment
    lapse_tv_to_bel_ratio: float         # TV as fraction of BEL (1.0 = TV ≈ BEL)

    # ---- Expense stress (SII DR Art 145–146) ----
    expense_loading_shock_factor: float  # 0.10 = 10% loading increase
    expense_inflation_shock_pa: float    # 0.01 = 1% additional p.a. inflation

    # ---- Currency stress (SII DR Art 188) ----
    currency_shock_factor: float  # 0.25 = 25% FX shock

    # ---- Operational risk (SII DR Art 218) ----
    op_risk_bscr_cap_factor: float  # 0.30 = 30% of BSCR cap
    op_risk_bel_factor: float       # 0.0045 = 0.45% of BEL

    # ---- Risk Margin (SII DR Art 37–39) ----
    cost_of_capital_rate: float  # 0.06 = 6% CoC

    # ---- Correlation matrices (SII DR 2015/35 Annexes IV–V) ----
    # Stored as tuple-of-tuples (row-major, immutable).
    # market_corr  [3×3]: [interest, spread, currency]
    market_corr: tuple[tuple[float, ...], ...]
    # life_corr    [4×4]: [mortality, longevity, lapse, expense]
    life_corr: tuple[tuple[float, ...], ...]
    # module_corr  [3×3]: [market, life, counterparty_default]
    module_corr: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if self.spread_up_bps < 0:
            raise ValueError(f"spread_up_bps must be >= 0, got {self.spread_up_bps}")
        if self.spread_down_bps < 0:
            raise ValueError(f"spread_down_bps must be >= 0, got {self.spread_down_bps}")
        if self.rate_up_bps < 0:
            raise ValueError(f"rate_up_bps must be >= 0, got {self.rate_up_bps}")
        if self.rate_down_bps < 0:
            raise ValueError(f"rate_down_bps must be >= 0, got {self.rate_down_bps}")
        if not (0.0 < self.longevity_mortality_stress_factor < 1.0):
            raise ValueError(
                f"longevity_mortality_stress_factor must be in (0, 1), "
                f"got {self.longevity_mortality_stress_factor}"
            )
        if not (0.0 <= self.lapse_permanent_shock_factor < 1.0):
            raise ValueError(
                f"lapse_permanent_shock_factor must be in [0, 1), "
                f"got {self.lapse_permanent_shock_factor}"
            )
        if not (0.0 <= self.lapse_mass_shock_factor <= 1.0):
            raise ValueError(
                f"lapse_mass_shock_factor must be in [0, 1], "
                f"got {self.lapse_mass_shock_factor}"
            )
        if self.lapse_tv_to_bel_ratio < 0.0:
            raise ValueError(
                f"lapse_tv_to_bel_ratio must be >= 0, got {self.lapse_tv_to_bel_ratio}"
            )
        if not (0.0 <= self.expense_loading_shock_factor):
            raise ValueError(
                f"expense_loading_shock_factor must be >= 0, "
                f"got {self.expense_loading_shock_factor}"
            )
        if self.expense_inflation_shock_pa < 0.0:
            raise ValueError(
                f"expense_inflation_shock_pa must be >= 0, "
                f"got {self.expense_inflation_shock_pa}"
            )
        if not (0.0 <= self.currency_shock_factor <= 1.0):
            raise ValueError(
                f"currency_shock_factor must be in [0, 1], "
                f"got {self.currency_shock_factor}"
            )
        if not (0.0 <= self.op_risk_bscr_cap_factor <= 1.0):
            raise ValueError(
                f"op_risk_bscr_cap_factor must be in [0, 1], "
                f"got {self.op_risk_bscr_cap_factor}"
            )
        if self.op_risk_bel_factor < 0.0:
            raise ValueError(
                f"op_risk_bel_factor must be >= 0, got {self.op_risk_bel_factor}"
            )
        if not (0.0 <= self.cost_of_capital_rate <= 1.0):
            raise ValueError(
                f"cost_of_capital_rate must be in [0, 1], "
                f"got {self.cost_of_capital_rate}"
            )
        _validate_corr_matrix(self.market_corr, "market_corr", expected_size=3)
        _validate_corr_matrix(self.life_corr, "life_corr", expected_size=4)
        _validate_corr_matrix(self.module_corr, "module_corr", expected_size=3)

    @classmethod
    def sii_standard_formula(cls) -> "SCRStressAssumptions":
        """
        Return the SII Delegated Regulation 2015/35 prescribed defaults.

        Sources
        -------
        Spread stress:   Art 176–180
        Interest stress: Art 166–169
        Longevity:       Art 137–138
        Lapse:           Art 142–144
        Expense:         Art 145–146
        Currency:        Art 188
        Op risk:         Art 218
        Risk Margin:     Art 37–39
        Correlations:    Annexes IV–V
        """
        return cls(
            spread_up_bps=75.0,
            spread_down_bps=25.0,
            rate_up_bps=100.0,
            rate_down_bps=100.0,
            longevity_mortality_stress_factor=0.20,
            lapse_permanent_shock_factor=0.50,
            lapse_mass_shock_factor=0.30,
            lapse_tv_to_bel_ratio=1.0,
            expense_loading_shock_factor=0.10,
            expense_inflation_shock_pa=0.01,
            currency_shock_factor=0.25,
            op_risk_bscr_cap_factor=0.30,
            op_risk_bel_factor=0.0045,
            cost_of_capital_rate=0.06,
            # SII DR Annex V — market: [interest, spread, currency]
            market_corr=(
                (1.00, 0.50, 0.25),
                (0.50, 1.00, 0.25),
                (0.25, 0.25, 1.00),
            ),
            # SII DR Annex IV — life: [mortality, longevity, lapse, expense]
            # corr(mortality, longevity) = -0.25 (opposite biometric risks)
            life_corr=(
                ( 1.00, -0.25,  0.00,  0.25),
                (-0.25,  1.00,  0.25,  0.25),
                ( 0.00,  0.25,  1.00,  0.50),
                ( 0.25,  0.25,  0.50,  1.00),
            ),
            # SII DR Annex IV — between-module: [market, life, counterparty_default]
            module_corr=(
                (1.00, 0.25, 0.25),
                (0.25, 1.00, 0.25),
                (0.25, 0.25, 1.00),
            ),
        )


def _validate_corr_matrix(
    matrix: tuple[tuple[float, ...], ...],
    name: str,
    expected_size: int,
) -> None:
    if len(matrix) != expected_size:
        raise ValueError(
            f"{name} must have {expected_size} rows, got {len(matrix)}"
        )
    for i, row in enumerate(matrix):
        if len(row) != expected_size:
            raise ValueError(
                f"{name} row {i} must have {expected_size} entries, got {len(row)}"
            )
        if not math.isclose(row[i], 1.0):
            raise ValueError(
                f"{name}[{i}][{i}] must be 1.0 (diagonal), got {row[i]}"
            )
    for i in range(expected_size):
        for j in range(expected_size):
            if not math.isclose(matrix[i][j], matrix[j][i], abs_tol=1e-9):
                raise ValueError(
                    f"{name} is not symmetric: [{i}][{j}]={matrix[i][j]} "
                    f"!= [{j}][{i}]={matrix[j][i]}"
                )
