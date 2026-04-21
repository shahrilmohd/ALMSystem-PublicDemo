"""
BPADataLoader — loads mortality tables and BPA model point files.

Two concerns are handled here:

1. Mortality basis construction
   ----------------------------
   load_mortality_basis(dir) reads four CSV files from a directory and
   constructs a MortalityBasis for the BPA engine:

     S3PMA.csv          age, qx   (S3PMA ultimate male rates)
     S3PFA.csv          age, qx   (S3PFA ultimate female rates)
     CMI_2023_M.csv     age, initial_rate  (CMI male improvement rates)
     CMI_2023_F.csv     age, initial_rate  (CMI female improvement rates)

   All four files must cover exactly ages 16–120 (TABLE_LENGTH = 105 rows).
   age column is used only for ordering; the array index is age - MIN_TABLE_AGE.

2. Model point loading
   -------------------
   BPADataLoader loads a single CSV/TSV file of BPA model points and validates
   it against the appropriate population type:

     "in_payment"   → BPAValidator.validate_in_payment
     "enhanced"     → BPAValidator.validate_enhanced
     "deferred"     → BPAValidator.validate_deferred
     "dependant"    → BPAValidator.validate_dependant

   The population_type is specified at construction time.

   An optional BPADealRegistry can be supplied.  When present, validate()
   cross-references every deal_id value in the file against the registry
   (DECISIONS.md §44).  When None, the deal_id format check still runs
   but the registry cross-reference is skipped (useful for isolated tests).

   Type coercion:
     float columns: age, weight, pension_pa, lpi_cap, lpi_floor, gmp_pa,
                    deferred_pension_pa, era, nra, revaluation_cap,
                    revaluation_floor, deferment_years, rating_years,
                    member_age, dependant_age
     str columns: mp_id, deal_id, sex, revaluation_type, member_sex,
                  dependant_sex, tranche_id (whitespace stripped)

Usage
-----
    # Mortality basis
    basis = BPADataLoader.load_mortality_basis("data/mortality/")

    # Model points — no registry (isolated tests)
    loader = BPADataLoader("model_points_in_payment.csv", "in_payment")
    loader.load()
    loader.validate()
    df = loader.to_dataframe()

    # Model points — with registry (production)
    registry = BPADealRegistry.from_csv("data/bpa/bpa_deals.csv")
    loader = BPADataLoader("model_points_in_payment.csv", "in_payment",
                           registry=registry)
    loader.load()
    loader.validate()
    df = loader.to_dataframe()
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional

import numpy as np
import pandas as pd

from data.loaders.base_loader import BaseLoader
from data.validators.bpa_validator import BPAValidator
from engine.liability.bpa.mortality import MortalityBasis, MIN_TABLE_AGE, TABLE_LENGTH

if TYPE_CHECKING:
    from engine.liability.bpa.registry import BPADealRegistry

# Population types the loader accepts
PopulationType = Literal["in_payment", "enhanced", "deferred", "dependant"]

# Columns to cast to float (all numeric MP columns across all population types)
_FLOAT_COLUMNS = (
    "age", "in_force_count", "weight",   # weight kept for dependant population
    "pension_pa", "lpi_cap", "lpi_floor", "gmp_pa",
    "deferred_pension_pa", "era", "nra", "revaluation_cap",
    "revaluation_floor", "deferment_years", "rating_years",
    "member_age", "dependant_age",
)

# String columns that need whitespace stripping beyond what _strip_strings catches
# (deal_id and tranche_id are object dtype and handled by the generic strip)
_STR_COLUMNS = (
    "mp_id", "deal_id", "sex", "revaluation_type",
    "member_sex", "dependant_sex", "tranche_id",
)

# Mortality CSV filenames (relative to the mortality directory)
_S3PMA_FILE  = "S3PMA.csv"
_S3PFA_FILE  = "S3PFA.csv"
_CMI_M_FILE  = "CMI_2023_M.csv"
_CMI_F_FILE  = "CMI_2023_F.csv"


class BPADataLoader(BaseLoader):
    """
    Loads and validates BPA model point CSV/TSV files.

    Parameters
    ----------
    file_path : str or Path
        Path to the model point file (.csv or .tsv).
    population_type : str
        One of "in_payment", "enhanced", "deferred", "dependant".
    column_map : dict, optional
        Source-to-engine column name mapping applied before validation.
    """

    def __init__(
        self,
        file_path: str | Path,
        population_type: PopulationType,
        column_map: Optional[dict[str, str]] = None,
        registry: "BPADealRegistry | None" = None,
    ) -> None:
        super().__init__()
        self._file_path       = Path(file_path)
        self._population_type = population_type
        self._column_map      = column_map or {}
        self._registry        = registry

    # ------------------------------------------------------------------
    # BaseLoader contract
    # ------------------------------------------------------------------

    def load(self) -> None:
        """
        Read the model point file and apply the column map.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the file extension is not .csv or .tsv.
        """
        if not self._file_path.exists():
            raise FileNotFoundError(
                f"BPA model point file not found: {self._file_path}"
            )

        suffix = self._file_path.suffix.lower()
        if suffix == ".csv":
            sep = ","
        elif suffix == ".tsv":
            sep = "\t"
        else:
            raise ValueError(
                f"Unsupported file format '{suffix}'. Use .csv or .tsv."
            )

        raw = pd.read_csv(self._file_path, sep=sep)
        raw = self._rename_columns(raw)
        self._raw = self._strip_strings(raw)
        self._logger.info(
            "Loaded %d BPA %s rows from %s",
            len(self._raw), self._population_type, self._file_path,
        )

    def validate(self) -> None:
        """
        Validate the loaded model points using BPAValidator.

        Raises
        ------
        RuntimeError
            If called before load().
        ValueError
            If any validation rule is violated.
        """
        self._require_loaded("validate")
        assert self._raw is not None
        validator = {
            "in_payment": BPAValidator.validate_in_payment,
            "enhanced":   BPAValidator.validate_enhanced,
            "deferred":   BPAValidator.validate_deferred,
            "dependant":  BPAValidator.validate_dependant,
        }[self._population_type]
        validator(self._raw, registry=self._registry)
        self._logger.info(
            "BPA %s validation passed: %d rows", self._population_type, len(self._raw)
        )

    def to_dataframe(self) -> pd.DataFrame:
        """
        Return a clean, typed DataFrame ready for engine injection.

        Raises
        ------
        RuntimeError
            If called before load().
        """
        self._require_loaded("to_dataframe")
        assert self._raw is not None
        df = self._raw.copy()

        for col in _FLOAT_COLUMNS:
            if col in df.columns:
                df[col] = df[col].astype(float)

        return df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._column_map:
            return df
        return df.rename(columns=self._column_map)

    @staticmethod
    def _strip_strings(df: pd.DataFrame) -> pd.DataFrame:
        str_cols = df.select_dtypes(include=["object", "str"]).columns
        df = df.copy()
        for col in str_cols:
            df[col] = df[col].str.strip()
        return df

    # ------------------------------------------------------------------
    # Class method: load mortality basis from directory
    # ------------------------------------------------------------------

    @classmethod
    def load_mortality_basis(
        cls,
        mortality_dir: str | Path,
        base_year: int = 2023,
        ltr: float = 0.01,
        convergence_period: int = 20,
        ae_ratio_male: float = 1.0,
        ae_ratio_female: float = 1.0,
    ) -> MortalityBasis:
        """
        Read S3PMA/S3PFA ultimate tables and CMI improvement rates from CSV files
        and return a fully constructed MortalityBasis.

        Parameters
        ----------
        mortality_dir : str or Path
            Directory containing S3PMA.csv, S3PFA.csv, CMI_2023_M.csv, CMI_2023_F.csv.
        base_year : int
            Calendar year of the CMI improvement rates (default 2023).
        ltr : float
            Long-term improvement rate (default 0.01).
        convergence_period : int
            Years over which initial improvement converges to LTR (default 20).
        ae_ratio_male, ae_ratio_female : float
            A/E ratio adjustments applied to the base tables (default 1.0).

        Returns
        -------
        MortalityBasis

        Raises
        ------
        FileNotFoundError
            If any required CSV file is missing.
        ValueError
            If any table has wrong row count or age range.
        """
        directory = Path(mortality_dir)

        base_m  = cls._load_mortality_table(directory / _S3PMA_FILE, "qx")
        base_f  = cls._load_mortality_table(directory / _S3PFA_FILE, "qx")
        impr_m  = cls._load_mortality_table(directory / _CMI_M_FILE, "initial_rate")
        impr_f  = cls._load_mortality_table(directory / _CMI_F_FILE, "initial_rate")

        return MortalityBasis(
            base_table_male=base_m,
            base_table_female=base_f,
            initial_improvement_male=impr_m,
            initial_improvement_female=impr_f,
            base_year=base_year,
            ltr=ltr,
            convergence_period=convergence_period,
            ae_ratio_male=ae_ratio_male,
            ae_ratio_female=ae_ratio_female,
        )

    @classmethod
    def _load_mortality_table(cls, path: Path, value_col: str) -> np.ndarray:
        """
        Load a mortality or improvement CSV and return a float array of length TABLE_LENGTH.

        The CSV must have columns 'age' and value_col (e.g. 'qx' or 'initial_rate').
        Rows are sorted by age, clipped to [MIN_TABLE_AGE, MIN_TABLE_AGE + TABLE_LENGTH - 1],
        and returned as a 1-D numpy array indexed by (age - MIN_TABLE_AGE).

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the resulting array length != TABLE_LENGTH or if required columns are missing.
        """
        if not path.exists():
            raise FileNotFoundError(f"Mortality table not found: {path}")

        df = pd.read_csv(path)
        missing_cols = {"age", value_col} - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"{path.name}: missing columns {missing_cols}. "
                f"Expected 'age' and '{value_col}'."
            )

        max_age = MIN_TABLE_AGE + TABLE_LENGTH - 1
        df = df[(df["age"] >= MIN_TABLE_AGE) & (df["age"] <= max_age)].copy()
        df = df.sort_values("age").reset_index(drop=True)

        if len(df) != TABLE_LENGTH:
            raise ValueError(
                f"{path.name}: expected {TABLE_LENGTH} rows for ages "
                f"{MIN_TABLE_AGE}–{max_age}, got {len(df)}."
            )

        return df[value_col].astype(float).to_numpy()
