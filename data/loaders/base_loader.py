"""
BaseLoader — abstract base class for all data loaders.

Purpose
-------
Every data loader (liability, asset, scenario) follows the same three-step
contract:

    1. load()         — read raw data from a source (file, database, etc.)
    2. validate()     — check raw data for required fields and valid values
    3. to_dataframe() — return a clean, typed pandas DataFrame

Callers (run modes, CLI, tests) always call these in the same order.  The
ABC enforces the contract at class definition time — a subclass that does not
implement all three methods cannot be instantiated.

Architecture note
-----------------
Loaders live in data/, not in engine/.  They are the only layer that knows
about files and databases.  The engine never imports from data/.  Data flows
in one direction:

    data/ (loaders) → engine/ (run modes)

A loader reads, validates, and hands off a clean DataFrame.  It never knows
what the engine will do with that data.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import pandas as pd


class BaseLoader(ABC):
    """
    Abstract base class for all ALM data loaders.

    Subclasses must implement load(), validate(), and to_dataframe().
    Call them in that order: load → validate → to_dataframe.

    The logger is available as self._logger in all subclasses.
    """

    def __init__(self) -> None:
        self._logger = logging.getLogger(
            self.__class__.__module__ + "." + self.__class__.__name__
        )
        self._raw: pd.DataFrame | None = None

    # -----------------------------------------------------------------------
    # Abstract interface — subclasses must implement all three
    # -----------------------------------------------------------------------

    @abstractmethod
    def load(self) -> None:
        """
        Read raw data from the source and store it in self._raw.

        After this call, self._raw is a pandas DataFrame containing the
        unvalidated data exactly as it was read from the source.

        Raises
        ------
        FileNotFoundError
            If a required file does not exist.
        IOError
            On any read failure.
        """

    @abstractmethod
    def validate(self) -> None:
        """
        Validate self._raw.  Raise ValueError with a descriptive message on
        any rule violation.

        Must be called after load().

        Raises
        ------
        ValueError
            If any required column is missing, any value is out of range,
            or any consistency rule is violated.
        RuntimeError
            If called before load().
        """

    @abstractmethod
    def to_dataframe(self) -> pd.DataFrame:
        """
        Return a clean, typed DataFrame ready for injection into a run mode.

        Must be called after load() and validate().

        Returns
        -------
        pd.DataFrame
            Validated and typed model point data.

        Raises
        ------
        RuntimeError
            If called before load().
        """

    # -----------------------------------------------------------------------
    # Shared guard — prevents calling validate/to_dataframe before load
    # -----------------------------------------------------------------------

    def _require_loaded(self, caller: str) -> None:
        """Raise RuntimeError if self._raw has not been set by load()."""
        if self._raw is None:
            raise RuntimeError(
                f"{self.__class__.__name__}.{caller}() called before load(). "
                "Call load() first."
            )
