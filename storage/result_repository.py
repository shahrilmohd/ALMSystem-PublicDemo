"""
ResultRepository — persistence for projection timestep results.

Architectural rule (DECISIONS.md §29):
    Nothing outside storage/ calls SQLAlchemy directly.
    All result record access goes through this class.

Bulk insert strategy (DECISIONS.md §29):
    save_all() uses session.bulk_insert_mappings() in batches of 500 rows.
    For a 1,000-scenario × 120-month stochastic run (120,000 rows) this means
    240 INSERT statements rather than 120,000.

Usage
-----
    with Session() as session:
        repo = ResultRepository(session)
        repo.save_all("run_001", result_store)
        session.commit()

        df = repo.get_dataframe("run_001")
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
from sqlalchemy import insert, select
from sqlalchemy.orm import Session

from engine.results.result_store import RESULT_COLUMNS, ResultStore
from storage.models.result_record import ResultRecord


_BATCH_SIZE = 500


class ResultRepository:
    """
    Bulk-write and query operations for ResultRecord (result_records table).

    Parameters
    ----------
    session : sqlalchemy.orm.Session
        Active database session.  The caller owns session lifecycle.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save_all(self, run_id: str, store: ResultStore) -> None:
        """
        Bulk-insert all results from a ResultStore.

        Converts the store to a flat DataFrame (via as_dataframe()), then
        writes rows to the database in batches of 500.  The caller must
        call session.commit() to persist.

        Parameters
        ----------
        run_id : str
            Must match store's run_id and an existing RunRecord.run_id.
        store : ResultStore
            Completed result store from a projection run.

        Raises
        ------
        ValueError
            If store.run_id does not match run_id.
        """
        if store.run_id != run_id:
            raise ValueError(
                f"store.run_id={store.run_id!r} does not match run_id={run_id!r}."
            )

        df = store.as_dataframe()
        if df.empty:
            return

        # Build list of plain dicts — one per row, matching ORM column names.
        mappings: list[dict] = []
        for _, row in df.iterrows():
            mappings.append({
                "run_id":              run_id,
                "scenario_id":        int(row["scenario_id"]),
                "timestep":           int(row["timestep"]),
                "cohort_id":          row["cohort_id"] if pd.notna(row["cohort_id"]) else None,
                "premiums":           float(row["premiums"]),
                "death_claims":       float(row["death_claims"]),
                "surrender_payments": float(row["surrender_payments"]),
                "maturity_payments":  float(row["maturity_payments"]),
                "expenses":           float(row["expenses"]),
                "net_outgo":          float(row["net_outgo"]),
                "in_force_start":     float(row["in_force_start"]),
                "deaths":             float(row["deaths"]),
                "lapses":             float(row["lapses"]),
                "maturities":         float(row["maturities"]),
                "in_force_end":       float(row["in_force_end"]),
                "bel":                float(row["bel"]),
                "reserve":            float(row["reserve"]),
                "total_market_value": _nullable_float(row["total_market_value"]),
                "total_book_value":   _nullable_float(row["total_book_value"]),
                "cash_balance":       _nullable_float(row["cash_balance"]),
                "eir_income":         _nullable_float(row["eir_income"]),
                "coupon_income":      _nullable_float(row["coupon_income"]),
                "dividend_income":    _nullable_float(row["dividend_income"]),
                "unrealised_gl":      _nullable_float(row["unrealised_gl"]),
                "realised_gl":        _nullable_float(row["realised_gl"]),
                "oci_reserve":        _nullable_float(row["oci_reserve"]),
                "mv_ac":              _nullable_float(row["mv_ac"]),
                "mv_fvtpl":           _nullable_float(row["mv_fvtpl"]),
                "mv_fvoci":           _nullable_float(row["mv_fvoci"]),
                "bel_pre_ma":         _nullable_float(row["bel_pre_ma"]),
                "bel_post_ma":        _nullable_float(row["bel_post_ma"]),
            })

        # Bulk-insert in batches of _BATCH_SIZE
        for start in range(0, len(mappings), _BATCH_SIZE):
            batch = mappings[start : start + _BATCH_SIZE]
            self._session.execute(insert(ResultRecord), batch)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_dataframe(self, run_id: str) -> pd.DataFrame:
        """
        All results for a run as a flat DataFrame.

        Returns a DataFrame with columns matching RESULT_COLUMNS, sorted by
        (scenario_id, cohort_id, timestep).  Returns an empty DataFrame with
        the correct schema if the run has no stored results.
        """
        stmt = (
            select(ResultRecord)
            .where(ResultRecord.run_id == run_id)
            .order_by(
                ResultRecord.scenario_id,
                ResultRecord.cohort_id,
                ResultRecord.timestep,
            )
        )
        records = list(self._session.scalars(stmt))
        if not records:
            return pd.DataFrame(columns=list(RESULT_COLUMNS))
        return pd.DataFrame([_record_to_dict(r) for r in records])

    def get_scenario(
        self,
        run_id: str,
        scenario_id: int,
        cohort_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Results for a single (run_id, scenario_id, cohort_id) combination,
        sorted by timestep ascending.

        Parameters
        ----------
        run_id      : str
        scenario_id : int
        cohort_id   : str | None
            None (default) for non-BPA runs.

        Returns an empty DataFrame with correct schema if not found.
        """
        stmt = (
            select(ResultRecord)
            .where(
                ResultRecord.run_id == run_id,
                ResultRecord.scenario_id == scenario_id,
                ResultRecord.cohort_id == cohort_id,
            )
            .order_by(ResultRecord.timestep)
        )
        records = list(self._session.scalars(stmt))
        if not records:
            return pd.DataFrame(columns=list(RESULT_COLUMNS))
        return pd.DataFrame([_record_to_dict(r) for r in records])

    def result_count(self, run_id: str) -> int:
        """Number of result rows stored for a given run_id."""
        stmt = select(ResultRecord).where(ResultRecord.run_id == run_id)
        return len(list(self._session.scalars(stmt)))


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _nullable_float(value) -> Optional[float]:
    """Convert a pandas value to float, or None if the value is NaN/None."""
    if value is None:
        return None
    try:
        f = float(value)
        return None if pd.isna(f) else f
    except (TypeError, ValueError):
        return None


def _record_to_dict(r: ResultRecord) -> dict:
    """Flatten a ResultRecord ORM object into a plain dict matching RESULT_COLUMNS."""
    return {
        "run_id":              r.run_id,
        "scenario_id":        r.scenario_id,
        "timestep":           r.timestep,
        "cohort_id":          r.cohort_id,
        "premiums":           r.premiums,
        "death_claims":       r.death_claims,
        "surrender_payments": r.surrender_payments,
        "maturity_payments":  r.maturity_payments,
        "expenses":           r.expenses,
        "net_outgo":          r.net_outgo,
        "in_force_start":     r.in_force_start,
        "deaths":             r.deaths,
        "lapses":             r.lapses,
        "maturities":         r.maturities,
        "in_force_end":       r.in_force_end,
        "bel":                r.bel,
        "reserve":            r.reserve,
        "total_market_value": r.total_market_value,
        "total_book_value":   r.total_book_value,
        "cash_balance":       r.cash_balance,
        "eir_income":         r.eir_income,
        "coupon_income":      r.coupon_income,
        "dividend_income":    r.dividend_income,
        "unrealised_gl":      r.unrealised_gl,
        "realised_gl":        r.realised_gl,
        "oci_reserve":        r.oci_reserve,
        "mv_ac":              r.mv_ac,
        "mv_fvtpl":           r.mv_fvtpl,
        "mv_fvoci":           r.mv_fvoci,
        "bel_pre_ma":         r.bel_pre_ma,
        "bel_post_ma":        r.bel_post_ma,
    }
