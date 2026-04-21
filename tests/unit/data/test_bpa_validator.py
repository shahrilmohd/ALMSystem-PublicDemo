"""
Unit tests for data/validators/bpa_validator.py
"""
import pandas as pd
import pytest

from data.validators.bpa_validator import BPAValidator
from engine.liability.bpa.registry import BPADealMetadata, BPADealRegistry
from datetime import date


# ---------------------------------------------------------------------------
# Shared registry fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def registry() -> BPADealRegistry:
    return BPADealRegistry([
        BPADealMetadata(
            deal_id        = "AcmePension_2024Q3",
            deal_type      = "buyout",
            inception_date = date(2024, 9, 30),
            deal_name      = "Acme Pension Scheme Ltd Q3 2024",
            ma_eligible    = True,
        ),
    ])


# ---------------------------------------------------------------------------
# Helpers — minimal valid DataFrames for each population type
# ---------------------------------------------------------------------------

def valid_in_payment(**overrides):
    row = {
        "mp_id": "IP001", "deal_id": "AcmePension_2024Q3",
        "sex": "M", "age": 70.0, "in_force_count": 1.0,
        "pension_pa": 12000.0, "lpi_cap": 0.05, "lpi_floor": 0.0, "gmp_pa": 0.0,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def valid_enhanced(**overrides):
    row = {
        "mp_id": "EN001", "deal_id": "AcmePension_2024Q3",
        "sex": "F", "age": 65.0, "in_force_count": 1.0,
        "pension_pa": 8000.0, "lpi_cap": 0.05, "lpi_floor": 0.0,
        "gmp_pa": 0.0, "rating_years": 5.0,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def valid_deferred(**overrides):
    row = {
        "mp_id": "DEF001", "deal_id": "AcmePension_2024Q3",
        "sex": "M", "age": 55.0, "in_force_count": 1.0,
        "deferred_pension_pa": 5000.0, "era": 55.0, "nra": 65.0,
        "revaluation_type": "CPI", "revaluation_cap": 0.05, "revaluation_floor": 0.0,
        "deferment_years": 10.0, "tv_eligible": 1,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def valid_dependant(**overrides):
    row = {
        "mp_id": "DEP001", "deal_id": "AcmePension_2024Q3",
        "member_sex": "M", "member_age": 70.0,
        "dependant_sex": "F", "dependant_age": 67.0,
        "weight": 0.6, "pension_pa": 6000.0, "lpi_cap": 0.05, "lpi_floor": 0.0,
    }
    row.update(overrides)
    return pd.DataFrame([row])


# ---------------------------------------------------------------------------
# in_payment
# ---------------------------------------------------------------------------

class TestInPayment:

    def test_valid_passes(self):
        BPAValidator.validate_in_payment(valid_in_payment())

    def test_missing_column_raises(self):
        df = valid_in_payment().drop(columns=["pension_pa"])
        with pytest.raises(ValueError, match="Missing required columns"):
            BPAValidator.validate_in_payment(df)

    def test_missing_deal_id_raises(self):
        df = valid_in_payment().drop(columns=["deal_id"])
        with pytest.raises(ValueError, match="Missing required columns"):
            BPAValidator.validate_in_payment(df)

    def test_negative_age_raises(self):
        with pytest.raises(ValueError, match="age"):
            BPAValidator.validate_in_payment(valid_in_payment(age=-1.0))

    def test_negative_in_force_count_raises(self):
        with pytest.raises(ValueError, match="in_force_count"):
            BPAValidator.validate_in_payment(valid_in_payment(in_force_count=-0.1))

    def test_zero_pension_raises(self):
        with pytest.raises(ValueError, match="pension_pa"):
            BPAValidator.validate_in_payment(valid_in_payment(pension_pa=0.0))

    def test_negative_gmp_raises(self):
        with pytest.raises(ValueError, match="gmp_pa"):
            BPAValidator.validate_in_payment(valid_in_payment(gmp_pa=-1.0))

    def test_invalid_sex_raises(self):
        with pytest.raises(ValueError, match="sex"):
            BPAValidator.validate_in_payment(valid_in_payment(sex="X"))

    def test_floor_exceeds_cap_raises(self):
        with pytest.raises(ValueError, match="lpi_floor"):
            BPAValidator.validate_in_payment(
                valid_in_payment(lpi_cap=0.03, lpi_floor=0.05)
            )

    def test_zero_in_force_count_is_valid(self):
        BPAValidator.validate_in_payment(valid_in_payment(in_force_count=0.0))

    def test_blank_deal_id_raises(self):
        with pytest.raises(ValueError, match="deal_id"):
            BPAValidator.validate_in_payment(valid_in_payment(deal_id="  "))

    def test_unknown_deal_id_with_registry_raises(self, registry):
        with pytest.raises(ValueError, match="deal_id"):
            BPAValidator.validate_in_payment(
                valid_in_payment(deal_id="UnknownDeal_2099Q1"),
                registry=registry,
            )

    def test_known_deal_id_with_registry_passes(self, registry):
        BPAValidator.validate_in_payment(valid_in_payment(), registry=registry)

    def test_tranche_id_fully_populated_passes(self):
        df = valid_in_payment()
        df["tranche_id"] = "pre97"
        BPAValidator.validate_in_payment(df)

    def test_tranche_id_partially_populated_raises(self):
        df = pd.concat([valid_in_payment(), valid_in_payment(mp_id="IP002")])
        df = df.reset_index(drop=True)
        df["tranche_id"] = ["pre97", None]
        with pytest.raises(ValueError, match="tranche_id"):
            BPAValidator.validate_in_payment(df)

    def test_tranche_id_absent_passes(self):
        # No tranche_id column at all — valid
        df = valid_in_payment()
        assert "tranche_id" not in df.columns
        BPAValidator.validate_in_payment(df)


# ---------------------------------------------------------------------------
# enhanced
# ---------------------------------------------------------------------------

class TestEnhanced:

    def test_valid_passes(self):
        BPAValidator.validate_enhanced(valid_enhanced())

    def test_missing_rating_years_raises(self):
        df = valid_enhanced().drop(columns=["rating_years"])
        with pytest.raises(ValueError, match="Missing required columns"):
            BPAValidator.validate_enhanced(df)

    def test_negative_rating_years_raises(self):
        with pytest.raises(ValueError, match="rating_years"):
            BPAValidator.validate_enhanced(valid_enhanced(rating_years=-1.0))

    def test_zero_rating_years_valid(self):
        BPAValidator.validate_enhanced(valid_enhanced(rating_years=0.0))

    def test_invalid_sex_raises(self):
        with pytest.raises(ValueError, match="sex"):
            BPAValidator.validate_enhanced(valid_enhanced(sex="U"))

    def test_missing_deal_id_raises(self):
        df = valid_enhanced().drop(columns=["deal_id"])
        with pytest.raises(ValueError, match="Missing required columns"):
            BPAValidator.validate_enhanced(df)

    def test_blank_deal_id_raises(self):
        with pytest.raises(ValueError, match="deal_id"):
            BPAValidator.validate_enhanced(valid_enhanced(deal_id=""))


# ---------------------------------------------------------------------------
# deferred
# ---------------------------------------------------------------------------

class TestDeferred:

    def test_valid_passes(self):
        BPAValidator.validate_deferred(valid_deferred())

    def test_missing_nra_raises(self):
        df = valid_deferred().drop(columns=["nra"])
        with pytest.raises(ValueError, match="Missing required columns"):
            BPAValidator.validate_deferred(df)

    def test_missing_tv_eligible_raises(self):
        df = valid_deferred().drop(columns=["tv_eligible"])
        with pytest.raises(ValueError, match="Missing required columns"):
            BPAValidator.validate_deferred(df)

    def test_nra_le_era_raises(self):
        with pytest.raises(ValueError, match="nra"):
            BPAValidator.validate_deferred(valid_deferred(era=65.0, nra=65.0))

    def test_nra_lt_era_raises(self):
        with pytest.raises(ValueError, match="nra"):
            BPAValidator.validate_deferred(valid_deferred(era=65.0, nra=60.0))

    def test_invalid_revaluation_type_raises(self):
        with pytest.raises(ValueError, match="revaluation_type"):
            BPAValidator.validate_deferred(valid_deferred(revaluation_type="LINKED"))

    def test_valid_revaluation_types(self):
        for rv_type in ("CPI", "RPI", "fixed"):
            BPAValidator.validate_deferred(valid_deferred(revaluation_type=rv_type))

    def test_zero_pension_raises(self):
        with pytest.raises(ValueError, match="deferred_pension_pa"):
            BPAValidator.validate_deferred(valid_deferred(deferred_pension_pa=0.0))

    def test_negative_deferment_years_raises(self):
        with pytest.raises(ValueError, match="deferment_years"):
            BPAValidator.validate_deferred(valid_deferred(deferment_years=-1.0))

    def test_cap_lt_floor_raises(self):
        with pytest.raises(ValueError, match="revaluation_cap"):
            BPAValidator.validate_deferred(
                valid_deferred(revaluation_cap=0.0, revaluation_floor=0.03)
            )

    def test_invalid_sex_raises(self):
        with pytest.raises(ValueError, match="sex"):
            BPAValidator.validate_deferred(valid_deferred(sex="O"))

    def test_tv_eligible_zero_passes(self):
        BPAValidator.validate_deferred(valid_deferred(tv_eligible=0))

    def test_tv_eligible_one_passes(self):
        BPAValidator.validate_deferred(valid_deferred(tv_eligible=1))

    def test_tv_eligible_invalid_value_raises(self):
        with pytest.raises(ValueError, match="tv_eligible"):
            BPAValidator.validate_deferred(valid_deferred(tv_eligible=2))

    def test_tv_eligible_null_raises(self):
        df = valid_deferred()
        df["tv_eligible"] = None
        with pytest.raises(ValueError, match="tv_eligible"):
            BPAValidator.validate_deferred(df)

    def test_missing_deal_id_raises(self):
        df = valid_deferred().drop(columns=["deal_id"])
        with pytest.raises(ValueError, match="Missing required columns"):
            BPAValidator.validate_deferred(df)

    def test_unknown_deal_id_with_registry_raises(self, registry):
        with pytest.raises(ValueError, match="deal_id"):
            BPAValidator.validate_deferred(
                valid_deferred(deal_id="Ghost_2000Q1"),
                registry=registry,
            )

    def test_tranche_id_blank_in_some_rows_raises(self):
        df = pd.concat([valid_deferred(), valid_deferred(mp_id="DEF002")])
        df = df.reset_index(drop=True)
        df["tranche_id"] = ["pre97", ""]
        with pytest.raises(ValueError, match="tranche_id"):
            BPAValidator.validate_deferred(df)


# ---------------------------------------------------------------------------
# dependant
# ---------------------------------------------------------------------------

class TestDependant:

    def test_valid_passes(self):
        BPAValidator.validate_dependant(valid_dependant())

    def test_missing_dependant_age_raises(self):
        df = valid_dependant().drop(columns=["dependant_age"])
        with pytest.raises(ValueError, match="Missing required columns"):
            BPAValidator.validate_dependant(df)

    def test_zero_pension_raises(self):
        with pytest.raises(ValueError, match="pension_pa"):
            BPAValidator.validate_dependant(valid_dependant(pension_pa=0.0))

    def test_invalid_member_sex_raises(self):
        with pytest.raises(ValueError, match="member_sex"):
            BPAValidator.validate_dependant(valid_dependant(member_sex="Z"))

    def test_invalid_dependant_sex_raises(self):
        with pytest.raises(ValueError, match="dependant_sex"):
            BPAValidator.validate_dependant(valid_dependant(dependant_sex="Z"))

    def test_floor_exceeds_cap_raises(self):
        with pytest.raises(ValueError, match="lpi_floor"):
            BPAValidator.validate_dependant(
                valid_dependant(lpi_cap=0.02, lpi_floor=0.03)
            )

    def test_zero_weight_valid(self):
        BPAValidator.validate_dependant(valid_dependant(weight=0.0))

    def test_missing_deal_id_raises(self):
        df = valid_dependant().drop(columns=["deal_id"])
        with pytest.raises(ValueError, match="Missing required columns"):
            BPAValidator.validate_dependant(df)

    def test_known_deal_id_with_registry_passes(self, registry):
        BPAValidator.validate_dependant(valid_dependant(), registry=registry)

    def test_tranche_id_fully_populated_passes(self):
        df = valid_dependant()
        df["tranche_id"] = "GMP"
        BPAValidator.validate_dependant(df)

    def test_tranche_id_absent_passes(self):
        df = valid_dependant()
        assert "tranche_id" not in df.columns
        BPAValidator.validate_dependant(df)


# ---------------------------------------------------------------------------
# BPADealRegistry
# ---------------------------------------------------------------------------

class TestBPADealRegistry:

    def test_get_known_deal(self, registry):
        meta = registry.get("AcmePension_2024Q3")
        assert meta.deal_type == "buyout"
        assert meta.ma_eligible is True
        assert meta.inception_date == date(2024, 9, 30)

    def test_get_unknown_raises(self, registry):
        with pytest.raises(KeyError, match="UnknownDeal"):
            registry.get("UnknownDeal")

    def test_contains(self, registry):
        assert "AcmePension_2024Q3" in registry
        assert "Ghost_2099Q1" not in registry

    def test_all_deal_ids_sorted(self, registry):
        assert registry.all_deal_ids() == ["AcmePension_2024Q3"]

    def test_duplicate_deal_id_raises(self):
        meta = BPADealMetadata(
            deal_id="Dup_2024Q1", deal_type="buyout",
            inception_date=date(2024, 1, 1), deal_name="Dup", ma_eligible=False,
        )
        with pytest.raises(ValueError, match="Duplicate deal_id"):
            BPADealRegistry([meta, meta])

    def test_len(self, registry):
        assert len(registry) == 1
