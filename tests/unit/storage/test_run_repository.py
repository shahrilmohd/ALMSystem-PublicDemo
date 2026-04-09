"""
Unit tests for RunRepository (storage/run_repository.py).

Rules under test
----------------
RunRecord ORM model:
  1.  All fields stored and accessible after insert + commit.
  2.  __repr__ contains run_id, run_type, and status.

RunRepository — save():
  3.  Saved record is retrievable by run_id.
  4.  save() is idempotent — calling twice with same run_id updates the record.
  5.  Multiple distinct run_ids all stored independently.

RunRepository — get():
  6.  Returns the correct RunRecord for a known run_id.
  7.  Raises KeyError for an unknown run_id.

RunRepository — exists():
  8.  Returns True for a saved run_id.
  9.  Returns False for an unknown run_id.

RunRepository — list_all():
  10. Returns all saved records.
  11. Ordered by created_at descending (most recent first).
  12. Returns empty list when no runs exist.

RunRepository — update_status():
  13. status field updated correctly.
  14. started_at populated when supplied.
  15. completed_at populated when supplied.
  16. duration_seconds populated when supplied.
  17. error_message populated when supplied.
  18. Fields not supplied remain unchanged (partial update).
  19. Raises KeyError for an unknown run_id.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from storage.models.run_record import RunRecord
from storage.run_repository import RunRepository


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_record(
    run_id: str = "run_001",
    run_type: str = "DETERMINISTIC",
    status: str = "PENDING",
    created_at: datetime | None = None,
    config_json: str = "{}",
    n_scenarios: int | None = None,
    n_timesteps: int | None = 12,
) -> RunRecord:
    return RunRecord(
        run_id=run_id,
        run_type=run_type,
        status=status,
        created_at=created_at or datetime(2026, 1, 1, 9, 0, 0),
        config_json=config_json,
        n_scenarios=n_scenarios,
        n_timesteps=n_timesteps,
    )


# ---------------------------------------------------------------------------
# RunRecord ORM model
# ---------------------------------------------------------------------------

class TestRunRecord:

    def test_all_fields_stored(self, session):
        record = make_record(run_id="r1", run_type="STOCHASTIC",
                             status="COMPLETED", n_scenarios=100)
        session.add(record)
        session.commit()

        fetched = session.get(RunRecord, "r1")
        assert fetched.run_id       == "r1"
        assert fetched.run_type     == "STOCHASTIC"
        assert fetched.status       == "COMPLETED"
        assert fetched.n_scenarios  == 100
        assert fetched.n_timesteps  == 12
        assert fetched.config_json  == "{}"

    def test_repr_contains_key_fields(self):
        r = make_record(run_id="abc", run_type="LIABILITY_ONLY", status="RUNNING")
        rep = repr(r)
        assert "abc"            in rep
        assert "LIABILITY_ONLY" in rep
        assert "RUNNING"        in rep


# ---------------------------------------------------------------------------
# save()
# ---------------------------------------------------------------------------

class TestSave:

    def test_saved_record_is_retrievable(self, session):
        repo = RunRepository(session)
        repo.save(make_record(run_id="r1"))
        session.commit()
        assert repo.get("r1").run_id == "r1"

    def test_save_is_idempotent(self, session):
        repo = RunRepository(session)
        repo.save(make_record(run_id="r1", status="PENDING"))
        session.commit()

        # Save again with updated status
        repo.save(make_record(run_id="r1", status="COMPLETED"))
        session.commit()

        record = repo.get("r1")
        assert record.status == "COMPLETED"

        # Still only one row
        assert len(repo.list_all()) == 1

    def test_multiple_run_ids_stored_independently(self, session):
        repo = RunRepository(session)
        for i in range(1, 4):
            repo.save(make_record(run_id=f"run_{i:03d}"))
        session.commit()
        assert len(repo.list_all()) == 3


# ---------------------------------------------------------------------------
# get()
# ---------------------------------------------------------------------------

class TestGet:

    def test_returns_correct_record(self, session):
        repo = RunRepository(session)
        repo.save(make_record(run_id="r1", run_type="STOCHASTIC"))
        session.commit()
        assert repo.get("r1").run_type == "STOCHASTIC"

    def test_unknown_run_id_raises_key_error(self, session):
        repo = RunRepository(session)
        with pytest.raises(KeyError):
            repo.get("nonexistent")


# ---------------------------------------------------------------------------
# exists()
# ---------------------------------------------------------------------------

class TestExists:

    def test_returns_true_for_saved_run(self, session):
        repo = RunRepository(session)
        repo.save(make_record(run_id="r1"))
        session.commit()
        assert repo.exists("r1") is True

    def test_returns_false_for_unknown_run(self, session):
        assert RunRepository(session).exists("ghost") is False


# ---------------------------------------------------------------------------
# list_all()
# ---------------------------------------------------------------------------

class TestListAll:

    def test_returns_all_records(self, session):
        repo = RunRepository(session)
        repo.save(make_record(run_id="r1"))
        repo.save(make_record(run_id="r2"))
        session.commit()
        assert len(repo.list_all()) == 2

    def test_ordered_by_created_at_descending(self, session):
        repo = RunRepository(session)
        base = datetime(2026, 1, 1)
        repo.save(make_record(run_id="old", created_at=base))
        repo.save(make_record(run_id="new", created_at=base + timedelta(hours=1)))
        session.commit()

        ids = [r.run_id for r in repo.list_all()]
        assert ids == ["new", "old"]

    def test_empty_when_no_runs(self, session):
        assert RunRepository(session).list_all() == []


# ---------------------------------------------------------------------------
# update_status()
# ---------------------------------------------------------------------------

class TestUpdateStatus:

    @pytest.fixture
    def repo_with_run(self, session):
        repo = RunRepository(session)
        repo.save(make_record(run_id="r1", status="PENDING"))
        session.commit()
        return repo

    def test_status_updated(self, repo_with_run, session):
        repo_with_run.update_status("r1", "RUNNING")
        session.commit()
        assert repo_with_run.get("r1").status == "RUNNING"

    def test_started_at_populated(self, repo_with_run, session):
        t = datetime(2026, 3, 28, 10, 0, 0)
        repo_with_run.update_status("r1", "RUNNING", started_at=t)
        session.commit()
        assert repo_with_run.get("r1").started_at == t

    def test_completed_at_populated(self, repo_with_run, session):
        t = datetime(2026, 3, 28, 10, 5, 0)
        repo_with_run.update_status("r1", "COMPLETED", completed_at=t)
        session.commit()
        assert repo_with_run.get("r1").completed_at == t

    def test_duration_seconds_populated(self, repo_with_run, session):
        repo_with_run.update_status("r1", "COMPLETED", duration_seconds=77.5)
        session.commit()
        assert repo_with_run.get("r1").duration_seconds == pytest.approx(77.5)

    def test_error_message_populated(self, repo_with_run, session):
        repo_with_run.update_status("r1", "FAILED", error_message="division by zero")
        session.commit()
        assert "division by zero" in repo_with_run.get("r1").error_message

    def test_partial_update_leaves_other_fields_unchanged(self, repo_with_run, session):
        # Only update status — n_timesteps and config_json should be unchanged
        repo_with_run.update_status("r1", "RUNNING")
        session.commit()
        record = repo_with_run.get("r1")
        assert record.n_timesteps == 12
        assert record.config_json  == "{}"
        assert record.started_at   is None   # not supplied → stays None

    def test_unknown_run_id_raises_key_error(self, session):
        repo = RunRepository(session)
        with pytest.raises(KeyError):
            repo.update_status("ghost", "RUNNING")
