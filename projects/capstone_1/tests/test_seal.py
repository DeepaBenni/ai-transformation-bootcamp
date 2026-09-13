import pandas as pd
import pytest

from app import config
from app.data import seal


def _frame():
    return pd.DataFrame(
        {
            "row_key": ["a", "b", "c", "d"],
            "period": ["2026-04", "2026-04", "2026-05", "2026-06"],
            "label": [0, 1, 1, 0],
        }
    )


def test_hash_is_stable_regardless_of_input_order():
    assert seal.holdout_hash(["b", "a"]) == seal.holdout_hash(["a", "b"])


def test_hash_changes_when_a_key_changes():
    assert seal.holdout_hash(["a", "b"]) != seal.holdout_hash(["a", "c"])


def test_development_frame_never_returns_a_holdout_month(tmp_path, monkeypatch):
    parquet = tmp_path / "flights_clean.parquet"
    _frame().to_parquet(parquet, index=False)
    monkeypatch.setattr(config, "CLEAN_PARQUET", parquet)
    monkeypatch.setattr(seal, "CLEAN_PARQUET", parquet)

    out = seal.development_frame()

    assert set(out.period) == {"2026-04"}
    assert not set(out.period) & set(config.HOLDOUT_MONTHS)


def test_development_frame_drops_every_sealed_row(tmp_path, monkeypatch):
    parquet = tmp_path / "flights_clean.parquet"
    _frame().to_parquet(parquet, index=False)
    monkeypatch.setattr(seal, "CLEAN_PARQUET", parquet)

    assert len(seal.development_frame()) == 2


def test_missing_parquet_names_the_command_that_builds_it(tmp_path, monkeypatch):
    monkeypatch.setattr(seal, "CLEAN_PARQUET", tmp_path / "absent.parquet")
    with pytest.raises(FileNotFoundError, match="app.cli prepare"):
        seal.development_frame()
