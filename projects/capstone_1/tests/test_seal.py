import inspect
import json

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


def _write_seal(tmp_path, monkeypatch, **overrides):
    seal_path = tmp_path / "holdout_seal.json"
    payload = {
        "holdout_start": "2026-05",
        "holdout_sha256": seal.holdout_hash(["c", "d"]),
        "holdout_rows": 2,
        "development_rows": 2,
        "opened": False,
        **overrides,
    }
    seal_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(seal, "SEAL_PATH", seal_path)
    return seal_path


def _write_clean_parquet(tmp_path, monkeypatch, frame=None):
    parquet = tmp_path / "flights_clean.parquet"
    (frame if frame is not None else _frame()).to_parquet(parquet, index=False)
    monkeypatch.setattr(seal, "CLEAN_PARQUET", parquet)
    return parquet


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


def test_holdout_frame_refuses_without_explicit_unseal(tmp_path, monkeypatch):
    _write_seal(tmp_path, monkeypatch)
    _write_clean_parquet(tmp_path, monkeypatch)

    with pytest.raises(seal.SealViolation, match="sealed"):
        seal.holdout_frame()


def test_holdout_frame_refuses_when_already_opened(tmp_path, monkeypatch):
    _write_seal(tmp_path, monkeypatch, opened=True, opened_utc="2026-05-01T00:00:00+00:00")
    _write_clean_parquet(tmp_path, monkeypatch)

    with pytest.raises(seal.SealViolation, match="already opened"):
        seal.holdout_frame(unseal=True)


def test_holdout_frame_unseals_once_and_records_the_opening(tmp_path, monkeypatch):
    seal_path = _write_seal(tmp_path, monkeypatch)
    _write_clean_parquet(tmp_path, monkeypatch)

    out = seal.holdout_frame(unseal=True)

    assert set(out.period) == set(config.HOLDOUT_MONTHS)
    assert len(out) == 2

    recorded = json.loads(seal_path.read_text(encoding="utf-8"))
    assert recorded["opened"] is True
    assert "opened_utc" in recorded


def test_verify_raises_on_tampered_row_key(tmp_path, monkeypatch):
    _write_seal(tmp_path, monkeypatch)
    tampered = _frame()
    tampered.loc[tampered.row_key == "d", "row_key"] = "tampered"
    _write_clean_parquet(tmp_path, monkeypatch, frame=tampered)

    with pytest.raises(seal.SealViolation, match="Holdout hash mismatch") as excinfo:
        seal.verify()

    message = str(excinfo.value)
    assert seal.holdout_hash(["c", "d"]) in message
    assert seal.holdout_hash(["c", "tampered"]) in message


def test_verify_has_no_force_parameter():
    assert "force" not in inspect.signature(seal.verify).parameters
