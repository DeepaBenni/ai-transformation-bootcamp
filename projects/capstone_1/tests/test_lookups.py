import pandas as pd
import pytest

from app.serving import lookups


@pytest.fixture(autouse=True)
def _reset_cache():
    lookups.reset_cache()
    yield
    lookups.reset_cache()


def _features_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "period": ["2026-04", "2026-04", "2026-04"],
            "Origin": ["ATL", "ATL", "ORD"],
            "Dest": ["ORD", "ORD", "ATL"],
            "Reporting_Airline": ["DL", "DL", "AA"],
            "route": ["ATL-ORD", "ATL-ORD", "ORD-ATL"],
            "dep_hour": [17, 17, 9],
            "origin_hour_late_rate": [0.3, 0.3, 0.1],
            "route_late_rate": [0.25, 0.25, 0.15],
            "carrier_late_rate": [0.22, 0.22, 0.18],
            "origin_late_rate": [0.2, 0.2, 0.12],
            "dest_late_rate": [0.19, 0.19, 0.21],
            "carrier_origin_late_rate": [0.24, 0.24, 0.17],
            "route_late_rate_n": [500, 500, 40],
        }
    )


def test_build_lookups_groups_by_route(monkeypatch, tmp_path):
    parquet = tmp_path / "features.parquet"
    _features_frame().to_parquet(parquet)
    monkeypatch.setattr(lookups, "FEATURES_PARQUET", parquet)

    out = lookups.build_lookups()

    assert out["route"]["ATL-ORD"] == pytest.approx(0.25)
    assert out["route_n"]["ORD-ATL"] == pytest.approx(40)


def test_load_lookups_returns_empty_when_no_parquet(monkeypatch, tmp_path):
    monkeypatch.setattr(lookups, "FEATURES_PARQUET", tmp_path / "missing.parquet")
    monkeypatch.setattr(lookups, "LOOKUP_PATH", tmp_path / "lookup_tables.pkl")
    assert lookups.load_lookups() == {}


def test_load_lookups_caches_across_calls(monkeypatch, tmp_path):
    parquet = tmp_path / "features.parquet"
    _features_frame().to_parquet(parquet)
    monkeypatch.setattr(lookups, "FEATURES_PARQUET", parquet)
    monkeypatch.setattr(lookups, "LOOKUP_PATH", tmp_path / "lookup_tables.pkl")

    first = lookups.load_lookups()
    parquet.unlink()  # prove the second call doesn't re-read the parquet
    second = lookups.load_lookups()
    assert first is second
