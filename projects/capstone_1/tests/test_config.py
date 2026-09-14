from pathlib import Path

from app import config


def test_holdout_months_are_the_sealed_pair():
    assert config.HOLDOUT_MONTHS == ("2026-05", "2026-06")


def test_development_months_exclude_the_holdout():
    assert set(config.DEV_MONTHS).isdisjoint(config.HOLDOUT_MONTHS)
    assert len(config.DEV_MONTHS) == 10


def test_fold_validation_months_are_the_last_five_development_months():
    assert config.FOLD_VALIDATION_MONTHS == (
        "2025-12",
        "2026-01",
        "2026-02",
        "2026-03",
        "2026-04",
    )


def test_prior_and_alpha_are_the_documented_constants():
    assert config.PRIOR == 0.2242
    assert config.ALPHA == 20


def test_data_dir_honours_the_environment_variable(monkeypatch, tmp_path):
    monkeypatch.setenv("SIGNALCRAFT_DATA_DIR", str(tmp_path))
    reloaded = config.resolve_data_dir()
    assert reloaded == tmp_path


def test_data_dir_falls_back_when_env_var_absent(monkeypatch):
    monkeypatch.delenv("SIGNALCRAFT_DATA_DIR", raising=False)
    assert isinstance(config.resolve_data_dir(), Path)
