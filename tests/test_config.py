"""Tests for environment-driven configuration."""

from pathlib import Path

import pytest

from spam_detector import config


def test_model_path_defaults_to_the_repository(monkeypatch):
    monkeypatch.delenv("SPAM_MODEL_PATH", raising=False)

    assert config.model_path() == config.MODELS_DIR / config.MODEL_FILENAME


def test_model_path_honours_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("SPAM_MODEL_PATH", str(tmp_path / "custom.joblib"))

    assert config.model_path() == tmp_path / "custom.joblib"
    assert config.metrics_path() == tmp_path / config.METRICS_FILENAME


def test_feedback_path_defaults_and_override(monkeypatch, tmp_path):
    monkeypatch.delenv("SPAM_FEEDBACK_LOG", raising=False)
    assert config.feedback_path() == config.DEFAULT_FEEDBACK_PATH

    monkeypatch.setenv("SPAM_FEEDBACK_LOG", str(tmp_path / "fb.jsonl"))
    assert config.feedback_path() == tmp_path / "fb.jsonl"


def test_threshold_defaults_to_half(monkeypatch):
    monkeypatch.delenv("SPAM_THRESHOLD", raising=False)

    assert config.spam_threshold() == 0.5


@pytest.mark.parametrize("raw", ["", "   "])
def test_threshold_ignores_blank_values(monkeypatch, raw):
    monkeypatch.setenv("SPAM_THRESHOLD", raw)

    assert config.spam_threshold() == 0.5


def test_threshold_reads_the_environment(monkeypatch):
    monkeypatch.setenv("SPAM_THRESHOLD", "0.35")

    assert config.spam_threshold() == 0.35


@pytest.mark.parametrize("raw", ["abc", "1.5", "-0.2"])
def test_threshold_rejects_invalid_values(monkeypatch, raw):
    monkeypatch.setenv("SPAM_THRESHOLD", raw)

    with pytest.raises(ValueError, match="SPAM_THRESHOLD"):
        config.spam_threshold()


def test_drift_window_defaults_and_override(monkeypatch):
    monkeypatch.delenv("SPAM_DRIFT_WINDOW", raising=False)
    assert config.drift_window() == 200

    monkeypatch.setenv("SPAM_DRIFT_WINDOW", "50")
    assert config.drift_window() == 50


@pytest.mark.parametrize("raw", ["abc", "0", "-5"])
def test_drift_window_rejects_invalid_values(monkeypatch, raw):
    monkeypatch.setenv("SPAM_DRIFT_WINDOW", raw)

    with pytest.raises(ValueError, match="SPAM_DRIFT_WINDOW"):
        config.drift_window()


def test_dataset_paths_honour_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("SPAM_DATA_DIR", str(tmp_path / "raw"))
    monkeypatch.setenv("SPAM_DATASET_PATH", str(tmp_path / "data.tsv"))

    assert config.data_dir() == Path(tmp_path / "raw")
    assert config.dataset_path() == tmp_path / "data.tsv"
