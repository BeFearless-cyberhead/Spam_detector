"""Shared fixtures for the test suite."""

from __future__ import annotations

import joblib
import pytest

from spam_detector.model import build_model

#: Tiny, obviously separable corpus so a fitted model is available in
#: milliseconds without touching the real dataset.
TRAINING_MESSAGES = [
    "Team meeting tomorrow morning",
    "Please call me after class",
    "Dinner is ready at home",
    "Can you send the homework notes",
    "Win free cash prize claim now",
    "Congratulations claim your free prize",
    "Urgent winner cash reward claim now",
    "Free bonus cash prize click now",
]

TRAINING_LABELS = [0, 0, 0, 0, 1, 1, 1, 1]


@pytest.fixture
def fitted_model():
    """A pipeline fitted on the tiny in-memory corpus.

    Uncalibrated on purpose: the corpus has only four messages per class, which
    is too small for cross-validated calibration. The dedicated
    ``calibrated_model`` fixture covers that path.
    """
    model = build_model(calibrated=False)
    model.fit(TRAINING_MESSAGES, TRAINING_LABELS)
    return model


@pytest.fixture
def calibrated_model():
    """A probability-calibrated pipeline fitted on the tiny corpus."""
    model = build_model(calibration_cv=2)
    model.fit(TRAINING_MESSAGES, TRAINING_LABELS)
    return model


@pytest.fixture
def model_path(tmp_path, fitted_model):
    """The tiny fitted model persisted to a temporary directory."""
    path = tmp_path / "spam_model.joblib"
    joblib.dump(fitted_model, path)
    return path
