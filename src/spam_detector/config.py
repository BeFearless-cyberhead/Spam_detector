"""Filesystem layout and environment-driven configuration.

All paths default to the repository layout and can be overridden with
environment variables, which is what the container images rely on:

* ``SPAM_MODEL_PATH`` -- joblib artifact served by the API and the UI.
* ``SPAM_DATASET_PATH`` -- tab-separated dataset read by the trainer.
* ``SPAM_DATA_DIR`` -- directory used by the dataset downloader.
* ``SPAM_FEEDBACK_LOG`` -- JSONL file that receives ``/feedback`` reports.
* ``SPAM_THRESHOLD`` -- default spam probability threshold (``0.5``).
* ``SPAM_DRIFT_WINDOW`` -- number of recent messages used for drift checks.

Note:
    Paths are derived from this file's location, so an *editable* install
    (``pip install -e .``) resolves them to the checkout. For a regular install
    set the environment variables instead.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Repository root: ``<repo>/src/spam_detector/config.py`` → parents[2].
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

#: File names used for the dataset and the trained artifacts.
DATASET_FILENAME = "SMSSpamCollection"
MODEL_FILENAME = "spam_model.joblib"
METRICS_FILENAME = "metrics.json"
FEEDBACK_FILENAME = "feedback.jsonl"

#: UCI SMS Spam Collection archive (CC BY 4.0).
DATASET_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"

DEFAULT_DATASET_PATH = DATA_DIR / DATASET_FILENAME
DEFAULT_MODEL_PATH = MODELS_DIR / MODEL_FILENAME
DEFAULT_FEEDBACK_PATH = DATA_DIR / FEEDBACK_FILENAME

#: Default decision threshold and drift window size.
DEFAULT_THRESHOLD = 0.5
DEFAULT_DRIFT_WINDOW = 200


def dataset_path() -> Path:
    """Return the dataset location to read or write.

    Returns:
        ``SPAM_DATASET_PATH`` when set, otherwise ``data/SMSSpamCollection``.
    """
    return Path(os.getenv("SPAM_DATASET_PATH", DEFAULT_DATASET_PATH))


def data_dir() -> Path:
    """Return the directory the downloader writes into.

    Returns:
        ``SPAM_DATA_DIR`` when set, otherwise ``data/``.
    """
    return Path(os.getenv("SPAM_DATA_DIR", DATA_DIR))


def model_path() -> Path:
    """Return the model artifact location served by the API and the UI.

    Returns:
        ``SPAM_MODEL_PATH`` when set, otherwise ``models/spam_model.joblib``.
    """
    return Path(os.getenv("SPAM_MODEL_PATH", DEFAULT_MODEL_PATH))


def metrics_path() -> Path:
    """Return the evaluation-metrics file that belongs to :func:`model_path`.

    Metrics always live next to the model they describe.

    Returns:
        The sibling ``metrics.json`` of :func:`model_path`.
    """
    return model_path().parent / METRICS_FILENAME


def feedback_path() -> Path:
    """Return the JSONL file that receives misclassification reports.

    Returns:
        ``SPAM_FEEDBACK_LOG`` when set, otherwise ``data/feedback.jsonl``.
    """
    return Path(os.getenv("SPAM_FEEDBACK_LOG", DEFAULT_FEEDBACK_PATH))


def spam_threshold() -> float:
    """Return the default probability threshold used at inference time.

    Returns:
        ``SPAM_THRESHOLD`` parsed as a float, otherwise ``0.5``.

    Raises:
        ValueError: If the variable is not a number in ``[0.0, 1.0]``.
    """
    raw = os.getenv("SPAM_THRESHOLD")

    if raw is None or not raw.strip():
        return DEFAULT_THRESHOLD

    try:
        threshold = float(raw)
    except ValueError as exc:
        raise ValueError(f"SPAM_THRESHOLD must be a number, got {raw!r}.") from exc

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("SPAM_THRESHOLD must be between 0.0 and 1.0.")

    return threshold


def drift_window() -> int:
    """Return how many recent messages the drift monitor keeps in its window.

    Returns:
        ``SPAM_DRIFT_WINDOW`` parsed as a positive int, otherwise ``200``.

    Raises:
        ValueError: If the variable is not a positive integer.
    """
    raw = os.getenv("SPAM_DRIFT_WINDOW")

    if raw is None or not raw.strip():
        return DEFAULT_DRIFT_WINDOW

    try:
        window = int(raw)
    except ValueError as exc:
        raise ValueError(f"SPAM_DRIFT_WINDOW must be an integer, got {raw!r}.") from exc

    if window < 1:
        raise ValueError("SPAM_DRIFT_WINDOW must be at least 1.")

    return window
