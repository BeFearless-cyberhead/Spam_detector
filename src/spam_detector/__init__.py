"""SMS spam detection: pipeline, training, Streamlit UI, and FastAPI service.

The package is installed in editable mode (``pip install -e ".[all]"``) so the
UI, API, and prediction CLI can be launched from anywhere on the machine:

    spam-detector-train      # retrain from the CLI
    spam-detector-predict    # classify one message
    spam-detector-api        # serve the REST API
    spam-detector-ui         # launch the Streamlit demo
"""

from __future__ import annotations

from spam_detector.model import (
    DEFAULT_THRESHOLD,
    LABELS,
    MAX_MESSAGE_LENGTH,
    build_model,
    explain_prediction,
    load_model,
    predict_spam,
    predict_spam_probability,
    validate_message,
)

__version__ = "2.0.0"

__all__ = [
    "DEFAULT_THRESHOLD",
    "LABELS",
    "MAX_MESSAGE_LENGTH",
    "__version__",
    "build_model",
    "explain_prediction",
    "load_model",
    "predict_spam",
    "predict_spam_probability",
    "validate_message",
]
