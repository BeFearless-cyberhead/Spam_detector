"""SMS spam detection: pipeline, training, Streamlit UI, and FastAPI service.

The package is installed in editable mode (``pip install -e ".[all]"``) so the
UI and API can be launched from anywhere on the machine:

    spam-detector-train     # retrain from the CLI
    spam-detector-api       # serve the REST API
    spam-detector-ui        # launch the Streamlit demo
"""

from __future__ import annotations

from spam_detector.model import (
    LABELS,
    MAX_MESSAGE_LENGTH,
    build_model,
    load_model,
    predict_spam,
    predict_spam_probability,
    validate_message,
)

__version__ = "1.0.0"

__all__ = [
    "LABELS",
    "MAX_MESSAGE_LENGTH",
    "__version__",
    "build_model",
    "load_model",
    "predict_spam",
    "predict_spam_probability",
    "validate_message",
]
