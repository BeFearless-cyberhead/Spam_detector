"""Core spam-classification logic shared by the CLI, the UI, and the API.

The model is a scikit-learn :class:`~sklearn.pipeline.Pipeline` that turns a raw
message into TF-IDF features (unigrams + bigrams) and scores it with a
regularised logistic-regression classifier. Keeping the pipeline definition in
one module means the Streamlit UI, the FastAPI service, and the training script
can never drift apart.
"""

from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

MODEL_FILENAME = "spam_model.joblib"
METRICS_FILENAME = "metrics.json"

#: Longest message accepted for scoring. Guards the API against oversized input.
MAX_MESSAGE_LENGTH = 10_000

#: Label mapping used everywhere a human-readable label is required.
LABELS = {0: "ham", 1: "spam"}


def validate_message(message: str) -> str:
    """Validate and normalise a raw message before scoring.

    Args:
        message: The raw user-supplied message.

    Returns:
        The message with surrounding whitespace removed.

    Raises:
        TypeError: If ``message`` is not a string.
        ValueError: If the message is empty or longer than
            :data:`MAX_MESSAGE_LENGTH` characters.
    """
    if not isinstance(message, str):
        raise TypeError("Message must be a string.")

    cleaned = message.strip()

    if not cleaned:
        raise ValueError("Please enter a non-empty message.")

    if len(cleaned) > MAX_MESSAGE_LENGTH:
        raise ValueError(f"Message must not exceed {MAX_MESSAGE_LENGTH} characters.")

    return cleaned


def build_model() -> Pipeline:
    """Create an unfitted TF-IDF + logistic-regression pipeline.

    Returns:
        A fresh, untrained :class:`~sklearn.pipeline.Pipeline`.
    """
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    max_features=30_000,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )


def predict_spam(model: Pipeline, message: str) -> bool:
    """Predict whether a message is spam.

    Args:
        model: A fitted pipeline produced by :func:`build_model`.
        message: The raw message to classify.

    Returns:
        ``True`` when the message is predicted spam, ``False`` otherwise.

    Raises:
        TypeError: If ``message`` is not a string.
        ValueError: If the message is empty or too long.
    """
    cleaned = validate_message(message)
    prediction = model.predict([cleaned])[0]
    return bool(prediction == 1)


def predict_spam_probability(model: Pipeline, message: str) -> float:
    """Return the model's estimated probability that a message is spam.

    Args:
        model: A fitted pipeline produced by :func:`build_model`.
        message: The raw message to classify.

    Returns:
        A float in ``[0.0, 1.0]``; higher values mean "more likely spam".

    Raises:
        TypeError: If ``message`` is not a string.
        ValueError: If the message is empty or too long.
    """
    cleaned = validate_message(message)
    probabilities = model.predict_proba([cleaned])[0]
    return float(probabilities[1])


def load_model(path: str | Path) -> Pipeline:
    """Load a persisted pipeline and check that it has the expected shape.

    Args:
        path: Location of a ``joblib`` artifact written by ``train.py``.

    Returns:
        The deserialised :class:`~sklearn.pipeline.Pipeline`.

    Raises:
        FileNotFoundError: If no artifact exists at ``path``.
        TypeError: If the artifact is not a scikit-learn pipeline.

    Note:
        Only load artifacts you created or explicitly trust; ``joblib`` uses
        pickle semantics under the hood.
    """
    model_path = Path(path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = joblib.load(model_path)

    if not isinstance(model, Pipeline):
        raise TypeError("The saved model is not a supported pipeline.")

    return model
