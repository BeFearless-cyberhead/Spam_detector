"""Core spam-classification logic shared by the CLI, the UI, and the API.

The model is a scikit-learn :class:`~sklearn.pipeline.Pipeline` that turns a raw
message into TF-IDF features and scores it with a regularised logistic-regression
classifier. Keeping the pipeline definition in one module means the Streamlit UI,
the FastAPI service, and the training script can never drift apart.

Features are combined with a :class:`~sklearn.pipeline.FeatureUnion`:

* **word n-grams** -- unigrams and bigrams, the bulk of the signal;
* **character n-grams** (``char_wb``, 3-5) -- robust to the abbreviations,
  misspellings, and obfuscated tokens (``fr33``, ``ca$h``) common in spam.

By default the classifier is wrapped in
:class:`~sklearn.calibration.CalibratedClassifierCV` so the reported confidence
is a calibrated probability rather than an uncalibrated decision-function
score. Pass ``calibrated=False`` to train the bare logistic regression (the
trainer uses this while searching hyperparameters).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

#: Longest message accepted for scoring. Guards the API against oversized input.
MAX_MESSAGE_LENGTH = 10_000

#: Label mapping used everywhere a human-readable label is required.
LABELS = {0: "ham", 1: "spam"}

#: Probability at or above which a message is labelled spam.
DEFAULT_THRESHOLD = 0.5

#: Number of features shown by :func:`explain_prediction` by default.
DEFAULT_TOP_TOKENS = 5


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


def build_model(
    *,
    calibrated: bool = True,
    use_char_ngrams: bool = True,
    word_ngram_range: tuple[int, int] = (1, 2),
    char_ngram_range: tuple[int, int] = (3, 5),
    word_max_features: int = 30_000,
    char_max_features: int = 20_000,
    c: float = 1.0,
    class_weight: str | None = "balanced",
    calibration_cv: int = 5,
    random_state: int = 42,
) -> Pipeline:
    """Create an unfitted TF-IDF + logistic-regression pipeline.

    Args:
        calibrated: Wrap the classifier in ``CalibratedClassifierCV`` so
            ``predict_proba`` returns calibrated probabilities.
        use_char_ngrams: Add a character n-gram TF-IDF block next to the word
            block.
        word_ngram_range: N-gram range for the word vectoriser.
        char_ngram_range: N-gram range for the character vectoriser.
        word_max_features: Vocabulary cap for the word vectoriser.
        char_max_features: Vocabulary cap for the character vectoriser.
        c: Inverse regularisation strength for the logistic regression.
        class_weight: ``"balanced"`` (default) or ``None``.
        calibration_cv: Folds used by ``CalibratedClassifierCV``.
        random_state: Seed for the classifier's solver.

    Returns:
        A fresh, untrained :class:`~sklearn.pipeline.Pipeline`.
    """
    transformers: list[tuple[str, Any]] = [
        (
            "word",
            TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=word_ngram_range,
                sublinear_tf=True,
                max_features=word_max_features,
            ),
        )
    ]

    if use_char_ngrams:
        transformers.append(
            (
                "char",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    analyzer="char_wb",
                    ngram_range=char_ngram_range,
                    sublinear_tf=True,
                    max_features=char_max_features,
                ),
            )
        )

    classifier: Any = LogisticRegression(
        class_weight=class_weight,
        C=c,
        max_iter=1000,
        random_state=random_state,
    )

    if calibrated:
        classifier = CalibratedClassifierCV(
            classifier,
            method="sigmoid",
            cv=calibration_cv,
        )

    return Pipeline(
        steps=[
            ("tfidf", FeatureUnion(transformers)),
            ("classifier", classifier),
        ]
    )


def predict_spam(
    model: Pipeline,
    message: str,
    threshold: float = DEFAULT_THRESHOLD,
) -> bool:
    """Predict whether a message is spam.

    Args:
        model: A fitted pipeline produced by :func:`build_model`.
        message: The raw message to classify.
        threshold: Spam probability at or above which the message is flagged.
            Lower it to catch more spam at the cost of more false positives.

    Returns:
        ``True`` when the message is predicted spam, ``False`` otherwise.

    Raises:
        TypeError: If ``message`` is not a string.
        ValueError: If the message is empty, too long, or ``threshold`` is
            outside ``[0.0, 1.0]``.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be between 0.0 and 1.0.")

    return predict_spam_probability(model, message) >= threshold


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


def _coefficients(model: Pipeline) -> np.ndarray:
    """Return the logistic-regression weights as a single vector.

    A calibrated pipeline holds one logistic regression per fold; their
    coefficients are averaged so the explanation reflects the consensus.
    """
    classifier = model.named_steps["classifier"]

    if isinstance(classifier, CalibratedClassifierCV):
        coefficients = [
            calibrated.estimator.coef_[0]
            for calibrated in classifier.calibrated_classifiers_
        ]
        return np.mean(coefficients, axis=0)

    return classifier.coef_[0]


def explain_prediction(
    model: Pipeline,
    message: str,
    top_k: int = DEFAULT_TOP_TOKENS,
) -> list[dict[str, Any]]:
    """Explain a prediction with the tokens that pushed it hardest.

    Contributions are the TF-IDF value of a feature multiplied by its logistic
    regression coefficient. Positive weights push towards spam, negative ones
    towards ham. Character n-grams are reported with ``"source": "char"``.

    Args:
        model: A fitted pipeline produced by :func:`build_model`.
        message: The raw message to explain.
        top_k: How many tokens to return for each direction.

    Returns:
        Up to ``2 * top_k`` mappings of ``token``, ``weight``, ``source`` and
        ``direction`` (``"spam"`` or ``"ham"``), strongest first.

    Raises:
        TypeError: If ``message`` is not a string.
        ValueError: If the message is empty, too long, or ``top_k`` is not
            positive.
    """
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    cleaned = validate_message(message)
    features = model.named_steps["tfidf"]
    coefficients = _coefficients(model)

    row = features.transform([cleaned])
    contributions = row.multiply(coefficients).toarray().ravel()
    indices = np.flatnonzero(contributions)
    names = features.get_feature_names_out()

    ranked = sorted(indices, key=lambda index: contributions[index], reverse=True)

    def make(index: int, direction: str) -> dict[str, Any]:
        name = str(names[index])
        source, separator, token = name.partition("__")

        if not separator:
            source, token = "word", name

        return {
            "token": token,
            "weight": round(float(contributions[index]), 6),
            "source": source or "word",
            "direction": direction,
        }

    spam_side = [make(index, "spam") for index in ranked[:top_k]]
    ham_side = [
        make(index, "ham") for index in ranked[::-1][:top_k] if contributions[index] < 0
    ]

    return spam_side + ham_side


def load_model(path: str | Path) -> Pipeline:
    """Load a persisted pipeline and check that it has the expected shape.

    Args:
        path: Location of a ``joblib`` artifact written by
            :mod:`spam_detector.train`.

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
