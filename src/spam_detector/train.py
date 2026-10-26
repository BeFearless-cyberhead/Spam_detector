"""Train, evaluate, and persist the spam classifier.

Usage::

    python -m spam_detector.train          # full dataset -> models/
    python -m spam_detector.train --search # cross-validated model selection
    python -m spam_detector.train --help   # every option

The script writes two artifacts:

* ``spam_model.joblib`` -- the fitted pipeline used by the UI and the API.
* ``metrics.json`` -- the held-out evaluation report plus training metadata
  (dataset fingerprint, library versions, timestamp, calibration scores, and
  the length/vocabulary statistics the drift monitor compares against) for
  reproducibility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    brier_score_loss,
    classification_report,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from spam_detector.config import (
    METRICS_FILENAME,
    MODEL_FILENAME,
    MODELS_DIR,
    dataset_path,
    spam_threshold,
)
from spam_detector.drift import dataset_statistics
from spam_detector.model import build_model

#: Minimum number of unique messages required per class.
MIN_MESSAGES_PER_CLASS = 5

#: Candidate configurations explored by ``--search``. Kept deliberately small:
#: on the SMS corpus each candidate trains in a couple of seconds.
SEARCH_GRID: list[dict[str, Any]] = [
    {"use_char_ngrams": True, "word_ngram_range": (1, 2), "c": 1.0},
    {"use_char_ngrams": True, "word_ngram_range": (1, 2), "c": 0.5},
    {"use_char_ngrams": True, "word_ngram_range": (1, 2), "c": 2.0},
    {"use_char_ngrams": True, "word_ngram_range": (1, 3), "c": 1.0},
    {"use_char_ngrams": False, "word_ngram_range": (1, 2), "c": 1.0},
    {"use_char_ngrams": False, "word_ngram_range": (1, 3), "c": 1.0},
]


def load_dataset(path: Path) -> pd.DataFrame:
    """Read a tab-separated ``label<TAB>message`` dataset.

    The dataset is validated rather than trusted: malformed rows, contradictory
    duplicate labels, and class-imbalanced edge cases all raise instead of
    silently producing a bad model.

    Args:
        path: Location of the dataset file.

    Returns:
        A frame with ``label`` (str), ``message`` (str) and ``target`` (int)
        columns, deduplicated on ``message``.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If the file is empty, malformed, ambiguous, or has fewer
            than :data:`MIN_MESSAGES_PER_CLASS` unique messages per class.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\nDownload it with: python -m spam_detector.data"
        )

    records = []

    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.rstrip("\r\n")

            if not line.strip():
                continue

            label, separator, message = line.partition("\t")

            if not separator or label not in {"ham", "spam"}:
                raise ValueError(f"Invalid dataset record on line {line_number}.")

            message = message.strip()

            if message:
                records.append((label, message))

    if not records:
        raise ValueError("The dataset contains no usable messages.")

    frame = pd.DataFrame(records, columns=["label", "message"])

    # Do not silently choose a label for contradictory records.
    label_counts = frame.groupby("message")["label"].nunique()

    if label_counts.gt(1).any():
        raise ValueError(
            "The dataset contains identical messages with conflicting labels."
        )

    # Prevent identical messages appearing in both train and test sets.
    frame = frame.drop_duplicates(subset=["message"]).copy()
    frame["target"] = frame["label"].map({"ham": 0, "spam": 1})

    counts = frame["target"].value_counts()

    if len(counts) != 2 or counts.min() < MIN_MESSAGES_PER_CLASS:
        raise ValueError(
            f"At least {MIN_MESSAGES_PER_CLASS} unique messages per class are required."
        )

    return frame


def sha256_of(path: Path) -> str:
    """Return the SHA-256 digest of a file, for dataset provenance.

    Args:
        path: File to hash.

    Returns:
        Lowercase hexadecimal digest.
    """
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def select_hyperparameters(
    x_train: pd.Series,
    y_train: pd.Series,
    *,
    seed: int = 42,
    cv: int = 5,
    grid: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Pick pipeline hyperparameters with stratified cross-validation.

    Each candidate is scored by mean spam F1 over the folds -- the metric this
    project optimises for, since missing spam is costlier than a false alarm.

    Args:
        x_train: Training messages.
        y_train: Training targets.
        seed: Seed for the shuffled stratified folds.
        cv: Requested number of folds (reduced automatically for small data).
        grid: Candidate configurations; defaults to :data:`SEARCH_GRID`.

    Returns:
        A JSON-serialisable report with the winning ``params``, its mean/std
        score, and every candidate's result.
    """
    candidates = grid if grid is not None else SEARCH_GRID
    smallest_class = int(pd.Series(list(y_train)).value_counts().min())
    folds = max(2, min(cv, smallest_class))

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    results = []

    for params in candidates:
        scores = cross_val_score(
            build_model(calibrated=False, random_state=seed, **params),
            x_train,
            y_train,
            cv=splitter,
            scoring="f1",
        )
        results.append(
            {
                "params": _serialise_params(params),
                "mean_f1": round(float(np.mean(scores)), 6),
                "std_f1": round(float(np.std(scores)), 6),
                "folds": [round(float(score), 6) for score in scores],
            }
        )

    best = max(results, key=lambda result: result["mean_f1"])
    best_params = next(
        candidate
        for candidate, result in zip(candidates, results, strict=True)
        if result is best
    )

    return {
        "selection_metric": "spam_f1",
        "cv_folds": folds,
        "params": _serialise_params(best_params),
        "mean_f1": best["mean_f1"],
        "std_f1": best["std_f1"],
        "candidates": results,
    }


def _serialise_params(params: dict[str, Any]) -> dict[str, Any]:
    """Convert pipeline parameters into JSON-friendly values."""
    return {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in params.items()
    }


def evaluate_probabilities(
    y_test: pd.Series,
    probabilities: np.ndarray,
    *,
    threshold: float,
) -> dict[str, Any]:
    """Summarise probabilistic and thresholded quality.

    Args:
        y_test: True labels for the held-out split.
        probabilities: Spam probability for every test message.
        threshold: Probability at or above which a message is called spam.

    Returns:
        A JSON-serialisable mapping with the Brier score, log loss, and
        precision/recall/F1 at ``threshold``.
    """
    predictions = (probabilities >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        labels=[1],
        average=None,
        zero_division=0,
    )

    return {
        "threshold": threshold,
        "threshold_metrics": {
            "precision": round(float(precision[0]), 6),
            "recall": round(float(recall[0]), 6),
            "f1": round(float(f1[0]), 6),
        },
        "brier_score": round(
            float(brier_score_loss(y_test, probabilities, pos_label=1)),
            6,
        ),
        "log_loss": round(float(log_loss(y_test, probabilities, labels=[0, 1])), 6),
    }


def build_metrics_report(
    y_test: pd.Series,
    predictions,
    *,
    data_path: Path,
    row_count: int,
    train_messages: int,
    probabilities: np.ndarray | None = None,
    threshold: float | None = None,
    calibrated: bool = False,
    search: dict[str, Any] | None = None,
    training_statistics: dict[str, Any] | None = None,
) -> dict:
    """Assemble the evaluation report saved next to the model.

    Args:
        y_test: True labels for the held-out split.
        predictions: Predicted labels for the held-out split.
        data_path: Dataset that produced the model.
        row_count: Number of usable rows in the dataset.
        train_messages: Number of rows used for fitting.
        probabilities: Optional spam probabilities for the held-out split.
        threshold: Decision threshold used for ``predictions``.
        calibrated: Whether the saved pipeline is probability-calibrated.
        search: Optional hyperparameter-search report.
        training_statistics: Optional length/vocabulary stats for drift checks.

    Returns:
        A JSON-serialisable mapping of metrics and training metadata.
    """
    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["ham", "spam"],
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])

    metrics: dict[str, Any] = {
        "training_messages": train_messages,
        "test_messages": len(y_test),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_label_order": ["ham", "spam"],
        "confusion_matrix_axes": {"rows": "actual", "columns": "predicted"},
        "training": {
            "dataset": data_path.name,
            "dataset_rows": int(row_count),
            "dataset_sha256": sha256_of(data_path),
            "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "python_version": platform.python_version(),
            "scikit_learn_version": sklearn.__version__,
            "calibrated": calibrated,
            "calibration_method": "sigmoid" if calibrated else None,
        },
    }

    if probabilities is not None and threshold is not None:
        metrics["probability"] = evaluate_probabilities(
            y_test,
            probabilities,
            threshold=threshold,
        )

    if search is not None:
        metrics["training"]["hyperparameter_search"] = search

    if training_statistics is not None:
        metrics["training_statistics"] = training_statistics

    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train the TF-IDF + logistic-regression spam classifier.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=dataset_path(),
        help="Tab-separated dataset (default: data/SMSSpamCollection).",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=MODELS_DIR,
        help="Where the model and metrics are written (default: models/).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of the data held out for evaluation (default: 0.2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the train/test split (default: 42).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=spam_threshold(),
        help="Spam probability threshold for the reported metrics (default: 0.5).",
    )
    parser.add_argument(
        "--search",
        action="store_true",
        help="Select hyperparameters with stratified cross-validation.",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=5,
        help="Cross-validation folds used by --search (default: 5).",
    )
    parser.add_argument(
        "--no-calibrate",
        action="store_true",
        help="Save the uncalibrated logistic regression instead of the "
        "calibrated classifier.",
    )
    parser.add_argument(
        "--no-char-ngrams",
        action="store_true",
        help="Disable character n-grams (overrides the search-selected value).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Train the model, evaluate it, and write the artifacts.

    Args:
        argv: Optional argument list, forwarded to :func:`parse_args`.
    """
    args = parse_args(argv)

    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0.0 and 1.0.")

    frame = load_dataset(args.data)

    x_train, x_test, y_train, y_test = train_test_split(
        frame["message"],
        frame["target"],
        test_size=args.test_size,
        random_state=args.seed,
        stratify=frame["target"],
    )

    search = None
    model_params: dict[str, Any] = {}

    if args.search:
        search = select_hyperparameters(
            x_train,
            y_train,
            seed=args.seed,
            cv=args.cv,
        )
        model_params = {
            "use_char_ngrams": search["params"]["use_char_ngrams"],
            "word_ngram_range": tuple(search["params"]["word_ngram_range"]),
            "c": search["params"]["c"],
        }

    if args.no_char_ngrams:
        model_params["use_char_ngrams"] = False

    model = build_model(
        calibrated=not args.no_calibrate,
        calibration_cv=max(2, min(5, int(y_train.value_counts().min()))),
        random_state=args.seed,
        **model_params,
    )
    model.fit(x_train, y_train)

    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= args.threshold).astype(int)

    metrics = build_metrics_report(
        y_test,
        predictions,
        data_path=args.data,
        row_count=len(frame),
        train_messages=len(x_train),
        probabilities=probabilities,
        threshold=args.threshold,
        calibrated=not args.no_calibrate,
        search=search,
        training_statistics=dataset_statistics(frame["message"]),
    )

    model_path = args.model_dir / MODEL_FILENAME
    metrics_path = args.model_dir / METRICS_FILENAME

    args.model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Training messages: {len(x_train)}")
    print(f"Test messages: {len(x_test)}")
    print(f"Decision threshold: {args.threshold:.2f}")
    print(f"Calibrated probabilities: {not args.no_calibrate}")

    if search:
        chosen = search["params"]
        print(
            f"Selected parameters by {search['cv_folds']}-fold CV "
            f"(mean spam F1 {search['mean_f1']:.4f}): {chosen}"
        )

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["ham", "spam"],
            zero_division=0,
        )
    )
    print("Confusion matrix: rows=actual, columns=predicted")
    print("Label order: ham, spam")
    print(metrics["confusion_matrix"])

    if "probability" in metrics:
        probability = metrics["probability"]
        print(f"\nBrier score: {probability['brier_score']:.4f}")
        print(f"Log loss: {probability['log_loss']:.4f}")

    print(f"\nModel saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as exc:
        sys.exit(f"Training failed: {exc}")
