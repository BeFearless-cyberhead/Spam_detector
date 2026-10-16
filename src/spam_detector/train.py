"""Train, evaluate, and persist the spam classifier.

Usage::

    python -m spam_detector.train          # full dataset -> models/
    python -m spam_detector.train --help   # every option

The script writes two artifacts:

* ``spam_model.joblib`` -- the fitted pipeline used by the UI and the API.
* ``metrics.json`` -- the held-out evaluation report plus training metadata
  (dataset fingerprint, library versions, timestamp) for reproducibility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from spam_detector.config import (
    METRICS_FILENAME,
    MODEL_FILENAME,
    MODELS_DIR,
    dataset_path,
)
from spam_detector.model import build_model

#: Minimum number of unique messages required per class.
MIN_MESSAGES_PER_CLASS = 5


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


def build_metrics_report(
    y_test: pd.Series,
    predictions,
    *,
    data_path: Path,
    row_count: int,
    train_messages: int,
) -> dict:
    """Assemble the evaluation report saved next to the model.

    Args:
        y_test: True labels for the held-out split.
        predictions: Predicted labels for the held-out split.
        data_path: Dataset that produced the model.
        row_count: Number of usable rows in the dataset.
        train_messages: Number of rows used for fitting.

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

    return {
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
        },
    }


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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Train the model, evaluate it, and write the artifacts.

    Args:
        argv: Optional argument list, forwarded to :func:`parse_args`.
    """
    args = parse_args(argv)
    frame = load_dataset(args.data)

    x_train, x_test, y_train, y_test = train_test_split(
        frame["message"],
        frame["target"],
        test_size=args.test_size,
        random_state=args.seed,
        stratify=frame["target"],
    )

    model = build_model()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    metrics = build_metrics_report(
        y_test,
        predictions,
        data_path=args.data,
        row_count=len(frame),
        train_messages=len(x_train),
    )

    model_path = args.model_dir / MODEL_FILENAME
    metrics_path = args.model_dir / METRICS_FILENAME

    args.model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Training messages: {len(x_train)}")
    print(f"Test messages: {len(x_test)}")
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
    print(f"\nModel saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as exc:
        sys.exit(f"Training failed: {exc}")
