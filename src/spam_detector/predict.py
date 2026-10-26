"""Classify one message from the command line.

Usage::

    spam-detector-predict "Congratulations, you won a free prize!"
    spam-detector-predict "Call me when you land" --threshold 0.7
    spam-detector-predict "Claim your cash now" --explain --json

Messages are passed as a single argument, so quote them in the shell.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from spam_detector.config import model_path, spam_threshold
from spam_detector.model import (
    DEFAULT_TOP_TOKENS,
    LABELS,
    explain_prediction,
    load_model,
    predict_spam,
    predict_spam_probability,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description="Classify a single SMS/email message as spam or ham.",
    )
    parser.add_argument("message", help="The message text, quoted in the shell.")
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Model artifact (default: SPAM_MODEL_PATH or models/spam_model.joblib).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Spam probability threshold (default: SPAM_THRESHOLD or 0.5).",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Show the tokens that most influenced the verdict.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=DEFAULT_TOP_TOKENS,
        help=f"Tokens to show per direction with --explain (default: "
        f"{DEFAULT_TOP_TOKENS}).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a single JSON object instead of human-readable text.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Load the model, classify the message, and print the verdict.

    Args:
        argv: Optional argument list, forwarded to :func:`parse_args`.
    """
    args = parse_args(argv)
    threshold = args.threshold if args.threshold is not None else spam_threshold()

    try:
        artifact = args.model if args.model is not None else model_path()
        model = load_model(artifact)
        probability = predict_spam_probability(model, args.message)
        is_spam = predict_spam(model, args.message, threshold=threshold)
        explanations = (
            explain_prediction(model, args.message, top_k=args.top)
            if args.explain
            else None
        )
    except (OSError, TypeError, ValueError) as exc:
        sys.exit(f"Prediction failed: {exc}")

    label = LABELS[int(is_spam)]

    if args.json:
        print(
            json.dumps(
                {
                    "label": label,
                    "is_spam": is_spam,
                    "spam_probability": round(probability, 6),
                    "threshold": threshold,
                    "explanations": explanations,
                }
            )
        )
        return

    print(f"{label}  (spam probability {probability:.1%}, threshold {threshold:.2f})")

    if explanations:
        spam_tokens = [
            contribution
            for contribution in explanations
            if contribution["direction"] == "spam"
        ]
        ham_tokens = [
            contribution
            for contribution in explanations
            if contribution["direction"] == "ham"
        ]

        if spam_tokens:
            print(
                "  pushes spam: "
                + ", ".join(
                    f"{item['token']} ({item['weight']:+.3f})" for item in spam_tokens
                )
            )
        if ham_tokens:
            print(
                "  pushes ham:  "
                + ", ".join(
                    f"{item['token']} ({item['weight']:+.3f})" for item in ham_tokens
                )
            )


if __name__ == "__main__":
    main()
