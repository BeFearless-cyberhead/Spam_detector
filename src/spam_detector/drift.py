"""Lightweight drift monitoring for incoming messages.

The monitor keeps a rolling window of recent messages and compares two signals
against the statistics recorded at training time:

* **length** -- mean character count versus the training mean, expressed as a
  z-score;
* **vocabulary** -- the share of tokens never seen during training (the
  out-of-vocabulary, or OOV, rate).

Rising OOV rate is the classic early warning that the input distribution has
moved away from the training data and the model should be retrained. The
monitor is descriptive, not diagnostic: it flags *when* to look, not *why*.
"""

from __future__ import annotations

import re
import threading
from collections import Counter, deque
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

#: Same tokenisation the TF-IDF word vectoriser uses by default.
TOKEN_PATTERN = re.compile(r"(?u)\b\w\w+\b")

#: Signals at or beyond these values raise a watch / drift flag.
LENGTH_Z_WATCH = 2.0
LENGTH_Z_DRIFT = 3.0
OOV_WATCH = 0.15
OOV_DRIFT = 0.30

#: Messages needed before the monitor reports a status.
MINIMUM_MESSAGES = 20


def tokenize(message: str) -> list[str]:
    """Split a message into lowercase word tokens.

    Args:
        message: Raw message text.

    Returns:
        Tokens of two or more word characters, lowercased.
    """
    return [token.lower() for token in TOKEN_PATTERN.findall(message)]


def dataset_statistics(messages: Iterable[str]) -> dict[str, Any]:
    """Compute the reference statistics saved next to the model.

    Args:
        messages: Every training message.

    Returns:
        A JSON-serialisable mapping with length summaries, the vocabulary size,
        and the sorted training vocabulary used to measure OOV rates.
    """
    messages = list(messages)
    lengths = [len(message) for message in messages]

    vocabulary = sorted({token for message in messages for token in tokenize(message)})

    return {
        "messages": len(lengths),
        "length": {
            "mean": round(_mean(lengths), 3),
            "stdev": round(_stdev(lengths), 3),
            "median": round(_percentile(lengths, 0.5), 3),
            "p95": round(_percentile(lengths, 0.95), 3),
            "max": max(lengths) if lengths else 0,
        },
        "vocabulary_size": len(vocabulary),
        "vocabulary": vocabulary,
    }


@dataclass(frozen=True)
class ReferenceStatistics:
    """Training-time summary the monitor compares incoming traffic against."""

    mean_length: float
    stdev_length: float
    p95_length: float
    vocabulary: frozenset[str]

    @classmethod
    def from_metrics(
        cls,
        metrics: dict[str, Any] | None,
    ) -> ReferenceStatistics | None:
        """Build a reference from a parsed ``metrics.json``.

        Args:
            metrics: Parsed metrics file, or ``None`` when unavailable.

        Returns:
            The reference statistics, or ``None`` for older metric files that
            predate drift tracking.
        """
        if not metrics:
            return None

        statistics = metrics.get("training_statistics") or {}
        length = statistics.get("length") or {}

        if "mean" not in length:
            return None

        return cls(
            mean_length=float(length["mean"]),
            stdev_length=float(length.get("stdev", 0.0)),
            p95_length=float(length.get("p95", length["mean"])),
            vocabulary=frozenset(statistics.get("vocabulary") or ()),
        )


def _mean(values: list[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def _stdev(values: list[int]) -> float:
    if len(values) < 2:
        return 0.0

    average = _mean(values)
    variance = sum((value - average) ** 2 for value in values) / (len(values) - 1)
    return variance**0.5


def _percentile(values: list[int], fraction: float) -> float:
    """Linear-interpolated percentile, matching numpy's default method."""
    if not values:
        return 0.0

    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower

    return ordered[lower] * (1 - weight) + ordered[upper] * weight


class DriftMonitor:
    """Rolling-window monitor for message length and vocabulary drift.

    Args:
        reference: Training statistics to compare against. Without a reference
            the monitor still reports descriptive statistics.
        window_size: How many recent messages to keep.
        minimum_messages: Messages required before a status is reported.
    """

    def __init__(
        self,
        reference: ReferenceStatistics | None = None,
        *,
        window_size: int = 200,
        minimum_messages: int = MINIMUM_MESSAGES,
    ) -> None:
        """Initialise the monitor and its rolling window.

        Args:
            reference: Training statistics to compare against.
            window_size: How many recent messages to keep.
            minimum_messages: Messages required before a status is reported.

        Raises:
            ValueError: If either size is less than one.
        """
        if window_size < 1:
            raise ValueError("window_size must be at least 1.")
        if minimum_messages < 1:
            raise ValueError("minimum_messages must be at least 1.")

        self.reference = reference
        self.window_size = window_size
        self.minimum_messages = minimum_messages
        self._lengths: deque[int] = deque(maxlen=window_size)
        self._token_lists: deque[list[str]] = deque(maxlen=window_size)
        self._messages_seen = 0
        self._lock = threading.Lock()

    def observe(self, message: str) -> None:
        """Record one scored message.

        Args:
            message: The message that was just classified.

        Raises:
            TypeError: If ``message`` is not a string.
        """
        if not isinstance(message, str):
            raise TypeError("Message must be a string.")

        tokens = tokenize(message)

        with self._lock:
            self._lengths.append(len(message))
            self._token_lists.append(tokens)
            self._messages_seen += 1

    def reset(self) -> None:
        """Forget every observed message (used by tests and manual resets)."""
        with self._lock:
            self._lengths.clear()
            self._token_lists.clear()
            self._messages_seen = 0

    def snapshot(self) -> dict[str, Any]:
        """Describe the current window and any drift signals.

        Returns:
            A JSON-serialisable snapshot with the status (``insufficient_data``,
            ``reference_missing``, ``ok``, ``watch`` or ``drift``), window
            statistics, the comparison signals, and the most common terms.
        """
        with self._lock:
            lengths = list(self._lengths)
            token_lists = [list(tokens) for tokens in self._token_lists]
            messages_seen = self._messages_seen

        window_messages = len(lengths)
        mean_length = _mean(lengths)
        p95_length = _percentile(lengths, 0.95)

        counts = Counter(token for tokens in token_lists for token in tokens)
        total_tokens = sum(counts.values())

        snapshot: dict[str, Any] = {
            "status": "insufficient_data",
            "messages_seen": messages_seen,
            "window_size": self.window_size,
            "window_messages": window_messages,
            "minimum_messages": self.minimum_messages,
            "mean_length": round(mean_length, 3),
            "p95_length": round(p95_length, 3),
            "oov_rate": None,
            "length_zscore": None,
            "top_terms": dict(counts.most_common(10)),
            "reference": None,
        }

        reference = self.reference

        if reference is None:
            snapshot["status"] = "reference_missing"
            return snapshot

        snapshot["reference"] = {
            "mean_length": reference.mean_length,
            "stdev_length": reference.stdev_length,
            "p95_length": reference.p95_length,
            "vocabulary_size": len(reference.vocabulary),
        }

        if window_messages < self.minimum_messages:
            return snapshot

        if reference.stdev_length > 0:
            snapshot["length_zscore"] = round(
                (mean_length - reference.mean_length) / reference.stdev_length,
                3,
            )

        if total_tokens:
            out_of_vocabulary = sum(
                count
                for token, count in counts.items()
                if token not in reference.vocabulary
            )
            snapshot["oov_rate"] = round(out_of_vocabulary / total_tokens, 6)

        snapshot["status"] = self._classify(
            snapshot["length_zscore"],
            snapshot["oov_rate"],
        )

        return snapshot

    @staticmethod
    def _classify(length_zscore: float | None, oov_rate: float | None) -> str:
        """Turn the raw signals into a single status label."""
        length_is_drifting = (
            length_zscore is not None and abs(length_zscore) >= LENGTH_Z_DRIFT
        )
        vocabulary_is_drifting = oov_rate is not None and oov_rate >= OOV_DRIFT

        if length_is_drifting or vocabulary_is_drifting:
            return "drift"

        length_is_watchable = (
            length_zscore is not None and abs(length_zscore) >= LENGTH_Z_WATCH
        )
        vocabulary_is_watchable = oov_rate is not None and oov_rate >= OOV_WATCH

        if length_is_watchable or vocabulary_is_watchable:
            return "watch"

        return "ok"
