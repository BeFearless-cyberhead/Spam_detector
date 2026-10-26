"""Unit tests for the drift monitor and training statistics."""

import pytest

from spam_detector.drift import (
    DriftMonitor,
    ReferenceStatistics,
    dataset_statistics,
    tokenize,
)


def reference() -> ReferenceStatistics:
    """A small, stable training reference for the monitor tests."""
    return ReferenceStatistics(
        mean_length=20.0,
        stdev_length=10.0,
        p95_length=30.0,
        vocabulary=frozenset({"meeting", "team", "tomorrow", "free", "prize", "win"}),
    )


def test_tokenize_lowercases_and_drops_single_characters():
    assert tokenize("Hi! WIN free CASH 2 U") == ["hi", "win", "free", "cash"]


def test_dataset_statistics_summarises_lengths_and_vocabulary():
    statistics = dataset_statistics(
        ["team meeting tomorrow", "free prize", "win free cash prize"]
    )

    assert statistics["messages"] == 3
    assert statistics["length"]["mean"] == pytest.approx((21 + 10 + 19) / 3, rel=1e-3)
    assert statistics["length"]["max"] == 21
    assert statistics["vocabulary_size"] == len(
        {"team", "meeting", "tomorrow", "free", "prize", "win", "cash"}
    )
    assert "meeting" in statistics["vocabulary"]


def test_dataset_statistics_handles_an_empty_corpus():
    statistics = dataset_statistics([])

    assert statistics["messages"] == 0
    assert statistics["length"]["mean"] == 0
    assert statistics["vocabulary"] == []


def test_reference_from_metrics_reads_training_statistics():
    parsed = ReferenceStatistics.from_metrics(
        {
            "training_statistics": {
                "length": {"mean": 50.0, "stdev": 10.0, "p95": 70.0},
                "vocabulary": ["a", "b"],
            }
        }
    )

    assert parsed is not None
    assert parsed.mean_length == 50.0
    assert parsed.p95_length == 70.0
    assert parsed.vocabulary == frozenset({"a", "b"})


@pytest.mark.parametrize("metrics", [None, {}, {"training_statistics": {}}])
def test_reference_from_metrics_tolerates_older_reports(metrics):
    assert ReferenceStatistics.from_metrics(metrics) is None


def test_monitor_reports_insufficient_data_first():
    monitor = DriftMonitor(reference(), minimum_messages=5)
    monitor.observe("team meeting tomorrow")

    snapshot = monitor.snapshot()

    assert snapshot["status"] == "insufficient_data"
    assert snapshot["window_messages"] == 1
    assert snapshot["messages_seen"] == 1
    assert snapshot["reference"]["mean_length"] == 20.0
    assert snapshot["length_zscore"] is None


def test_monitor_without_reference_reports_reference_missing():
    monitor = DriftMonitor()
    monitor.observe("team meeting tomorrow")

    assert monitor.snapshot()["status"] == "reference_missing"


def test_monitor_reports_ok_for_in_distribution_messages():
    monitor = DriftMonitor(reference(), minimum_messages=3)

    for message in ["team meeting tomorrow", "free prize team", "win free prize"]:
        monitor.observe(message)

    snapshot = monitor.snapshot()

    assert snapshot["status"] == "ok"
    assert snapshot["oov_rate"] == 0.0
    assert abs(snapshot["length_zscore"]) < 2


def test_monitor_watches_an_oov_increase():
    monitor = DriftMonitor(reference(), minimum_messages=1)

    monitor.observe("team meeting mystery tomorrow")

    snapshot = monitor.snapshot()

    assert snapshot["status"] == "watch"
    assert 0.15 <= snapshot["oov_rate"] < 0.3


def test_monitor_flags_drift_on_vocabulary_shift():
    monitor = DriftMonitor(reference(), minimum_messages=2)

    monitor.observe("zzz qqq xxx")
    monitor.observe("yyy vvv uuu")

    snapshot = monitor.snapshot()

    assert snapshot["status"] == "drift"
    assert snapshot["oov_rate"] == 1.0


def test_monitor_flags_drift_on_length_shift():
    monitor = DriftMonitor(reference(), minimum_messages=2)

    for _ in range(2):
        monitor.observe("team meeting tomorrow free prize " * 5)

    snapshot = monitor.snapshot()

    assert snapshot["status"] == "drift"
    assert snapshot["oov_rate"] == 0.0
    assert snapshot["length_zscore"] > 3


def test_monitor_window_is_bounded():
    monitor = DriftMonitor(reference(), window_size=3, minimum_messages=2)

    for index in range(10):
        monitor.observe(f"team meeting {index}")

    snapshot = monitor.snapshot()

    assert snapshot["messages_seen"] == 10
    assert snapshot["window_messages"] == 3


def test_monitor_reports_top_terms():
    monitor = DriftMonitor(reference(), minimum_messages=1)
    monitor.observe("free free prize meeting")

    assert monitor.snapshot()["top_terms"]["free"] == 2


def test_monitor_reset_clears_the_window():
    monitor = DriftMonitor(reference(), minimum_messages=1)
    monitor.observe("team meeting")
    monitor.reset()

    snapshot = monitor.snapshot()

    assert snapshot["messages_seen"] == 0
    assert snapshot["window_messages"] == 0
    assert snapshot["status"] == "insufficient_data"


def test_monitor_rejects_non_string_messages():
    with pytest.raises(TypeError):
        DriftMonitor().observe(None)


@pytest.mark.parametrize(
    ("window_size", "minimum_messages"),
    [(0, 1), (-1, 1), (10, 0)],
)
def test_monitor_rejects_invalid_sizes(window_size, minimum_messages):
    with pytest.raises(ValueError):
        DriftMonitor(window_size=window_size, minimum_messages=minimum_messages)


def test_monitor_handles_zero_variance_reference():
    flat = ReferenceStatistics(
        mean_length=10.0,
        stdev_length=0.0,
        p95_length=10.0,
        vocabulary=frozenset({"team"}),
    )
    monitor = DriftMonitor(flat, minimum_messages=1)
    monitor.observe("team")

    snapshot = monitor.snapshot()

    assert snapshot["length_zscore"] is None
    assert snapshot["status"] == "ok"
