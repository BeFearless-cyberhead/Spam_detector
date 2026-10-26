import json
from pathlib import Path

import joblib
import pytest
from sklearn.pipeline import Pipeline

from spam_detector.model import build_model, predict_spam
from spam_detector.train import (
    SEARCH_GRID,
    load_dataset,
    main,
    parse_args,
    select_hyperparameters,
    sha256_of,
)

SAMPLE_PATH = Path(__file__).resolve().parent.parent / "samples" / "sample_messages.tsv"


def write_dataset(tmp_path: Path, rows: list[tuple[str, str]]) -> Path:
    """Write a tab-separated dataset and return its path."""
    path = tmp_path / "dataset.tsv"
    path.write_text(
        "".join(f"{label}\t{message}\n" for label, message in rows),
        encoding="utf-8",
    )
    return path


def balanced_rows(prefix: str = "message") -> list[tuple[str, str]]:
    """Five ham and five spam rows, enough to pass validation."""
    return [
        *[("ham", f"{prefix} ham {index}") for index in range(5)],
        *[("spam", f"{prefix} spam {index}") for index in range(5)],
    ]


def test_sample_dataset_is_valid():
    frame = load_dataset(SAMPLE_PATH)

    assert set(frame["label"]) == {"ham", "spam"}
    assert set(frame["target"]) == {0, 1}
    assert frame["message"].is_unique
    assert len(frame) >= 10


def test_sample_dataset_is_reproducible_across_loads():
    first = load_dataset(SAMPLE_PATH)
    second = load_dataset(SAMPLE_PATH)

    assert first.equals(second)


def test_missing_dataset_raises_with_hint(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"spam_detector\.data"):
        load_dataset(tmp_path / "nope.tsv")


def test_empty_dataset_is_rejected(tmp_path):
    path = write_dataset(tmp_path, [])

    with pytest.raises(ValueError, match="no usable messages"):
        load_dataset(path)


def test_blank_lines_are_skipped(tmp_path):
    path = tmp_path / "dataset.tsv"
    path.write_text(
        "\n".join(f"{label}\t{message}" for label, message in balanced_rows())
        + "\n\n   \n",
        encoding="utf-8",
    )

    assert len(load_dataset(path)) == 10


def test_malformed_record_is_rejected(tmp_path):
    path = write_dataset(
        tmp_path,
        [*balanced_rows(), ("not-a-label", "mystery message")],
    )

    with pytest.raises(ValueError, match="Invalid dataset record"):
        load_dataset(path)


def test_row_without_tab_separator_is_rejected(tmp_path):
    path = tmp_path / "dataset.tsv"
    path.write_text(
        "".join(f"{label}\t{message}\n" for label, message in balanced_rows())
        + "ham this row uses spaces\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid dataset record"):
        load_dataset(path)


def test_conflicting_duplicate_labels_are_rejected(tmp_path):
    path = write_dataset(
        tmp_path,
        [*balanced_rows(), ("ham", "duplicate text"), ("spam", "duplicate text")],
    )

    with pytest.raises(ValueError, match="conflicting labels"):
        load_dataset(path)


def test_exact_duplicates_are_dropped(tmp_path):
    path = write_dataset(
        tmp_path,
        [*balanced_rows(), ("ham", "message ham 0"), ("spam", "message spam 0")],
    )

    assert len(load_dataset(path)) == 10


def test_single_class_dataset_is_rejected(tmp_path):
    path = write_dataset(
        tmp_path,
        [("ham", f"ham only {index}") for index in range(10)],
    )

    with pytest.raises(ValueError, match="unique messages per class"):
        load_dataset(path)


def test_too_few_messages_per_class_is_rejected(tmp_path):
    rows = [*[("ham", f"ham {index}") for index in range(6)]]
    rows += [("spam", f"spam {index}") for index in range(4)]

    with pytest.raises(ValueError, match="unique messages per class"):
        load_dataset(write_dataset(tmp_path, rows))


def test_test_size_must_split_both_classes(tmp_path):
    with pytest.raises(ValueError):
        main(
            [
                "--data",
                str(write_dataset(tmp_path, balanced_rows())),
                "--model-dir",
                str(tmp_path / "out"),
                "--test-size",
                "1.5",
            ]
        )


def test_training_writes_model_and_metrics(tmp_path):
    output_dir = tmp_path / "artifacts"

    main(
        [
            "--data",
            str(SAMPLE_PATH),
            "--model-dir",
            str(output_dir),
            "--test-size",
            "0.3",
            "--seed",
            "7",
        ]
    )

    model_path = output_dir / "spam_model.joblib"
    metrics_path = output_dir / "metrics.json"

    assert model_path.exists()
    assert metrics_path.exists()

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    assert metrics["training_messages"] + metrics["test_messages"] == len(
        load_dataset(SAMPLE_PATH)
    )
    assert metrics["test_messages"] > 0
    assert metrics["confusion_matrix_label_order"] == ["ham", "spam"]
    assert metrics["training"]["dataset"] == SAMPLE_PATH.name
    assert metrics["training"]["dataset_rows"] == len(load_dataset(SAMPLE_PATH))
    assert len(metrics["training"]["dataset_sha256"]) == 64


def test_training_reports_calibration_and_drift_statistics(tmp_path):
    main(
        [
            "--data",
            str(SAMPLE_PATH),
            "--model-dir",
            str(tmp_path),
            "--test-size",
            "0.3",
        ]
    )

    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))

    assert metrics["training"]["calibrated"] is True
    assert metrics["training"]["calibration_method"] == "sigmoid"

    probability = metrics["probability"]
    assert probability["threshold"] == 0.5
    assert 0.0 <= probability["brier_score"] <= 1.0
    assert probability["log_loss"] >= 0.0
    assert set(probability["threshold_metrics"]) == {"precision", "recall", "f1"}

    statistics = metrics["training_statistics"]
    assert statistics["messages"] == len(load_dataset(SAMPLE_PATH))
    assert statistics["vocabulary_size"] > 0
    assert statistics["vocabulary"]
    assert statistics["length"]["mean"] > 0


def test_training_can_skip_calibration(tmp_path):
    main(
        [
            "--data",
            str(SAMPLE_PATH),
            "--model-dir",
            str(tmp_path),
            "--test-size",
            "0.3",
            "--no-calibrate",
        ]
    )

    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))

    assert metrics["training"]["calibrated"] is False
    assert metrics["training"]["calibration_method"] is None


def test_training_uses_the_requested_threshold(tmp_path):
    main(
        [
            "--data",
            str(SAMPLE_PATH),
            "--model-dir",
            str(tmp_path),
            "--test-size",
            "0.3",
            "--threshold",
            "0.3",
        ]
    )

    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))

    assert metrics["probability"]["threshold"] == 0.3


def test_training_rejects_an_invalid_threshold(tmp_path):
    with pytest.raises(ValueError, match="threshold"):
        main(
            [
                "--data",
                str(SAMPLE_PATH),
                "--model-dir",
                str(tmp_path),
                "--threshold",
                "1.5",
            ]
        )


def test_training_records_a_hyperparameter_search(tmp_path):
    main(
        [
            "--data",
            str(SAMPLE_PATH),
            "--model-dir",
            str(tmp_path),
            "--test-size",
            "0.3",
            "--search",
            "--cv",
            "2",
        ]
    )

    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    search = metrics["training"]["hyperparameter_search"]

    assert search["selection_metric"] == "spam_f1"
    assert search["cv_folds"] == 2
    assert len(search["candidates"]) == len(SEARCH_GRID)
    assert search["params"]["use_char_ngrams"] in {True, False}


def test_select_hyperparameters_picks_the_best_candidate():
    frame = load_dataset(SAMPLE_PATH)

    report = select_hyperparameters(
        frame["message"],
        frame["target"],
        seed=1,
        grid=[
            {"use_char_ngrams": True, "word_ngram_range": (1, 2), "c": 1.0},
            {"use_char_ngrams": False, "word_ngram_range": (1, 2), "c": 1.0},
        ],
    )

    assert report["cv_folds"] == 5
    assert len(report["candidates"]) == 2
    assert report["mean_f1"] == max(
        candidate["mean_f1"] for candidate in report["candidates"]
    )
    assert set(report["params"]) == {"use_char_ngrams", "word_ngram_range", "c"}
    assert isinstance(report["params"]["word_ngram_range"], list)


def test_search_reduces_folds_for_tiny_datasets():
    report = select_hyperparameters(
        ["aa bb", "cc dd", "ee ff", "gg hh", "ii jj", "kk ll", "mm nn", "oo pp"],
        [0, 0, 0, 0, 1, 1, 1, 1],
        grid=[{"use_char_ngrams": False, "word_ngram_range": (1, 2), "c": 1.0}],
    )

    assert report["cv_folds"] == 4


def test_saved_model_matches_live_pipeline(tmp_path):
    main(
        [
            "--data",
            str(SAMPLE_PATH),
            "--model-dir",
            str(tmp_path),
            "--test-size",
            "0.3",
        ]
    )

    model = joblib.load(tmp_path / "spam_model.joblib")

    assert isinstance(model, Pipeline)
    assert isinstance(predict_spam(model, "Win a free cash prize now"), bool)
    assert isinstance(build_model(), Pipeline)


@pytest.mark.parametrize(
    ("args", "attribute", "expected"),
    [
        ([], "test_size", 0.2),
        ([], "seed", 42),
        (["--test-size", "0.25"], "test_size", 0.25),
        (["--seed", "1"], "seed", 1),
        (["--model-dir", "out"], "model_dir", Path("out")),
        ([], "search", False),
        (["--search"], "search", True),
        ([], "no_calibrate", False),
        (["--no-calibrate"], "no_calibrate", True),
        ([], "cv", 5),
        (["--cv", "3"], "cv", 3),
        ([], "no_char_ngrams", False),
        (["--no-char-ngrams"], "no_char_ngrams", True),
    ],
)
def test_parse_args(args, attribute, expected):
    assert getattr(parse_args(args), attribute) == expected


def test_parse_args_reads_the_default_threshold_from_the_environment(monkeypatch):
    monkeypatch.setenv("SPAM_THRESHOLD", "0.35")

    assert parse_args([]).threshold == 0.35


def test_default_dataset_path_lives_in_data_dir():
    assert parse_args([]).data.name == "SMSSpamCollection"


def test_sha256_is_stable(tmp_path):
    path = tmp_path / "file.txt"
    path.write_text("stable", encoding="utf-8")

    assert sha256_of(path) == sha256_of(path)
    assert len(sha256_of(path)) == 64
