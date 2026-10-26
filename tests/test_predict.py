"""Tests for the single-message prediction CLI."""

import json

import joblib
import pytest

from spam_detector.predict import main, parse_args

SPAM_MESSAGE = "Win free cash prize claim now"
HAM_MESSAGE = "Team meeting tomorrow morning"


@pytest.fixture
def artifact(fitted_model, tmp_path):
    """The tiny fitted pipeline saved to a temporary artifact."""
    path = tmp_path / "spam_model.joblib"
    joblib.dump(fitted_model, path)
    return path


def test_predict_prints_the_verdict(artifact, capsys):
    main([SPAM_MESSAGE, "--model", str(artifact)])

    output = capsys.readouterr().out

    assert output.startswith("spam")
    assert "spam probability" in output


def test_predict_identifies_ham(artifact, capsys):
    main([HAM_MESSAGE, "--model", str(artifact)])

    assert capsys.readouterr().out.startswith("ham")


def test_predict_json_output(artifact, capsys):
    main([SPAM_MESSAGE, "--model", str(artifact), "--json"])

    payload = json.loads(capsys.readouterr().out)

    assert payload["label"] == "spam"
    assert payload["is_spam"] is True
    assert 0.0 <= payload["spam_probability"] <= 1.0
    assert payload["threshold"] == 0.5
    assert payload["explanations"] is None


def test_predict_threshold_can_flip_the_label(artifact, capsys):
    main([SPAM_MESSAGE, "--model", str(artifact), "--threshold", "1.0", "--json"])

    payload = json.loads(capsys.readouterr().out)

    assert payload["label"] == "ham"
    assert payload["threshold"] == 1.0


def test_predict_explain_prints_contributions(artifact, capsys):
    main([SPAM_MESSAGE, "--model", str(artifact), "--explain"])

    output = capsys.readouterr().out

    assert "pushes spam:" in output
    assert "free" in output


def test_predict_explain_json_includes_contributions(artifact, capsys):
    main([SPAM_MESSAGE, "--model", str(artifact), "--explain", "--json", "--top", "2"])

    payload = json.loads(capsys.readouterr().out)

    assert payload["explanations"]
    assert all(item["direction"] in {"spam", "ham"} for item in payload["explanations"])


def test_predict_fails_cleanly_without_an_artifact(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        main([SPAM_MESSAGE, "--model", str(tmp_path / "missing.joblib")])

    assert "Model not found" in str(excinfo.value)


def test_predict_fails_cleanly_on_blank_messages(artifact):
    with pytest.raises(SystemExit) as excinfo:
        main(["   ", "--model", str(artifact)])

    assert "non-empty" in str(excinfo.value)


def test_parse_args_defaults():
    args = parse_args([SPAM_MESSAGE])

    assert args.message == SPAM_MESSAGE
    assert args.model is None
    assert args.threshold is None
    assert args.explain is False
    assert args.json is False
    assert args.top == 5
