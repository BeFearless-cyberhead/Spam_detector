import joblib
import pytest

from spam_model import (
    build_model,
    load_model,
    predict_spam,
    predict_spam_probability,
    validate_message,
)


@pytest.fixture
def fitted_model():
    messages = [
        "Team meeting tomorrow morning",
        "Please call me after class",
        "Dinner is ready at home",
        "Can you send the homework notes",
        "Win free cash prize claim now",
        "Congratulations claim your free prize",
        "Urgent winner cash reward claim now",
        "Free bonus cash prize click now",
    ]

    labels = [0, 0, 0, 0, 1, 1, 1, 1]

    model = build_model()
    model.fit(messages, labels)
    return model


@pytest.mark.parametrize("message", ["", " ", "\n\t"])
def test_rejects_blank_messages(message):
    with pytest.raises(ValueError):
        validate_message(message)


@pytest.mark.parametrize("message", [None, 123, [], True])
def test_rejects_non_string_messages(message):
    with pytest.raises(TypeError):
        validate_message(message)


def test_rejects_overlong_messages():
    with pytest.raises(ValueError):
        validate_message("x" * 10_001)


def test_trims_surrounding_whitespace():
    assert validate_message("  hello there  ") == "hello there"


def test_prediction_returns_python_boolean(fitted_model):
    result = predict_spam(fitted_model, "Please call me tomorrow")
    assert type(result) is bool


def test_learns_spam_training_example(fitted_model):
    assert (
        predict_spam(
            fitted_model,
            "Win free cash prize claim now",
        )
        is True
    )


def test_learns_ham_training_example(fitted_model):
    assert (
        predict_spam(
            fitted_model,
            "Team meeting tomorrow morning",
        )
        is False
    )


def test_saved_model_preserves_prediction(fitted_model, tmp_path):
    model_path = tmp_path / "test_model.joblib"
    message = "Free cash prize claim now"

    before = predict_spam(fitted_model, message)

    joblib.dump(fitted_model, model_path)
    restored_model = joblib.load(model_path)

    after = predict_spam(restored_model, message)

    assert before == after


def test_probability_is_bounded(fitted_model):
    for message in ["Win free cash prize now", "Team meeting tomorrow"]:
        probability = predict_spam_probability(fitted_model, message)

        assert isinstance(probability, float)
        assert 0.0 <= probability <= 1.0


def test_probability_agrees_with_the_hard_label(fitted_model):
    spam_probability = predict_spam_probability(
        fitted_model,
        "Win free cash prize claim now",
    )
    ham_probability = predict_spam_probability(
        fitted_model,
        "Team meeting tomorrow morning",
    )

    assert spam_probability > ham_probability


def test_probability_rejects_invalid_input(fitted_model):
    with pytest.raises(ValueError):
        predict_spam_probability(fitted_model, "   ")

    with pytest.raises(TypeError):
        predict_spam_probability(fitted_model, None)


def test_load_model_reads_a_saved_pipeline(fitted_model, tmp_path):
    model_path = tmp_path / "pipeline.joblib"
    joblib.dump(fitted_model, model_path)

    loaded = load_model(model_path)

    assert predict_spam(loaded, "Win free cash prize claim now") is True


def test_load_model_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "absent.joblib")


def test_load_model_rejects_unexpected_artifact(tmp_path):
    model_path = tmp_path / "not_a_pipeline.joblib"
    joblib.dump({"not": "a pipeline"}, model_path)

    with pytest.raises(TypeError):
        load_model(model_path)


def test_build_model_has_expected_pipeline_steps():
    assert [name for name, _ in build_model().steps] == ["tfidf", "classifier"]
