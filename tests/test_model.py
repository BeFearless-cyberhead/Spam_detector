import joblib
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion

from spam_detector.model import (
    build_model,
    explain_prediction,
    load_model,
    predict_spam,
    predict_spam_probability,
    validate_message,
)


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


def test_default_threshold_matches_the_probability(fitted_model):
    message = "Win free cash prize claim now"
    probability = predict_spam_probability(fitted_model, message)

    assert predict_spam(fitted_model, message) is (probability >= 0.5)


def test_threshold_trades_recall_for_precision(fitted_model):
    message = "Win free cash prize claim now"

    assert predict_spam(fitted_model, message, threshold=0.5) is True
    assert predict_spam(fitted_model, message, threshold=1.0) is False
    assert predict_spam(fitted_model, "Team meeting tomorrow", threshold=0.0) is True


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_invalid_threshold_is_rejected(fitted_model, threshold):
    with pytest.raises(ValueError, match="Threshold"):
        predict_spam(fitted_model, "hello", threshold=threshold)


def test_explain_prediction_reports_both_directions(fitted_model):
    contributions = explain_prediction(
        fitted_model,
        "Win free cash prize claim now",
        top_k=3,
    )

    assert contributions
    assert all(
        {"token", "weight", "source", "direction"} <= set(item)
        for item in contributions
    )

    spam_side = [item for item in contributions if item["direction"] == "spam"]
    ham_side = [item for item in contributions if item["direction"] == "ham"]

    assert spam_side
    assert all(item["weight"] > 0 for item in spam_side)
    assert all(item["weight"] < 0 for item in ham_side)
    assert len(spam_side) <= 3
    assert len(ham_side) <= 3


def test_explain_prediction_uses_word_and_char_features(fitted_model):
    contributions = explain_prediction(
        fitted_model,
        "WINNER!! claim your fr33 ca$h prize now",
        top_k=20,
    )

    sources = {item["source"] for item in contributions}

    assert "word" in sources
    assert "char" in sources


def test_explain_prediction_works_with_a_calibrated_model(calibrated_model):
    contributions = explain_prediction(
        calibrated_model,
        "Win free cash prize claim now",
        top_k=2,
    )

    assert contributions
    assert any(item["direction"] == "spam" for item in contributions)


@pytest.mark.parametrize("top_k", [0, -3])
def test_explain_rejects_invalid_top_k(fitted_model, top_k):
    with pytest.raises(ValueError, match="top_k"):
        explain_prediction(fitted_model, "hello", top_k=top_k)


def test_explain_rejects_blank_messages(fitted_model):
    with pytest.raises(ValueError):
        explain_prediction(fitted_model, "   ")


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


def test_build_model_is_calibrated_by_default():
    classifier = build_model().named_steps["classifier"]

    assert isinstance(classifier, CalibratedClassifierCV)
    assert classifier.method == "sigmoid"


def test_build_model_can_skip_calibration():
    classifier = build_model(calibrated=False).named_steps["classifier"]

    assert isinstance(classifier, LogisticRegression)


def test_build_model_combines_word_and_character_ngrams():
    features = build_model().named_steps["tfidf"]

    assert isinstance(features, FeatureUnion)
    assert [name for name, _ in features.transformer_list] == ["word", "char"]
    assert features.transformer_list[1][1].analyzer == "char_wb"


def test_build_model_can_skip_character_ngrams():
    features = build_model(use_char_ngrams=False).named_steps["tfidf"]

    assert [name for name, _ in features.transformer_list] == ["word"]


def test_build_model_honours_hyperparameters():
    model = build_model(
        calibrated=False,
        word_ngram_range=(1, 1),
        c=0.25,
    )
    word_vectorizer = dict(model.named_steps["tfidf"].transformer_list)["word"]

    assert word_vectorizer.ngram_range == (1, 1)
    assert model.named_steps["classifier"].C == 0.25
