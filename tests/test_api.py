import json

import pytest
from fastapi.testclient import TestClient

from spam_detector.api import create_app


@pytest.fixture
def client(model_path):
    """A TestClient wired to the tiny fitted model."""
    with TestClient(create_app(model_path)) as test_client:
        yield test_client


def test_health_reports_loaded_model(client, model_path):
    response = client.get("/health")

    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_path"] == str(model_path)


def test_health_ok_but_unloaded_without_artifact(tmp_path):
    with TestClient(create_app(tmp_path / "missing.joblib")) as test_client:
        body = test_client.get("/health").json()

    assert body["status"] == "ok"
    assert body["model_loaded"] is False


def test_predict_flags_spam(client):
    response = client.post(
        "/predict",
        json={"message": "Win free cash prize claim now"},
    )

    assert response.status_code == 200

    body = response.json()
    assert body["is_spam"] is True
    assert body["label"] == "spam"
    assert 0.0 <= body["spam_probability"] <= 1.0


def test_predict_flags_ham(client):
    response = client.post(
        "/predict",
        json={"message": "Team meeting tomorrow morning"},
    )

    assert response.status_code == 200

    body = response.json()
    assert body["is_spam"] is False
    assert body["label"] == "ham"


@pytest.mark.parametrize("message", ["", " ", "\n\t"])
def test_predict_rejects_blank_messages(client, message):
    response = client.post("/predict", json={"message": message})
    assert response.status_code == 422


def test_predict_rejects_oversized_message(client):
    response = client.post("/predict", json={"message": "x" * 10_001})
    assert response.status_code == 422


def test_predict_requires_message_field(client):
    response = client.post("/predict", json={})

    assert response.status_code == 422
    assert "message" in response.text


def test_predict_returns_503_without_artifact(tmp_path):
    with TestClient(create_app(tmp_path / "missing.joblib")) as test_client:
        response = test_client.post("/predict", json={"message": "hello"})

    assert response.status_code == 503
    assert "spam_detector.train" in response.json()["detail"]


def test_model_info_without_metrics(model_path):
    with TestClient(create_app(model_path)) as test_client:
        body = test_client.get("/model-info").json()

    assert body["metrics_available"] is False
    assert body["metrics"] is None


def test_model_info_returns_stored_metrics(model_path):
    metrics = {
        "accuracy": 0.98,
        "classification_report": {"accuracy": 0.98},
    }
    (model_path.parent / "metrics.json").write_text(
        json.dumps(metrics),
        encoding="utf-8",
    )

    with TestClient(create_app(model_path)) as test_client:
        body = test_client.get("/model-info").json()

    assert body["metrics_available"] is True
    assert body["metrics"]["accuracy"] == 0.98


def test_model_info_reports_corrupt_metrics(model_path):
    (model_path.parent / "metrics.json").write_text("{not json", encoding="utf-8")

    with TestClient(create_app(model_path)) as test_client:
        response = test_client.get("/model-info")

    assert response.status_code == 500


def test_openapi_schema_is_served(client):
    schema = client.get("/openapi.json").json()

    assert schema["info"]["title"] == "SMS Spam Detector API"
    assert "/predict" in schema["paths"]
    assert "/predict/batch" in schema["paths"]
    assert "/feedback" in schema["paths"]
    assert "/drift" in schema["paths"]


def test_predict_reports_the_threshold_used(client):
    response = client.post(
        "/predict",
        json={"message": "Win free cash prize claim now", "threshold": 0.9},
    )

    assert response.status_code == 200
    assert response.json()["threshold"] == 0.9


def test_predict_threshold_can_flip_the_label(client):
    message = "Win free cash prize claim now"

    assert client.post("/predict", json={"message": message}).json()["is_spam"] is True

    body = client.post("/predict", json={"message": message, "threshold": 1.0}).json()

    assert body["is_spam"] is False
    assert body["label"] == "ham"
    assert body["spam_probability"] < 1.0


def test_predict_returns_explanations_on_request(client):
    body = client.post(
        "/predict",
        json={"message": "Win free cash prize claim now", "explain": True},
    ).json()

    assert body["explanations"]

    contribution = body["explanations"][0]
    assert set(contribution) == {"token", "weight", "source", "direction"}
    assert contribution["direction"] in {"spam", "ham"}


def test_predict_omits_explanations_by_default(client):
    body = client.post(
        "/predict",
        json={"message": "Win free cash prize claim now"},
    ).json()

    assert body["explanations"] is None


def test_batch_predicts_every_message_in_order(client):
    response = client.post(
        "/predict/batch",
        json={
            "messages": [
                "Win free cash prize claim now",
                "Team meeting tomorrow morning",
            ]
        },
    )

    assert response.status_code == 200

    body = response.json()
    assert body["messages_processed"] == 2
    assert [result["label"] for result in body["results"]] == ["spam", "ham"]
    assert all(
        result["is_spam"] == (result["label"] == "spam") for result in body["results"]
    )


def test_batch_can_request_explanations(client):
    body = client.post(
        "/predict/batch",
        json={"messages": ["Win free cash prize"], "explain": True},
    ).json()

    assert body["results"][0]["explanations"]


def test_batch_rejects_an_empty_list(client):
    response = client.post("/predict/batch", json={"messages": []})

    assert response.status_code == 422


def test_batch_rejects_oversized_batches(client):
    response = client.post(
        "/predict/batch",
        json={"messages": ["hello"] * 101},
    )

    assert response.status_code == 422


def test_batch_rejects_blank_items(client):
    response = client.post("/predict/batch", json={"messages": ["   "]})

    assert response.status_code == 422


def test_batch_returns_503_without_artifact(tmp_path):
    with TestClient(create_app(tmp_path / "missing.joblib")) as test_client:
        response = test_client.post("/predict/batch", json={"messages": ["hello"]})

    assert response.status_code == 503


def test_feedback_appends_jsonl_records(model_path, tmp_path):
    feedback_log = tmp_path / "feedback" / "feedback.jsonl"

    with TestClient(create_app(model_path, feedback_log)) as test_client:
        response = test_client.post(
            "/feedback",
            json={
                "message": "Congratulations, claim your prize",
                "predicted_label": "ham",
                "correct_label": "spam",
                "spam_probability": 0.41,
                "note": "promotion from my bank",
            },
        )

        assert response.status_code == 200

        body = response.json()
        assert body["logged"] is True
        assert body["feedback_file"] == str(feedback_log)
        assert body["recorded_at_utc"].endswith("+00:00")

        assert feedback_log.exists()
        records = [
            json.loads(line)
            for line in feedback_log.read_text(encoding="utf-8").splitlines()
        ]

    assert len(records) == 1
    assert records[0]["predicted_label"] == "ham"
    assert records[0]["correct_label"] == "spam"
    assert records[0]["note"] == "promotion from my bank"
    assert records[0]["spam_probability"] == 0.41


@pytest.mark.parametrize(
    "payload",
    [
        {"message": "hello", "predicted_label": "maybe", "correct_label": "spam"},
        {
            "message": "hello",
            "predicted_label": "ham",
            "correct_label": "spam",
            "note": "x" * 501,
        },
        {"message": "", "predicted_label": "ham", "correct_label": "spam"},
    ],
)
def test_feedback_rejects_invalid_payloads(client, payload):
    assert client.post("/feedback", json=payload).status_code == 422


def test_drift_without_a_reference_model_is_reported(model_path):
    with TestClient(create_app(model_path)) as test_client:
        body = test_client.get("/drift").json()

    assert body["status"] == "reference_missing"
    assert body["reference"] is None


def test_drift_counts_scored_messages(client):
    client.post("/predict", json={"message": "Win free cash prize claim now"})
    client.post("/predict", json={"message": "Team meeting tomorrow morning"})

    body = client.get("/drift").json()

    assert body["messages_seen"] == 2
    assert body["window_messages"] == 2


def test_drift_flags_out_of_vocabulary_traffic(model_path):
    metrics = {
        "training_statistics": {
            "messages": 100,
            "length": {"mean": 20.0, "stdev": 2.0, "p95": 30.0},
            "vocabulary_size": 3,
            "vocabulary": ["meeting", "team", "tomorrow"],
        }
    }
    (model_path.parent / "metrics.json").write_text(
        json.dumps(metrics),
        encoding="utf-8",
    )

    with TestClient(create_app(model_path)) as test_client:
        for index in range(20):
            test_client.post(
                "/predict",
                json={"message": f"unseen gibberish tokens {index}"},
            )

        body = test_client.get("/drift").json()

    assert body["status"] == "drift"
    assert body["oov_rate"] > 0.5
    assert body["reference"]["vocabulary_size"] == 3


def test_resolve_model_path_prefers_the_explicit_override(tmp_path, monkeypatch):
    from spam_detector.api import resolve_model_path

    monkeypatch.setenv("SPAM_MODEL_PATH", str(tmp_path / "from-env.joblib"))

    assert resolve_model_path(tmp_path / "explicit.joblib") == (
        tmp_path / "explicit.joblib"
    )
    assert resolve_model_path() == tmp_path / "from-env.joblib"
