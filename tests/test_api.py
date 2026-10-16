import json

import joblib
import pytest
from fastapi.testclient import TestClient

from api import create_app
from spam_model import build_model

TRAIN_MESSAGES = [
    "Team meeting tomorrow morning",
    "Please call me after class",
    "Dinner is ready at home",
    "Can you send the homework notes",
    "Win free cash prize claim now",
    "Congratulations claim your free prize",
    "Urgent winner cash reward claim now",
    "Free bonus cash prize click now",
]

TRAIN_LABELS = [0, 0, 0, 0, 1, 1, 1, 1]


@pytest.fixture
def model_path(tmp_path):
    """Write a small fitted model to a temporary directory."""
    model = build_model()
    model.fit(TRAIN_MESSAGES, TRAIN_LABELS)

    path = tmp_path / "spam_model.joblib"
    joblib.dump(model, path)
    return path


@pytest.fixture
def client(model_path):
    """A TestClient wired to the temporary model."""
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
    assert "train.py" in response.json()["detail"]


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
