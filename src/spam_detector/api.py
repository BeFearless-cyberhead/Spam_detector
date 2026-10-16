"""FastAPI service exposing the spam classifier over HTTP.

Run locally::

    python -m uvicorn spam_detector.api:app --reload --port 8000
    # or, once the package is installed: spam-detector-api

Interactive docs are then available at http://127.0.0.1:8000/docs.

Configuration (environment variables):

* ``SPAM_MODEL_PATH`` -- path to the ``joblib`` artifact, default
  ``models/spam_model.joblib``.
* ``API_HOST`` / ``API_PORT`` -- bind address for the ``spam-detector-api``
  entry point.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from spam_detector.config import (
    METRICS_FILENAME,
)
from spam_detector.config import (
    model_path as default_model_path,
)
from spam_detector.model import (
    LABELS,
    MAX_MESSAGE_LENGTH,
    load_model,
    predict_spam,
    predict_spam_probability,
)

# Containers must bind all interfaces, hence 0.0.0.0 as the default host.
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000


class PredictRequest(BaseModel):
    """A single message submitted for classification."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=MAX_MESSAGE_LENGTH,
        description="Raw SMS/email text to classify.",
        examples=["Congratulations! You have won a free prize, claim now."],
    )


class PredictResponse(BaseModel):
    """The classifier's verdict for one message."""

    label: str = Field(..., description="Either 'spam' or 'ham'.", examples=["spam"])
    is_spam: bool = Field(..., description="Convenience boolean for the label.")
    spam_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence that the message is spam.",
        examples=[0.93],
    )


class HealthResponse(BaseModel):
    """Liveness/readiness probe payload."""

    status: str
    model_loaded: bool
    model_path: str


class ModelInfoResponse(BaseModel):
    """Service metadata plus the stored evaluation metrics, when available."""

    model_path: str
    metrics_available: bool
    metrics: dict[str, Any] | None = None


def resolve_model_path(override: str | Path | None = None) -> Path:
    """Work out which model artifact to serve.

    Args:
        override: Explicit path, taking precedence over the environment.

    Returns:
        The resolved model path (it may not exist yet).
    """
    if override is not None:
        return Path(override)

    return default_model_path()


def create_app(model_path: str | Path | None = None) -> FastAPI:
    """Build the FastAPI application.

    The model is loaded lazily on the first prediction so the service can start
    (and report an unhealthy status) even when no artifact has been trained yet.

    Args:
        model_path: Optional explicit model location; mainly used by tests.

    Returns:
        A configured :class:`~fastapi.FastAPI` instance.
    """
    resolved_path = resolve_model_path(model_path)

    app = FastAPI(
        title="SMS Spam Detector API",
        version="1.0.0",
        summary="TF-IDF + logistic-regression spam classifier.",
        description=(
            "REST wrapper around the scikit-learn pipeline used by the "
            "Streamlit demo. POST a message to `/predict` to get a verdict "
            "and a confidence score."
        ),
    )

    cache: dict[str, Any] = {"model": None}

    def get_model() -> Any:
        """Return the cached pipeline, loading it on first use."""
        if cache["model"] is None:
            if not resolved_path.exists():
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=(
                        "Model artifact is missing. Train it with "
                        "`python -m spam_detector.train` or set SPAM_MODEL_PATH."
                    ),
                )

            try:
                cache["model"] = load_model(resolved_path)
            except (OSError, TypeError) as exc:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"Model could not be loaded: {exc}",
                ) from exc

        return cache["model"]

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    def health() -> HealthResponse:
        """Report whether the process is up and whether a model is on disk."""
        return HealthResponse(
            status="ok",
            model_loaded=resolved_path.exists(),
            model_path=str(resolved_path),
        )

    @app.get("/model-info", response_model=ModelInfoResponse, tags=["ops"])
    def model_info() -> ModelInfoResponse:
        """Return the stored evaluation metrics produced by the trainer."""
        metrics_path = resolved_path.parent / METRICS_FILENAME

        if not metrics_path.exists():
            return ModelInfoResponse(
                model_path=str(resolved_path),
                metrics_available=False,
            )

        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Metrics file could not be read: {exc}",
            ) from exc

        return ModelInfoResponse(
            model_path=str(resolved_path),
            metrics_available=True,
            metrics=metrics,
        )

    @app.post(
        "/predict",
        response_model=PredictResponse,
        tags=["prediction"],
        responses={503: {"description": "Model artifact unavailable."}},
    )
    def predict(payload: PredictRequest) -> PredictResponse:
        """Classify a single message as spam or ham.

        Args:
            payload: The request body containing the message.

        Returns:
            The predicted label, an ``is_spam`` boolean, and the spam
            probability.
        """
        model = get_model()
        message = payload.message

        try:
            is_spam = predict_spam(model, message)
            probability = predict_spam_probability(model, message)
        except (TypeError, ValueError) as exc:
            # 422 = Unprocessable Content. The literal keeps compatibility with
            # Starlette versions where this status constant was renamed.
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        return PredictResponse(
            label=LABELS[int(is_spam)],
            is_spam=is_spam,
            spam_probability=round(probability, 6),
        )

    return app


def serve() -> None:
    """Run the API with uvicorn (``spam-detector-api`` console script)."""
    uvicorn.run(
        "spam_detector.api:app",
        host=os.getenv("API_HOST", DEFAULT_HOST),
        port=int(os.getenv("API_PORT", str(DEFAULT_PORT))),
    )


app = create_app()
