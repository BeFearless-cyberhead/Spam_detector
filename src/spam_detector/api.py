"""FastAPI service exposing the spam classifier over HTTP.

Run locally::

    python -m uvicorn spam_detector.api:app --reload --port 8000
    # or, once the package is installed: spam-detector-api

Interactive docs are then available at http://127.0.0.1:8000/docs.

Configuration (environment variables):

* ``SPAM_MODEL_PATH`` -- path to the ``joblib`` artifact, default
  ``models/spam_model.joblib``.
* ``SPAM_THRESHOLD`` -- default spam probability threshold, default ``0.5``.
* ``SPAM_FEEDBACK_LOG`` -- JSONL file for ``/feedback`` reports, default
  ``data/feedback.jsonl``.
* ``SPAM_DRIFT_WINDOW`` -- recent messages kept for drift checks, default ``200``.
* ``API_HOST`` / ``API_PORT`` -- bind address for the ``spam-detector-api``
  entry point.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal

import uvicorn
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, StringConstraints

from spam_detector import config
from spam_detector.config import (
    METRICS_FILENAME,
)
from spam_detector.config import (
    model_path as default_model_path,
)
from spam_detector.drift import DriftMonitor, ReferenceStatistics
from spam_detector.model import (
    DEFAULT_TOP_TOKENS,
    LABELS,
    MAX_MESSAGE_LENGTH,
    explain_prediction,
    load_model,
    predict_spam,
    predict_spam_probability,
    validate_message,
)

# Containers must bind all interfaces, hence 0.0.0.0 as the default host.
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000

#: Upper bound on messages accepted by one batch request.
MAX_BATCH_MESSAGES = 100

Message = Annotated[
    str,
    StringConstraints(min_length=1, max_length=MAX_MESSAGE_LENGTH),
]
Label = Literal["spam", "ham"]


class PredictRequest(BaseModel):
    """A single message submitted for classification."""

    message: Message = Field(
        ...,
        description="Raw SMS/email text to classify.",
        examples=["Congratulations! You have won a free prize, claim now."],
    )
    threshold: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Override the spam probability threshold for this request.",
    )
    explain: bool = Field(
        False,
        description="Include the tokens that most influenced the verdict.",
    )


class TokenContribution(BaseModel):
    """One feature that pushed a prediction towards spam or ham."""

    token: str = Field(..., examples=["free"])
    weight: float = Field(..., examples=[0.4213])
    source: str = Field(..., description="Feature source: 'word' or 'char'.")
    direction: Label = Field(..., description="'spam' or 'ham'.")


class PredictResponse(BaseModel):
    """The classifier's verdict for one message."""

    label: Label = Field(..., description="Either 'spam' or 'ham'.", examples=["spam"])
    is_spam: bool = Field(..., description="Convenience boolean for the label.")
    spam_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Calibrated confidence that the message is spam.",
        examples=[0.93],
    )
    threshold: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability threshold that produced the label.",
    )
    explanations: list[TokenContribution] | None = Field(
        None,
        description="Top contributing tokens, present when `explain` is true.",
    )


class BatchPredictRequest(BaseModel):
    """Several messages classified in one round trip."""

    messages: list[Message] = Field(
        ...,
        min_length=1,
        max_length=MAX_BATCH_MESSAGES,
        description=f"Between 1 and {MAX_BATCH_MESSAGES} messages.",
    )
    threshold: float | None = Field(None, ge=0.0, le=1.0)
    explain: bool = False


class BatchPredictResponse(BaseModel):
    """One verdict per submitted message, in request order."""

    results: list[PredictResponse]
    messages_processed: int


class FeedbackRequest(BaseModel):
    """An opt-in report that a prediction was wrong."""

    message: Message = Field(..., description="The original message.")
    predicted_label: Label = Field(..., description="What the model returned.")
    correct_label: Label = Field(..., description="The label you believe is right.")
    spam_probability: float | None = Field(None, ge=0.0, le=1.0)
    note: str | None = Field(
        None,
        max_length=500,
        description="Optional short explanation, e.g. 'promotion from my bank'.",
    )


class FeedbackResponse(BaseModel):
    """Confirmation that the report was appended to the feedback log."""

    logged: bool
    feedback_file: str
    recorded_at_utc: str


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


def read_metrics(metrics_file: Path) -> dict[str, Any] | None:
    """Read a metrics file, returning ``None`` when it is missing or corrupt.

    Args:
        metrics_file: Location of the ``metrics.json`` written by the trainer.

    Returns:
        The parsed mapping, or ``None``.
    """
    try:
        return json.loads(metrics_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def create_app(
    model_path: str | Path | None = None,
    feedback_log: str | Path | None = None,
) -> FastAPI:
    """Build the FastAPI application.

    The model is loaded lazily on the first prediction so the service can start
    (and report an unhealthy status) even when no artifact has been trained yet.

    Args:
        model_path: Optional explicit model location; mainly used by tests.
        feedback_log: Optional explicit feedback-log location; mainly used by
            tests.

    Returns:
        A configured :class:`~fastapi.FastAPI` instance.
    """
    resolved_path = resolve_model_path(model_path)
    resolved_feedback = (
        Path(feedback_log) if feedback_log is not None else config.feedback_path()
    )

    default_threshold = config.spam_threshold()
    metrics_file = resolved_path.parent / METRICS_FILENAME
    reference = ReferenceStatistics.from_metrics(read_metrics(metrics_file))
    monitor = DriftMonitor(reference, window_size=config.drift_window())

    app = FastAPI(
        title="SMS Spam Detector API",
        version="2.0.0",
        summary="Calibrated TF-IDF + logistic-regression spam classifier.",
        description=(
            "REST wrapper around the scikit-learn pipeline used by the "
            "Streamlit demo. POST a message to `/predict` for a verdict, a "
            "calibrated confidence score, and optional token explanations. "
            "`/predict/batch` scores up to 100 messages at once, `/feedback` "
            "logs misclassifications, and `/drift` summarises the input "
            "distribution of recent traffic."
        ),
    )

    cache: dict[str, Any] = {"model": None}
    feedback_lock = threading.Lock()

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

    def score(
        message: str,
        threshold: float,
        explain: bool,
    ) -> PredictResponse:
        """Classify one validated message and record it for drift monitoring."""
        model = get_model()

        try:
            probability = predict_spam_probability(model, message)
            is_spam = predict_spam(model, message, threshold=threshold)
            explanations = (
                [
                    TokenContribution(**contribution)
                    for contribution in explain_prediction(
                        model,
                        message,
                        top_k=DEFAULT_TOP_TOKENS,
                    )
                ]
                if explain
                else None
            )
        except (TypeError, ValueError) as exc:
            # 422 = Unprocessable Content. The literal keeps compatibility with
            # Starlette versions where this status constant was renamed.
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        cleaned = validate_message(message)
        monitor.observe(cleaned)

        return PredictResponse(
            label=LABELS[int(is_spam)],
            is_spam=is_spam,
            spam_probability=round(probability, 6),
            threshold=threshold,
            explanations=explanations,
        )

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
        if not metrics_file.exists():
            return ModelInfoResponse(
                model_path=str(resolved_path),
                metrics_available=False,
            )

        try:
            metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
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

    @app.get("/drift", tags=["ops"])
    def drift() -> dict[str, Any]:
        """Summarise length and vocabulary drift over recent traffic."""
        return monitor.snapshot()

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
            The predicted label, an ``is_spam`` boolean, the spam probability,
            the threshold used, and explanations when requested.
        """
        threshold = (
            payload.threshold if payload.threshold is not None else default_threshold
        )
        return score(payload.message, threshold, payload.explain)

    @app.post(
        "/predict/batch",
        response_model=BatchPredictResponse,
        tags=["prediction"],
        responses={503: {"description": "Model artifact unavailable."}},
    )
    def predict_batch(payload: BatchPredictRequest) -> BatchPredictResponse:
        """Classify a batch of messages, preserving request order.

        Args:
            payload: Up to :data:`MAX_BATCH_MESSAGES` messages plus optional
                threshold and explanation flags.

        Returns:
            One :class:`PredictResponse` per message.
        """
        threshold = (
            payload.threshold if payload.threshold is not None else default_threshold
        )
        results = [
            score(message, threshold, payload.explain) for message in payload.messages
        ]
        return BatchPredictResponse(
            results=results,
            messages_processed=len(results),
        )

    @app.post(
        "/feedback",
        response_model=FeedbackResponse,
        tags=["feedback"],
        responses={500: {"description": "Feedback log could not be written."}},
    )
    def feedback(payload: FeedbackRequest) -> FeedbackResponse:
        """Append a misclassification report to the feedback log.

        Reports are opt-in and stored as JSON Lines on the local filesystem;
        nothing is sent anywhere else.

        Args:
            payload: The message, the model's label, and the correct label.

        Returns:
            Confirmation with the log location and UTC timestamp.
        """
        recorded_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        record = {
            "recorded_at_utc": recorded_at,
            "message": validate_message(payload.message),
            "predicted_label": payload.predicted_label,
            "correct_label": payload.correct_label,
            "spam_probability": payload.spam_probability,
            "note": payload.note,
        }

        try:
            resolved_feedback.parent.mkdir(parents=True, exist_ok=True)

            with feedback_lock, resolved_feedback.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Feedback could not be recorded: {exc}",
            ) from exc

        return FeedbackResponse(
            logged=True,
            feedback_file=str(resolved_feedback),
            recorded_at_utc=recorded_at,
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
