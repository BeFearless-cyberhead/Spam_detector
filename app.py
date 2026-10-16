"""Streamlit demo UI for the spam classifier.

Run locally::

    streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from spam_model import (
    MAX_MESSAGE_LENGTH,
    METRICS_FILENAME,
    MODEL_FILENAME,
    load_model,
    predict_spam,
    predict_spam_probability,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / MODEL_FILENAME
METRICS_PATH = BASE_DIR / "models" / METRICS_FILENAME


@st.cache_resource
def load_cached_model(path: str, modified_at: int):
    """Load the pipeline once per process and cache it.

    Args:
        path: Model artifact location.
        modified_at: Modification time, used as part of the cache key so a
            retrain invalidates the cache.

    Returns:
        The fitted scikit-learn pipeline.
    """
    return load_model(path)


@st.cache_data
def load_metrics(path: str, modified_at: int) -> dict | None:
    """Read the saved evaluation metrics, if they exist.

    Args:
        path: Metrics file location.
        modified_at: Modification time, used as part of the cache key.

    Returns:
        The parsed metrics, or ``None`` when the file is missing or corrupt.
    """
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def render_model_card(metrics: dict) -> None:
    """Render the model's held-out metrics in the sidebar.

    Args:
        metrics: Parsed ``metrics.json`` contents.
    """
    report = metrics.get("classification_report", {})
    spam = report.get("spam", {})
    ham = report.get("ham", {})

    st.sidebar.subheader("Model card")
    st.sidebar.metric("Accuracy", f"{report.get('accuracy', 0):.1%}")

    columns = st.sidebar.columns(2)
    columns[0].metric("Spam precision", f"{spam.get('precision', 0):.1%}")
    columns[1].metric("Spam recall", f"{spam.get('recall', 0):.1%}")

    st.sidebar.caption(
        f"Ham F1 {ham.get('f1-score', 0):.2f} · Spam F1 {spam.get('f1-score', 0):.2f}"
    )

    training = metrics.get("training", {})

    if training:
        st.sidebar.caption(
            f"Trained on **{metrics.get('training_messages', 0):,}** messages "
            f"from `{training.get('dataset', 'n/a')}`"
        )
        st.sidebar.caption(
            f"scikit-learn {training.get('scikit_learn_version', 'n/a')} · "
            f"{training.get('trained_at_utc', 'n/a')}"
        )

    with st.sidebar.expander("Confusion matrix (rows = actual)"):
        matrix = metrics.get("confusion_matrix")

        if matrix and len(matrix) == 2 and all(len(row) == 2 for row in matrix):
            st.dataframe(
                [
                    {
                        "actual": "ham",
                        "pred. ham": matrix[0][0],
                        "pred. spam": matrix[0][1],
                    },
                    {
                        "actual": "spam",
                        "pred. ham": matrix[1][0],
                        "pred. spam": matrix[1][1],
                    },
                ],
                hide_index=True,
            )


def main() -> None:
    """Render the Streamlit application."""
    st.set_page_config(
        page_title="SMS Spam Detector",
        page_icon="📩",
        layout="centered",
    )

    st.title("📩 SMS Spam Detector")
    st.write("Is this message spam?")
    st.caption("Yes = predicted spam · No = predicted legitimate message")

    metrics = None

    if METRICS_PATH.exists():
        metrics = load_metrics(str(METRICS_PATH), METRICS_PATH.stat().st_mtime_ns)

    if metrics:
        render_model_card(metrics)

    if not MODEL_PATH.exists():
        st.warning("Train the model before using the application.")
        st.code(
            "python download_data.py\npython train.py",
            language="bash",
        )
        st.stop()

    try:
        model = load_cached_model(
            str(MODEL_PATH),
            MODEL_PATH.stat().st_mtime_ns,
        )
    except Exception:  # noqa: BLE001 - surface any load failure as a friendly UI message
        st.error(
            "The model could not be loaded. "
            "Retrain it in this environment with: python train.py"
        )
        st.stop()

    with st.form("message_form"):
        message = st.text_area(
            "Message",
            placeholder="Enter an SMS message...",
            height=160,
            max_chars=MAX_MESSAGE_LENGTH,
        )
        submitted = st.form_submit_button("Check message")

    if submitted:
        try:
            is_spam = predict_spam(model, message)
            probability = predict_spam_probability(model, message)
        except (TypeError, ValueError) as exc:
            st.warning(str(exc))
        except Exception:  # noqa: BLE001 - surface any prediction error as a friendly UI message
            st.error("Prediction failed. Check the model and retrain if necessary.")
        else:
            if is_spam:
                st.error(f"### Yes — spam\nConfidence: {probability:.1%}")
            else:
                st.success(
                    f"### No — looks legitimate\nSpam probability: {probability:.1%}"
                )

            st.progress(
                min(max(probability, 0.0), 1.0),
                text=f"Spam probability: {probability:.1%}",
            )

    st.divider()
    st.caption(
        "Educational demo. Predictions can be wrong; "
        "do not automatically delete messages based on this result."
    )


if __name__ == "__main__":
    main()
