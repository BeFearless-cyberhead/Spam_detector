"""End-to-end tests for the Streamlit UI using Streamlit's AppTest harness.

These render the app in-process (no browser required) and assert on the
resulting widget tree. They need the committed demo artifact, so they are
skipped when ``models/spam_model.joblib`` has been deleted.
"""

import pytest
from streamlit.testing.v1 import AppTest

from spam_detector import config
from spam_detector.cli import APP_PATH

pytestmark = pytest.mark.skipif(
    not config.model_path().exists(),
    reason="models/spam_model.joblib is missing; run `spam-detector-train` first",
)

SPAM_MESSAGE = "WINNER! You have been selected for a free cash prize, claim now!"
HAM_MESSAGE = "Are we still meeting at the library after class?"


@pytest.fixture
def app():
    """A freshly rendered app instance."""
    return AppTest.from_file(str(APP_PATH), default_timeout=60).run()


def test_app_renders_without_exceptions(app):
    assert not app.exception
    assert app.title[0].value == "📩 SMS Spam Detector"


def test_app_shows_model_card_metrics(app):
    assert not app.exception

    captions = " ".join(block.value for block in app.sidebar.caption)
    assert "SMSSpamCollection" in captions


def test_app_classifies_a_spam_message(app):
    app.text_area[0].set_value(SPAM_MESSAGE)
    app.button[0].click().run()

    assert not app.exception
    assert "spam" in " ".join(block.value for block in app.error).lower()


def test_app_classifies_a_ham_message(app):
    app.text_area[0].set_value(HAM_MESSAGE)
    app.button[0].click().run()

    assert not app.exception
    assert "legitimate" in " ".join(block.value for block in app.success).lower()


def test_app_warns_on_empty_message(app):
    app.text_area[0].set_value("   ")
    app.button[0].click().run()

    assert not app.exception
    assert "non-empty" in " ".join(block.value for block in app.warning)


def test_app_exposes_a_threshold_control(app):
    assert not app.exception
    assert app.slider[0].label == "Spam threshold"
    assert app.slider[0].value == 0.5


def test_app_shows_token_explanations(app):
    app.text_area[0].set_value(SPAM_MESSAGE)
    app.button[0].click().run()

    assert not app.exception
    assert any("token" in frame.value.columns for frame in app.dataframe)


def test_app_reports_input_drift(app):
    assert not app.exception
    assert any("Input drift" in expander.label for expander in app.sidebar.expander)
