"""Smoke tests for the Streamlit app."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parents[1] / "app.py")

PHISHING_ROW = (
    "-1, -1, 1, 1, -1, -1, 1, 1, 1, 1, "
    "0, 1, 1, 1, 1, 1, -1, -1, -1, -1, "
    "-1, -1, -1, 1, 1, 1, 1, 1, 1, 1"
)


@pytest.fixture(scope="module")
def app():
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert not at.exception, at.exception
    return at


def test_app_renders(app):
    assert len(app.tabs) == 3
    assert app.title[0].value
    assert len(app.button) == 1


def test_single_prediction_runs():
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.text_area[0].set_value(PHISHING_ROW)
    at.button[0].click().run()

    assert not at.exception, at.exception
    assert at.error or at.success
    assert len(at.metric) >= 2


def test_single_prediction_rejects_wrong_count():
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.text_area[0].set_value("1, 0, -1")
    at.button[0].click().run()

    assert not at.exception, at.exception
    assert any("30" in error.value for error in at.error)


def test_single_prediction_rejects_invalid_values():
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.text_area[0].set_value(", ".join(["7"] * 30))
    at.button[0].click().run()

    assert not at.exception, at.exception
    assert any("-1, 0, or 1" in error.value for error in at.error)


def test_batch_tab_lists_bundled_files(app):
    assert len(app.selectbox) == 1
    assert app.radio[0].options == [
        "📂 Use file from batch folder",
        "⬆️ Upload CSV",
    ]
