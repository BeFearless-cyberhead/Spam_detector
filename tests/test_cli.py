"""Tests for the console-script launchers."""

import sys
from pathlib import Path

from spam_detector import cli


def test_ui_command_runs_streamlit_on_the_packaged_app():
    command = cli.ui_command()

    assert command[0] == sys.executable
    assert command[1:4] == ["-m", "streamlit", "run"]
    assert Path(command[4]) == cli.APP_PATH
    assert "--server.headless=true" in command


def test_ui_command_targets_a_real_file():
    assert cli.APP_PATH.name == "app.py"
    assert cli.APP_PATH.exists()
    assert cli.APP_PATH.parent.name == "spam_detector"


def test_ui_command_honours_an_explicit_path(tmp_path):
    alternative = tmp_path / "app.py"

    assert cli.ui_command(alternative)[4] == str(alternative)
