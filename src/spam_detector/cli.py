"""Launcher helpers for the console entry points.

Installing the package exposes four commands:

* ``spam-detector-data``  -- download the dataset
* ``spam-detector-train`` -- train and evaluate the model
* ``spam-detector-api``   -- serve the REST API
* ``spam-detector-ui``    -- run the Streamlit demo
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

#: The Streamlit entry-point script shipped inside this package.
APP_PATH = Path(__file__).with_name("app.py")


def ui_command(app_path: Path = APP_PATH) -> list[str]:
    """Build the command that launches the Streamlit UI.

    Args:
        app_path: The Streamlit script to run.

    Returns:
        An argument list suitable for :func:`subprocess.call`.
    """
    return [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.headless=true",
    ]


def ui() -> None:
    """Launch the Streamlit UI (``spam-detector-ui``)."""
    sys.exit(subprocess.call(ui_command()))
