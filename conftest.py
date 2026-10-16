"""Pytest bootstrap.

Having a conftest.py at the repository root makes pytest (in the default
"prepend" import mode) add this directory to sys.path, so tests can
``import spam_model`` without the project being installed.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
