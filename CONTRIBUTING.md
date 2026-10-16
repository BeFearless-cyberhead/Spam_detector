# Contributing

Thanks for taking a look at this project. It is a portfolio-scale codebase, so the process below is intentionally light.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install --editable ".[all]"   # package + every extra, in editable mode
pre-commit install
```

The editable install is what makes `spam-detector-*` available on your PATH and lets the Streamlit app import the package.

## Development loop

```bash
make data train    # download the UCI dataset and retrain (skip if models/ already has the demo artifact)
make check         # ruff + pytest, the same gate CI runs
make ui            # launch the Streamlit UI
make api           # launch the API with autoreload
```

Equivalent without `make`:

```bash
python -m spam_detector.train
python -m streamlit run src/spam_detector/app.py
python -m uvicorn spam_detector.api:app --reload
```

## Quality gates

Every change must pass:

```bash
ruff check .
ruff format --check .
pytest -q
```

Coverage must stay at or above the `fail_under` threshold configured in `pyproject.toml` (90%):

```bash
pytest --cov --cov-report=term-missing
```

- Line length is 88 characters; formatting is enforced by `ruff format` (not hand-formatted).
- Public modules, functions, and classes need Google-style docstrings (`ruff` `D` rules).
- New behaviour needs a test. Tests must not touch the network or the downloaded dataset — build fixtures with `tmp_path`, use the committed `samples/sample_messages.tsv`, or monkeypatch the download helpers as `tests/test_data.py` does. Tests that need the trained artifact are skipped when `models/spam_model.joblib` is absent.
- Never edit `models/metrics.json` by hand; regenerate it with `spam-detector-train`.

### Where does new code go?

| Change | Module |
| :--- | :--- |
| Pipeline, validation, prediction | `src/spam_detector/model.py` |
| Paths, file names, environment variables | `src/spam_detector/config.py` |
| Training, splitting, metrics | `src/spam_detector/train.py` |
| Dataset download/extraction | `src/spam_detector/data.py` |
| HTTP interface | `src/spam_detector/api.py` |
| Streamlit interface | `src/spam_detector/app.py` |
| Console-script wrappers | `src/spam_detector/cli.py` |

Anything new must be re-exported from `src/spam_detector/__init__.py` if it belongs to the public API.

## Adding a dependency

1. Add the pin to `pyproject.toml` (`dependencies` or the relevant extra).
2. Mirror it in `requirements.txt` / `requirements-dev.txt` so both install paths stay in sync.
3. Update the tech-stack table in `README.md` if it changes the story of the project.

## Commits and pull requests

- Keep commits focused; the message should say *what changed and why* (imperative mood).
- One logical change per pull request. Describe the motivation, the approach, and how you verified it.
- If you change model behaviour or hyperparameters, include the before/after metrics from `metrics.json` in the PR description.

## Reporting bugs

Open an issue with:

- What you ran (exact command) and what you expected.
- The traceback or the full JSON error response.
- Output of `python -c "import sklearn, pandas, sys; print(sys.version, sklearn.__version__, pandas.__version__)"`.
- Whether a retrain (`spam-detector-train`) changes the behaviour — this distinguishes a stale artifact from a code bug.
