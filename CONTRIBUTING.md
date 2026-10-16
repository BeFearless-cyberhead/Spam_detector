# Contributing

Thanks for taking a look at this project. It is a portfolio-scale codebase, so the process below is intentionally light.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements-dev.txt
pre-commit install
```

## Development loop

```bash
make data train    # download the UCI dataset and retrain (skip if models/ already has the demo artifact)
make check         # ruff + pytest, the same gate CI runs
make ui            # streamlit run app.py
make api           # uvicorn api:app --reload --port 8000
```

## Quality gates

Every change must pass:

```bash
ruff check .
ruff format --check .
pytest -q
```

- Line length is 88 characters; formatting is enforced by `ruff format` (not hand-formatted).
- Public functions, classes, and modules need Google-style docstrings (`ruff` `D` rules).
- New behaviour needs a test. Tests must not depend on the downloaded dataset — build fixtures with `tmp_path`, or use the committed `samples/sample_messages.tsv`. Tests that need the trained artifact are skipped when `models/spam_model.joblib` is absent.
- Never edit `models/metrics.json` by hand; regenerate it with `python train.py`.

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
- Whether a retrain (`python train.py`) changes the behaviour — this distinguishes a stale artifact from a code bug.
