# Model Card — SMS Spam Detector

Following the structure proposed by Mitchell et al. (2019), *Model Cards for Model Reporting*.

## Model details

| Field | Value |
| :--- | :--- |
| Task | Binary text classification (SMS/email → spam or ham) |
| Architecture | `sklearn.pipeline.Pipeline` = `TfidfVectorizer` → `LogisticRegression` |
| Vectoriser | lowercase, `strip_accents="unicode"`, `ngram_range=(1, 2)`, `sublinear_tf=True`, `max_features=30_000` |
| Classifier | `LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)` |
| Artifact | `models/spam_model.joblib` (~1.4 MB) |
| Outputs | Hard label (`ham`/`spam`) and `predict_proba` spam confidence |
| Trained with | scikit-learn 1.7.2, Python 3.10.11 (see `models/metrics.json`) |
| Owner / contact | Ankit — see the repository issue tracker |

## Intended use

**Primary use:** educational and portfolio demonstration of an end-to-end NLP classification service, plus a reasonable baseline for English SMS spam filtering.

**Intended users:** developers evaluating the pipeline, recruiters/reviewers inspecting the code, and anyone prototyping a text-classification service.

**Out-of-scope / not intended for:**

- Automated deletion or blocking of messages without human review.
- Production filtering of high-stakes channels (banking OTPs, medical, legal) without retraining and monitoring.
- Languages other than English; the model has never seen multilingual text.
- Adversarial settings — spammers who know the model can evade it with token obfuscation.
- Any decision with legal, financial, or safety consequences.

## Training data

- **Source:** [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) (CC BY 4.0), downloaded at runtime by `spam_detector.data`.
- **Content:** 5,574 English SMS messages labelled `ham` (legitimate) or `spam`.
- **Preprocessing:** blank lines dropped, duplicates removed on message text, labels mapped to `{ham: 0, spam: 1}`. Rows where the same message carries conflicting labels raise an error.
- **Usable rows:** 5,160 unique messages (1,032 held out for testing).
- **Fingerprint:** `sha256 7d039a24a6083ed9ef0f806ebad56bbb976e3aeb8de05669173bfdc4996c239d`.

### Known dataset limitations

- Fixed vintage (messages collected around 2011–2012); spam vocabulary and campaign styles have moved on.
- SMS-only: shorter and more abbreviated than email or chat text.
- Predominantly English and UK-centric (many messages reference UK shortcodes and `£` amounts).
- Class imbalance: roughly 13% spam, and the spam class is topically narrow.

## Evaluation

Stratified 80/20 hold-out split (seed 42); 4,128 train / 1,032 test messages. No message appears in both splits.

| Class | Precision | Recall | F1 | Support |
| :--- | ---: | ---: | ---: | ---: |
| ham | 0.9934 | 0.9934 | 0.9934 | 904 |
| spam | 0.9531 | 0.9531 | 0.9531 | 128 |
| macro avg | 0.9732 | 0.9732 | 0.9732 | 1,032 |
| weighted avg | 0.9884 | 0.9884 | 0.9884 | 1,032 |
| **accuracy** | | | **0.9884** | 1,032 |

Confusion matrix (rows = actual, columns = predicted):

|  | pred. ham | pred. spam |
| :--- | ---: | ---: |
| **actual ham** | 898 | 6 |
| **actual spam** | 6 | 122 |

**Metric choice.** Precision and recall are balanced at the operating threshold (0.5, calibrated); a product that weights one over the other can move the threshold per request (`SPAM_THRESHOLD` env var or the API/CLI `--threshold` option) without retraining. Probabilities are sigmoid-calibrated — Brier score **0.0099**, log loss **0.0386** — so confidence values are honest.

**Training procedure.** Stratified 5-fold cross-validated search over word n-gram range, character n-grams (on), and regularisation strength `C` selected by spam F1, followed by sigmoid calibration. Length/vocabulary statistics are recorded for drift monitoring.

## Error analysis

Held-out results are concentrated in two failure modes:

1. **False negatives (6 messages)** — spam phrased in ordinary conversational English (no prize/urgent/free keywords).
2. **False positives (6 messages)** — legitimate messages that share spam vocabulary: promotions, prize/competition notices, delivery notifications with links, and messages full of numbers and capitalisation.

Because features are bag-of-ngrams, word order and negation are largely lost ("do not call this number" vs "call this number"), which is the main structural weakness.

## Fairness and bias considerations

- The dataset mixes personal SMS with corporate/marketing text; the model inherits those proportions and may under-serve dialects or informal spellings absent from the corpus.
- No demographic attributes are present, so subgroup performance cannot be measured. Metrics are reported only for the two label classes.
- The model should not be applied to populations or channels materially different from the training distribution without re-evaluation.

## Ethical and safety notes

- **No autonomous action.** The Streamlit UI states plainly that predictions can be wrong and that messages should not be deleted automatically.
- **Explainability.** Logistic-regression coefficients over TF-IDF features make the model broadly interpretable, and the pipeline is fully reproducible from `spam_detector.train`.
- **Security.** `joblib`/pickle artifacts execute code on load — only load models produced by this repository. The API caps message length at 10,000 characters and rejects non-string payloads via Pydantic.
- **No PII storage.** Predictions are scored in memory and never persisted; the only optional persistence is the `SPAM_FEEDBACK_LOG` JSONL supplied voluntarily via `/feedback`. The `SPAM_DRIFT_WINDOW` monitor sees message text transiently in memory only.

## Caveats and recommendations

- Retrain before any real deployment; the metrics describe a 2011-era snapshot.
- Re-evaluate on data from the actual target channel and vintage.
- Probabilities are calibrated with `CalibratedClassifierCV`; the `--no-calibrate` training flag restores the raw logistic regression if raw scores are preferred.
- `/drift` and the UI sidebar flag rising OOV rate and length shift as early signals that retraining is due, but the monitor is descriptive — a flagged status means investigate, not auto-retrain.
- Update `models/metrics.json` via `spam-detector-train` whenever the model changes; never hand-edit it.
