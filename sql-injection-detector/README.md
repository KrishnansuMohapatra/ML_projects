# 🛡️ SQL Injection Detector

A machine learning-based application that detects whether a given text/query is likely to be a **SQL Injection** or a **Normal Input**.

The project uses **TF-IDF feature extraction** with **Logistic Regression** and provides a simple **Streamlit web interface** for real-time prediction.

---

## 🚀 Features

* Detects SQL Injection patterns using Machine Learning
* Character-level and word-level TF-IDF features
* Logistic Regression classifier
* 99%+ accuracy on the held-out test dataset
* Adversarial testing for common SQL-like normal sentences
* Confidence score for predictions
* Simple Streamlit UI
* Lightweight and easy to run locally

---

## 🧠 How It Works

The system follows this pipeline:

```text
User Input
    ↓
Text Preprocessing
    ↓
TF-IDF Feature Extraction
    ↓
Logistic Regression
    ↓
Prediction
    ↓
Normal / SQL Injection
```

### 1. Dataset

The project uses the **SQLiV2 dataset** containing SQL Injection and normal examples.

The dataset contains two important columns:

```text
Sentence
Label
```

Where:

```text
0 → Normal
1 → SQL Injection
```

---

### 2. TF-IDF

TF-IDF converts text into numerical features that can be understood by the machine learning model.

The project uses:

* Character-level TF-IDF
* Word-level TF-IDF

Character features are particularly useful for detecting patterns involving:

```text
'
--
OR
=
UNION
SELECT
```

and other unusual character combinations.

---

### 3. Machine Learning Model

The classifier used is:

**Logistic Regression**

It predicts one of two classes:

```text
0 → Normal
1 → SQL Injection
```

---

## 📊 Model Performance

The final V3 model was evaluated on a held-out test set that was not used for training.

### Test Results

```text
Test samples:       6,740
Accuracy:            99.7033%
False Positives:    1
False Negatives:    19
```

### Confusion Matrix

```text
                 Predicted
               Normal  SQLi

Actual Normal   4459     1
Actual SQLi       19  2261
```

The model achieved approximately **99.17% recall for SQL Injection** on the held-out test set.

> Note: Dataset-based performance does not guarantee the same performance against real-world SQL Injection attacks.

---

## 🧪 Adversarial Testing

The model was also tested against manually selected examples designed to contain SQL-like words while still being normal English.

Examples:

```text
Please select a product
Select your preferred option
Union membership benefits
Update my profile
Delete my account
Select a country
```

The model was also tested against common SQL Injection patterns:

```text
admin' OR '1'='1
' OR 1=1 --
' AND 1=2 --
admin'--
' UNION SELECT NULL --
' OR 'a'='a' --
' AND SLEEP(5) --
' OR EXISTS(SELECT 1) --
```

The V3 model achieved:

```text
Adversarial Test Accuracy: 100%
```

on this manually created 18-example test set.

---

## 🌐 REST API (FastAPI)

The project ships a production-shaped REST service in `api/` alongside the
Streamlit UI. The training scripts in `src/` stay the source of truth for the
model; the API only loads a saved artifact and exposes it over HTTP.

### Run it

```bash
uv sync                              # installs fastapi, uvicorn, python-multipart
uv run uvicorn api.main:app --reload # http://localhost:8000
```

| URL | What |
| --- | --- |
| `http://localhost:8000/docs` | Swagger UI (interactive) |
| `http://localhost:8000/redoc` | ReDoc |
| `http://localhost:8000/openapi.json` | Machine-readable schema |

Docs are disabled automatically when `SQLI_ENVIRONMENT=production`.

### Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/api/v1/predict` | Classify one input |
| `POST` | `/api/v1/predict/batch` | Classify up to `SQLI_MAX_BATCH_SIZE` inputs |
| `GET` | `/api/v1/predict/{query}` | Quick check from a shell/URL |
| `GET` | `/api/v1/predict/{query}/explain` | Heuristic breakdown only |
| `POST` | `/api/v1/files/analyze` | Upload CSV/TXT → JSON, CSV or TXT report |
| `POST` | `/api/v1/files/report` | Upload CSV/TXT → plain-text report |
| `GET` | `/health`, `/ready` | Liveness / readiness probes |
| `GET` | `/api/v1/model` | Model version, threshold, limits |
| `POST` | `/api/v1/admin/reload` | Hot-reload the artifact (guarded) |

### Example: single prediction

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H 'Content-Type: application/json' \
  -d '{"query": "%27%20OR%201%3D1--", "explain": "full"}'
```

```json
{
  "model_name": "sql-injection-detector-v4",
  "model_version": "V4",
  "threshold": 0.5,
  "processing_time_ms": 2.45,
  "result": {
    "query": "%27%20OR%201%3D1--",
    "normalized_query": "' OR 1=1--",
    "verdict": "suspicious",
    "is_injection": true,
    "label": "SQL Injection",
    "injection_probability": 0.9903,
    "confidence": 0.9903,
    "risk_level": "high",
    "signals": [
      {"id": "boolean_tautology", "weight": 0.9, "matched_text": "OR 1=1", "...": "..."},
      {"id": "encoded_payload", "weight": 0.4, "matched_text": "%27", "...": "..."}
    ]
  }
}
```

### Example: batch

```bash
curl -X POST http://localhost:8000/api/v1/predict/batch \
  -H 'Content-Type: application/json' \
  -d '{"queries": ["'"'"' UNION SELECT NULL --", "Please select a product"]}'
```

Returns `count`, `flagged_count` and a `results` list that preserves input
order.

### Example: file analysis

```bash
# JSON results + summary
curl -X POST http://localhost:8000/api/v1/files/analyze -F "file=@queries.csv"

# downloadable results CSV
curl -X POST "http://localhost:8000/api/v1/files/analyze?format=csv" \
     -F "file=@queries.csv" -o results.csv

# downloadable plain-text report
curl -X POST "http://localhost:8000/api/v1/files/analyze?format=txt" \
     -F "file=@queries.csv" -o report.txt
```

The CSV column is auto-detected from `sentence`/`query`/`sql`/`text`/
`input`/`payload`, falling back to the first column. Override with
`?column=my_column`.

### Design notes

* **One normalisation path.** The API applies the *same* URL-decode +
  whitespace transform used at training time (`api/ml/preprocessing.py`).
  `src/train_v4.py` and `src/app.py` now import that same function, so the
  Streamlit UI no longer scores encoded payloads on out-of-distribution text.
* **Artifact-agnostic loader.** `api/ml/registry.py` accepts all three shapes
  the project has produced over time (bare pipeline, pipeline with a joined
  `FeatureUnion`, and the V4 dict) behind a single `predict()` call.
* **Explainable, not opaque.** Every response can carry heuristic *signals*
  (`boolean_tautology`, `union_select`, `time_based`, …) so a flagged input
  comes with a reason. Signals are advisory — they never override the model.
* **Threshold is configuration, not code.** `SQLI_DECISION_THRESHOLD` moves the
  flag boundary; `risk_level` bands are computed relative to it.
* **Fails loudly, degrades gracefully.** A missing artifact does not crash
  start-up: `/health` reports `degraded`, `/ready` returns 503, and `/predict`
  returns 503 with the expected path.
* **Validation at the edge.** Pydantic enforces input shapes (`extra="forbid"`),
  length/count limits are checked before the model is touched, and every error
  uses one JSON shape: `{error, message, request_id}`.
* **Observability.** `X-Request-ID` (echoed or generated) and
  `X-Process-Time-Ms` on every response; slow requests are logged.

### Security

Off by default for local use, enabled by configuration:

```bash
# require an X-API-Key header (comma-separated list)
export SQLI_API_KEYS="key-one,key-two"

# 60 requests/minute per key (in-memory sliding window)
export SQLI_RATE_LIMIT_PER_MINUTE=60

# restrict browser origins
export SQLI_CORS_ORIGINS="https://app.example.com"
```

Keys are compared with `hmac.compare_digest`. Health/readiness probes stay open
so orchestrators can probe without credentials. The in-memory limiter is
per-process — swap it for Redis if you run multiple workers or replicas.

The API **never executes the inputs it receives**; it only classifies text.

### Configuration

All settings are environment variables prefixed with `SQLI_` (see
`.env.example`). Highlights:

| Variable | Default | Meaning |
| --- | --- | --- |
| `SQLI_MODEL_PATH` | `models/sql_injection_model_v4.pkl` | Artifact to serve (relative → project root) |
| `SQLI_DECISION_THRESHOLD` | `0.5` | Probability at/above which an input is flagged |
| `SQLI_MAX_QUERY_LENGTH` | `10000` | Per-input character cap |
| `SQLI_MAX_BATCH_SIZE` | `500` | Batch endpoint cap |
| `SQLI_MAX_FILE_SIZE_MB` | `10` | Upload cap |
| `SQLI_MAX_FILE_ROWS` | `5000` | Rows analysed per upload |
| `SQLI_ENVIRONMENT` | `development` | `production` hides `/docs` |
| `SQLI_API_KEYS` | *(empty)* | Enables API-key auth |
| `SQLI_RATE_LIMIT_PER_MINUTE` | `0` | 0 disables rate limiting |
| `SQLI_ALLOW_MODEL_RELOAD` | `false` | Enables `/admin/reload` |

### Quick check without the docs UI

```bash
uv run python scripts/smoke_test.py                       # canned examples
uv run python scripts/smoke_test.py --file queries.csv    # analyse a file
uv run python scripts/smoke_test.py --api-key "$KEY"      # authenticated
```

### Docker

```bash
uv run python scripts/make_dev_dataset.py && uv run python scripts/train_dev_model.py
docker build -t sql-injection-detector-api .
docker run --rm -p 8000:8000 sql-injection-detector-api
```

The image runs as a non-root user, drops into `SQLI_ENVIRONMENT=production`
(docs off) and ships a `HEALTHCHECK` against `/health`.

---

## 🖥️ Streamlit Application

The project includes a simple web interface.

Run:

```bash
uv run streamlit run src/app.py
```

The application allows the user to enter text and receive a prediction:

```text
🚨 SQL Injection Detected
```

or:

```text
✅ Normal Input
```

along with the model's confidence.

---

## 📁 Project Structure

```text
sql-injection-detector/
│
├── data/
│   └── sqliv2.csv
│
├── models/
│   ├── sql_injection_model.pkl
│   ├── sql_injection_model_v2.pkl
│   ├── sql_injection_model_v3.pkl
│   └── sql_injection_model_v4.pkl
│
├── api/                     # FastAPI service layer
│   ├── main.py              # app factory, lifespan, middleware
│   ├── config.py            # SQLI_* settings (pydantic-settings)
│   ├── schemas.py           # request/response contracts
│   ├── services.py          # DetectionService (batching, reports, risk)
│   ├── dependencies.py      # auth, rate limiting, limit guards
│   ├── errors.py            # one JSON error shape
│   ├── ml/
│   │   ├── preprocessing.py # shared normalisation (training + inference)
│   │   ├── heuristics.py    # explainable signal rules
│   │   └── registry.py      # artifact loader for V1–V4 formats
│   └── routers/
│       ├── predict.py       # /predict, /predict/batch, /explain
│       ├── files.py         # /files/analyze, /files/report
│       ├── health.py        # /health, /ready, /model
│       └── admin.py         # /admin/reload
│
├── src/
│   ├── train.py
│   ├── train_v2.py
│   ├── train_v3.py
│   ├── train_v4.py
│   ├── predict.py
│   ├── adversarial_test.py
│   ├── evaluate_v3.py
│   └── app.py
│
├── scripts/
│   ├── make_dev_dataset.py  # synthetic dataset (no Kaggle login needed)
│   ├── train_dev_model.py   # trains the same V4 artifact shape
│   └── smoke_test.py        # CLI client for the API
│
├── tests/                   # pytest suite (82 tests)
├── .env.example
├── Dockerfile
├── Makefile
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/KrishnansuMohapatra/ML_projects
```

### 2. Install dependencies

This project uses `uv`.

If you already have `uv` installed:

```bash
uv sync
```

Or install the required packages:

```bash
uv add pandas scikit-learn joblib streamlit fastapi "uvicorn[standard]" \
       python-multipart pydantic-settings

# plus the test tooling
uv pip install pytest httpx
```

---

## ▶️ Run the Application

The API (the primary interface):

```bash
uv run uvicorn api.main:app --reload
# Swagger UI: http://localhost:8000/docs
```

The original Streamlit UI still works:

```bash
uv run streamlit run src/app.py
```

Both read the same artifact (`models/sql_injection_model_v4.pkl`).

### No dataset yet? Use the synthetic dev path

The Kaggle dataset and trained artifacts are git-ignored, so a fresh clone has
neither. To get a working API without downloading anything:

```bash
uv run python scripts/make_dev_dataset.py   # synthetic Sentence/Label CSV
uv run python scripts/train_dev_model.py    # writes the V4 artifact shape
uv run uvicorn api.main:app --reload
```

The synthetic model is for exercising the API, **not** for reporting accuracy —
use the real SQLiV2 dataset for that.

---

## 🔬 Run Model Training

To train the original model:

```bash
uv run src/train.py
```

V2:

```bash
uv run src/train_v2.py
```

V3:

```bash
uv run src/train_v3.py
```

V4 was created as an additional experiment:

```bash
uv run src/train_v4.py
```

V3 is currently retained as the preferred model because it provided the best overall combination of held-out performance and adversarial-test performance.

---

## 🧪 Run Evaluation

To evaluate V3 on the held-out test set:

```bash
uv run src/evaluate_v3.py
```

To run the adversarial test:

```bash
uv run src/adversarial_test.py
```

---

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **Scikit-learn**
* **TF-IDF**
* **Logistic Regression**
* **Joblib**
* **FastAPI** + **Pydantic v2** + **Uvicorn** — REST API
* **Pytest** + **HTTPX** — test suite
* **Streamlit** — original UI
* **uv**
* **Docker**

---

## 📚 What I Learned

Through this project, I learned:

* Text preprocessing
* Feature extraction with TF-IDF
* Character vs. word n-grams
* Logistic Regression for text classification
* Train/test splitting
* Model evaluation
* Confusion matrices
* Precision, recall and F1-score
* False positives and false negatives
* Adversarial testing
* Hard-negative examples
* Building a machine learning web application with Streamlit
* Designing a REST API with FastAPI (routers, dependencies, schemas)
* Request validation and consistent error contracts with Pydantic v2
* Keeping training-time and inference-time preprocessing identical
* Configuration management with pydantic-settings
* API-key auth, constant-time comparison and rate limiting basics
* Health/readiness probes and structured request logging
* Testing an API with pytest and FastAPI's TestClient

---

## ⚠️ Disclaimer

This project is intended for **educational and research purposes**.

It is a machine learning classifier and should not be considered a complete replacement for production-grade SQL Injection protection.

Production applications should also use security practices such as:

* Parameterized queries
* Prepared statements
* Input validation
* Proper database permissions
* Secure application architecture

---

## 👨‍💻 Author

**Krishnansu Mohapatra**

Machine Learning / AI / Python enthusiast

---

## ⭐ Future Improvements

Possible future improvements include:

* Testing on additional SQL Injection datasets
* Better handling of obfuscated SQL Injection
* Testing against real-world traffic
* Comparing additional ML models
* Adding explainable predictions
* Deploying the API (container image is provided)
* Redis-backed rate limiting for multi-worker deployments
* Model versioning and A/B comparison between artifacts
* ~~Adding an API for programmatic detection~~ — **done** (`api/`)

---



All credit for the original dataset belongs to its respective creator. The model training, preprocessing, evaluation, and application code in this repository represent my own project work, except where otherwise acknowledged.
## ⭐ If You Like This Project

If you found this project useful, consider giving the repository a ⭐ on GitHub.
