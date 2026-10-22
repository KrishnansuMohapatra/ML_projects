# 🎣 Phishing Website Detection using Machine Learning

A machine learning project that classifies a website as **Phishing** or **Legitimate** using 30 encoded URL and page features.

The model is a **Random Forest Classifier** trained on the UCI *Phishing Websites* dataset. The project ships with a reusable Python package, a command-line interface, and a **Streamlit** web app for single and batch predictions.

> ⚠️ **Educational project.** The model does not read a raw URL — it consumes pre-encoded feature values. It is not a replacement for browser security, URL reputation services, or secure web development practices.

---

## 📑 Table of Contents

- [Highlights](#-highlights)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Features](#-features)
- [Model](#-model)
- [Results](#-results)
- [Feature Importance](#-feature-importance)
- [Cross-Validation](#-cross-validation)
- [Dataset Quality & The Honest Number](#-dataset-quality--the-honest-number)
- [Error Analysis](#-error-analysis)
- [Streamlit App](#-streamlit-app)
- [Batch CSV Format](#-batch-csv-format)
- [CLI Reference](#-cli-reference)
- [Security Note](#-security-note)
- [Limitations](#-limitations)
- [Future Improvements](#-future-improvements)
- [What This Project Taught Me](#-what-this-project-taught-me)
- [License](#-license)

---

## ✨ Highlights

| | |
|---|---|
| **Model** | `RandomForestClassifier(n_estimators=100, random_state=42)` |
| **Dataset** | UCI Phishing Websites — 11,055 rows × 30 features |
| **Holdout accuracy** | 97.42% (optimistic) |
| **Strict accuracy** | 96.86% (duplicate-aware, the number to trust) |
| **Strict 5-fold CV** | 96.42% ± 0.33 pp |
| **Interface** | Streamlit web app + `phishing-detector` CLI |
| **Tests** | 30 pytest tests, wired into GitHub Actions |
| **Tooling** | `uv`, `src/` layout, `pyproject.toml`, GitHub Actions CI |

---

## 🚀 Quick Start

**Requirements:** Python 3.11+ and [`uv`](https://docs.astral.sh/uv/).

```bash
# 1. Clone the repository
git clone https://github.com/KrishnansuMohapatra/ML_projects.git
cd ML_projects/Phising_website_detection

# 2. Create the environment and install everything
uv sync

# 3. Run the web app
uv run streamlit run app.py
```

Streamlit opens at `http://localhost:8501`.

The trained model ships with the repository at `models/phishing_model.pkl`, so the app works immediately — no training step required.

### Retrain the model

```bash
uv run phishing-detector train
```

This prints the full evaluation report and rewrites `models/phishing_model.pkl`.

### Run the evaluation only

```bash
uv run phishing-detector evaluate
```

### Run the tests

```bash
uv run pytest
```

The suite covers dataset integrity, the strict evaluation logic, prediction
guards, and a Streamlit smoke test. GitHub Actions runs it on every push.

---

## 📁 Project Structure

```text
ML_projects/
│
├── README.md                          # Root index of all projects
│
└── Phising_website_detection/         # ← this project
    │
    ├── app.py                         # Streamlit web application
    ├── pyproject.toml                 # Project metadata + dependencies
    ├── uv.lock                        # Locked dependency versions
    ├── .python-version                # Pinned Python version (3.11)
    ├── LICENSE                        # MIT License
    │
    ├── src/
    │   └── phishing_detection/
    │       ├── __init__.py            # Public API
    │       ├── __main__.py            # python -m phishing_detection
    │       ├── config.py              # Paths, feature list, constants
    │       ├── data.py                # Dataset loading + validation
    │       ├── evaluate.py            # Metrics, strict eval, error analysis
    │       ├── predict.py             # Model loading + prediction
    │       ├── train.py               # Training pipeline
    │       └── cli.py                 # Command-line interface
    │
    ├── tests/                         # pytest suite (30 tests)
    │   ├── conftest.py
    │   ├── test_app.py                # Streamlit smoke tests
    │   ├── test_data.py
    │   ├── test_evaluate.py
    │   └── test_predict.py
    │
    ├── data/
    │   └── phishing_data.csv          # UCI Phishing Websites dataset
    │
    ├── models/
    │   └── phishing_model.pkl         # Trained Random Forest (joblib)
    │
    └── batch/                         # Sample CSVs for the batch tab
        ├── batch_legitimate_10.csv
        ├── batch_phishing_10.csv
        ├── batch_mixed_10.csv
        └── batch_mixed_50.csv
```

Automated checks for this project run in GitHub Actions from the repository root
(`.github/workflows/ci.yml`), scoped to the `Phising_website_detection/` folder.

---

## 📊 Dataset

The [UCI Phishing Websites](https://archive.ics.uci.edu/dataset/379/phishing+websites) dataset.

| Property | Value |
|---|---|
| Rows | 11,055 |
| Features | 30 |
| Target | `result` |
| Phishing (`1`) | 6,157 rows |
| Legitimate (`-1`) | 4,898 rows |

Most feature values are encoded as:

```text
-1, 0, 1
```

The target column encodes the label:

```text
-1 → Legitimate
 1 → Phishing
```

---

## 🔢 Features

The model consumes these 30 features, **in this exact order**:

```text
having_ip_address            url_length                 shortining_service
having_at_symbol             double_slash_redirecting    prefix_suffix
having_sub_domain            sslfinal_state             domain_registration_length
favicon                      port                       https_token
request_url                  url_of_anchor              links_in_tags
sfh                          submitting_to_email        abnormal_url
redirect                     on_mouseover               rightclick
popupwindow                  iframe                     age_of_domain
dnsrecord                    web_traffic                page_rank
google_index                 links_pointing_to_page     statistical_report
```

The canonical list lives in [`src/phishing_detection/config.py`](src/phishing_detection/config.py) and is imported by both the training pipeline and the web app, so the two can never drift apart.

---

## 🌲 Model

```python
RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
```

**Why Random Forest?**

- Handles non-linear relationships between features.
- Works naturally with the integer-encoded features.
- Needs no feature scaling (tree splits are scale-invariant).
- Produces feature importances out of the box.
- Robust and easy to interpret.

**No `StandardScaler` is used** — tree-based models do not need standardized features for this dataset.

---

## 📈 Results

Using `test_size=0.2`, `random_state=42`, and `stratify=y`:

```text
Accuracy: 97.42%
```

### Confusion matrix

```text
                predicted
              Legit  Phish
actual Legit    942      38
actual Phish     19    1212
```

```text
True Negatives  = 942    Legitimate, correctly flagged
False Positives = 38     Legitimate, wrongly called phishing
False Negatives = 19     Phishing, wrongly called legitimate  ← the costly error
True Positives  = 1212   Phishing, correctly caught
```

### Classification report

```text
              precision    recall  f1-score   support

  Legitimate     0.9802    0.9612    0.9706       980
    Phishing     0.9696    0.9846    0.9770      1231

    accuracy                         0.9742      2211
   macro avg           0.9749    0.9729    0.9738      2211
weighted avg           0.9743    0.9742    0.9742      2211
```

The phishing class scores approximately:

```text
Precision: 96.96%
Recall:    98.46%
F1-score:  97.70%
```

---

## 🔍 Feature Importance

The trained Random Forest ranks the features as follows:

```text
sslfinal_state                0.319667
url_of_anchor                 0.250049
web_traffic                   0.070803
having_sub_domain             0.069482
links_in_tags                 0.043887
prefix_suffix                 0.037888
sfh                           0.020226
links_pointing_to_page        0.019292
request_url                   0.017682
domain_registration_length    0.016681
age_of_domain                 0.016456
google_index                  0.013074
having_ip_address             0.012946
dnsrecord                     0.012769
page_rank                     0.012008
url_length                    0.008218
https_token                   0.006526
redirect                      0.005714
submitting_to_email           0.005458
having_at_symbol              0.005435
statistical_report            0.005113
shortining_service            0.004888
popupwindow                   0.004649
favicon                       0.004177
abnormal_url                  0.004102
double_slash_redirecting      0.003739
on_mouseover                  0.003208
iframe                        0.002332
port                          0.002281
rightclick                    0.001248
```

The two strongest features — `sslfinal_state` and `url_of_anchor` — account for roughly **57%** of the total importance.

> Feature importance shows which features the model *used*, not which features *cause* phishing.

---

## 🧪 Cross-Validation

### Standard 5-fold CV on the full dataset

```text
0.98371777
0.97919493
0.97602895
0.96065129
0.94255993
```

```text
Mean:        96.84%
Std dev:      1.51 percentage points
```

### 5-fold CV on cleaned unique patterns (strict)

```text
0.9590
0.9685
0.9659
0.9650
0.9624
```

```text
Mean:        96.42%
Std dev:      0.33 percentage points
```

The strict run is far more stable — removing duplicate and ambiguous rows cuts the fold-to-fold variance by more than half.

---

## ⚠️ Dataset Quality & The Honest Number

During analysis, the dataset turned out to contain **many duplicate feature vectors**.

With a plain random train/test split, a large share of test rows have an *identical* feature vector sitting in the training set. That inflates the score, so:

> **97.42% is not proof the model will hit 97.42% on completely unseen modern websites.**

On top of duplicates, some feature vectors carry **conflicting labels** — the same encoded pattern is labelled both phishing and legitimate. Those groups are inherently ambiguous.

The strict evaluation fixes this:

1. Group rows by their 30-feature vector.
2. Drop groups that contain more than one label.
3. Collapse remaining identical vectors to a single row.
4. Split, train, and evaluate.

```text
Rows before cleaning      : 11,055
Ambiguous groups dropped  :      64
Rows after cleaning       :   5,721
Strict holdout accuracy   :  96.86%
Strict 5-fold CV mean     :  96.42%
```

**The defensible claim for this model is a ~96–97% range on this dataset** — not the headline 97.42%.

---

## 🚨 Error Analysis

The test set contained **19 false negatives**:

```text
Actual:   Phishing
Predicted: Legitimate
```

In phishing detection this is the most expensive kind of mistake — a real attack site waved through as safe. Accuracy alone is therefore not a sufficient metric.

Features most skewed between *all phishing rows* and *false negatives*:

| Feature | All phishing | False negatives | Difference |
|---|---|---|---|
| `links_in_tags` | 0.052 | -0.632 | -0.684 |
| `prefix_suffix` | -0.524 | -1.000 | -0.476 |
| `age_of_domain` | 0.136 | -0.263 | -0.399 |
| `page_rank` | -0.402 | -0.789 | -0.387 |
| `sfh` | -0.456 | -0.842 | -0.386 |
| `url_of_anchor` | 0.371 | 0.000 | -0.371 |
| `sslfinal_state` | 0.842 | 0.474 | -0.369 |

False negatives tend to look *more legitimate* on the features the model relies on most — consistent with them being genuinely harder cases.

Metrics that matter for a security classifier:

- Precision
- Recall
- F1-score
- Confusion matrix
- False negatives (missed attacks)
- False positives (annoying false alarms)

---

## 🖥️ Streamlit App

```bash
uv run streamlit run app.py
```

The app has three tabs.

### 1. 🔍 Single Test

Paste exactly **30** values. Commas, spaces, newlines, and square brackets are all accepted.

```text
-1, -1, 1, 1, -1, -1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
-1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1
```

The app validates:

- Exactly 30 values were supplied
- Every value is `-1`, `0`, or `1`
- The values map to the correct feature order

It then shows the verdict and the probability split:

```text
🚨 Prediction: PHISHING
```

```text
✅ Prediction: LEGITIMATE
```

alongside `Phishing probability` / `Legitimate probability` metrics and an expandable view of the parsed input.

### 2. 📁 Batch CSV

- Pick any CSV from the `batch/` folder, or upload one.
- Rows predicted as phishing are flagged, sorted by probability, and highlighted in red.
- Summary metrics show total rows, phishing count, legitimate count, and phishing share.
- Download the flagged rows only, or the full prediction table, as CSV.

### 3. ℹ️ About

Explains the model, the dataset limitation, and — importantly — what the app does **not** do.

---

## 📄 Batch CSV Format

The CSV must contain the same 30 feature columns as the model.

- The `result` column is **optional** (it is ignored during prediction).
- All values must be `-1`, `0`, or `1`.
- Rows with missing or non-numeric values are rejected with the offending row numbers listed.

Output columns added by the app:

```text
row_number            (Excel-style row number, header = row 1)
prediction            (Phishing | Legitimate)
phishing_probability  (0.0 - 1.0, when the model supports predict_proba)
```

Sample files live in [`batch/`](batch/).

---

## 🖥️ CLI Reference

```bash
uv run phishing-detector <command> [options]
```

| Command | Description |
|---|---|
| `train` | Train the model, print the full report, save `models/phishing_model.pkl`. |
| `train --no-save` | Same, but do not write the model file. |
| `evaluate` | Duplicate-aware holdout evaluation + both cross-validations. |
| `predict <csv>` | Predict a CSV of features and print a summary. |
| `predict <csv> --output out.csv` | Also write the predictions to disk. |

The CLI is also available as a module:

```bash
uv run python -m phishing_detection evaluate
```

Use it as a library:

```python
from phishing_detection import load_dataset, train_model, evaluate_model

model, results = train_model()
print(results["accuracy"])
```

---

## 🔐 Security Note

This is an **educational** machine learning system.

It must not be treated as a replacement for established browser security, URL reputation services, sandboxing, or secure web development practices. Machine learning predictions can be wrong.

A real phishing defence layers multiple controls:

```text
URL reputation
+
Domain intelligence
+
Threat intelligence
+
Browser security
+
Machine learning
+
User awareness
```

Above all: websites should use secure authentication practices, and users should never enter sensitive information into a suspicious site.

---

## 🚧 Limitations

1. The dataset is an older benchmark dataset.
2. Features are encoded, not raw URLs.
3. The model does not analyse a URL string directly.
4. Duplicate feature vectors exist in the dataset.
5. Some feature combinations carry conflicting labels.
6. Real-world phishing sites continuously change.
7. Modern phishing techniques may not be represented.
8. Machine learning predictions are never guaranteed to be correct.

Treat this as an **educational and experimental** phishing detection system, not a production-grade cybersecurity solution.

---

## 🔮 Future Improvements

- URL-based feature extraction (the biggest gap)
- Character-level URL analysis and TF-IDF over URLs
- Gradient Boosting / XGBoost comparison
- Threshold optimisation for the false-negative trade-off
- Better handling of class imbalance
- SHAP-based explainability
- External threat-intelligence integration
- Domain-age and live DNS lookups
- API deployment and browser-extension integration
- Continuous evaluation against newer phishing datasets

These should be added when there is a genuine need — not to inflate complexity.

---

## 🧠 What This Project Taught Me

**Data handling** — loading with Pandas, separating features and target, checking dtypes, class distribution, and missing values.

**Machine learning** — train/test splitting, stratification, Random Forest, fitting, prediction, feature importance.

**Evaluation** — accuracy, precision, recall, F1-score, confusion matrix, false positives, false negatives, cross-validation.

**Model analysis** — duplicate detection, label-inconsistency analysis, class-wise feature comparison, error analysis, and stricter evaluation.

**Deployment** — persisting models with Joblib, loading them back, building a Streamlit app, single and batch prediction, CSV validation, downloadable results.

**Software engineering** — packaging a project into an installable module, a CLI, a testable evaluation layer, an automated test suite, and CI.

### The main lesson

> A high accuracy score does not automatically mean a model is ready for the real world.

A serious project asks:

```text
How was the data collected?
        ↓
Are there duplicates?
        ↓
Is there target leakage?
        ↓
Are labels reliable?
        ↓
How does the model perform on unseen data?
        ↓
What kinds of errors does it make?
        ↓
Can it be deployed safely?
```

This project is about understanding **how trustworthy that score actually is** — not just producing one.

---

## 🤝 Contributing

This project lives in the [`ML_projects`](https://github.com/KrishnansuMohapatra/ML_projects) monorepo, under the `Phising_website_detection/` folder.

To pick up the latest changes and work on it:

```bash
git clone https://github.com/KrishnansuMohapatra/ML_projects.git
cd ML_projects/Phising_website_detection
uv sync
```

Run the checks before opening a PR:

```bash
uv run pytest
uv run phishing-detector evaluate
```

Both run automatically in GitHub Actions for this folder on every push to `main`.

---

## 📄 License

Released under the [MIT License](LICENSE).

**Dataset:** UCI Phishing Websites — [Mohammadlou et al.](https://archive.ics.uci.edu/dataset/379/phishing+websites)

---

## 👨‍💻 Author

**Krishnansu Mohapatra**
MCA Graduate · Python · Machine Learning · AI · Data Analytics

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.63-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

**Build → Test → Analyze → Improve.**
