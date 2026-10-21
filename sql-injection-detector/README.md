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
uv add pandas scikit-learn joblib streamlit
```

---

## ▶️ Run the Application

Run:

```bash
uv run streamlit run src/app.py
```

The Streamlit application will open in your browser.

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
* **Streamlit**
* **uv**

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
* Deploying the Streamlit application
* Adding an API for programmatic detection

---



All credit for the original dataset belongs to its respective creator. The model training, preprocessing, evaluation, and application code in this repository represent my own project work, except where otherwise acknowledged.
## ⭐ If You Like This Project

If you found this project useful, consider giving the repository a ⭐ on GitHub.
