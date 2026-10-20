
from pathlib import Path
from urllib.parse import unquote
import re

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# ==================================================
# PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "sqliv2.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "sql_injection_model_v4.pkl"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# TEXT NORMALIZATION
# Use this same function in app.py before prediction.
# ==================================================

def normalize_text(value):
    text = str(value).strip()

    # Decode URL encoding at most twice.
    # Examples: %27 -> ' and %20 -> space
    for _ in range(2):
        decoded = unquote(text)
        if decoded == text:
            break
        text = decoded

    # Normalize whitespace without removing SQL punctuation.
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ==================================================
# LOAD AND CLEAN DATASET
# ==================================================

print("Loading SQLiV2 dataset...")

df = pd.read_csv(DATA_PATH, encoding="utf-16")
df = df[["Sentence", "Label"]].copy()

df["Sentence"] = df["Sentence"].fillna("").astype(str)
df["Sentence"] = df["Sentence"].map(normalize_text)
df["Label"] = pd.to_numeric(df["Label"], errors="coerce")

df = df.dropna(subset=["Label"])
df = df[df["Label"].isin([0, 1])]
df = df[df["Sentence"] != ""]
df["Label"] = df["Label"].astype(int)

# Remove exact duplicates after normalization.
df = df.drop_duplicates(subset=["Sentence"]).reset_index(drop=True)

print(f"Cleaned dataset: {df.shape}")
print("\nLabel distribution:")
print(df["Label"].value_counts())


# ==================================================
# SPLIT ORIGINAL DATA BEFORE AUGMENTATION
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    df["Sentence"],
    df["Label"],
    test_size=0.20,
    random_state=42,
    stratify=df["Label"],
)

# Keep these original held-out rows untouched by augmentation.
test_sentences = set(X_test.tolist())
train_sentences = set(X_train.tolist())


# ==================================================
# TARGETED POSITIVE EXAMPLES
# Label 1 = SQL injection / suspicious SQL pattern
# ==================================================

hard_positive_examples = [
    "' OR '1'='1",
    "' OR 1=1 --",
    "' AND 1=2 --",
    "admin'--",
    "' UNION SELECT NULL --",
    "' OR EXISTS(SELECT 1) --",
    "' AND SLEEP(5) --",
    "' OR 'a'='a' --",
    "' UNION SELECT username, password FROM users --",
    "admin' /* comment */ --",
    "'; DROP TABLE users; --",
    "DROP TABLE users",
    "DROP TABLE users;",
    "DROP TABLE temp --",
    "INSERT INTO users VALUES ('admin')",
    "UPDATE users SET password='test'",
    "DELETE FROM users WHERE id=1",
    "TRUNCATE TABLE users",
    "CREATE USER name IDENTIFIED BY pass123",
    "GRANT CONNECT TO name",
    "GRANT RESOURCE TO name",
    "' UNION SELECT NULL,NULL --",
    "%27%20OR%201%3D1--",
]


# ==================================================
# TARGETED NEGATIVE EXAMPLES
# Label 0 = normal text / legitimate SQL
# ==================================================

hard_negative_examples = [
    "Please select a product",
    "Select a country",
    "Select your preferred option",
    "Select your language",
    "Select a category",
    "Select a date",
    "Union membership benefits",
    "Union membership application",
    "Update my profile",
    "Update your contact details",
    "Delete my account",
    "Delete this message",
    "Insert your address",
    "Insert your phone number",
    "Drop-down menu",
    "Administrator account",
    "Contact the administrator",
    "Create a new user account",
    "Grant permission to the user",
    "Remove the old account",
    "Update the application",
    "SELECT * FROM users WHERE id = 1",
    "SELECT name FROM products",
    "SELECT COUNT(*) FROM orders",
    "SELECT id, name FROM customers WHERE active = 1",
    "The SQL SELECT statement retrieves records",
    "I want to update my profile",
    "Can you delete my account?",
    "Please update the application",
]


# ==================================================
# AUGMENT TRAINING DATA ONLY
# ==================================================

extra_df = pd.DataFrame({
    "Sentence": (
        hard_positive_examples + hard_negative_examples
    ),
    "Label": (
        [1] * len(hard_positive_examples)
        + [0] * len(hard_negative_examples)
    ),
})

extra_df["Sentence"] = extra_df["Sentence"].map(normalize_text)

# If normalized variants collide, keep only one example.
extra_df = extra_df.drop_duplicates(subset=["Sentence"])

# Never add a custom example that appears in held-out test data.
extra_df = extra_df[
    ~extra_df["Sentence"].isin(test_sentences)
].copy()

# Avoid conflicting labels with existing training examples.
extra_df = extra_df[
    ~extra_df["Sentence"].isin(train_sentences)
].copy()

X_train = pd.concat(
    [X_train, extra_df["Sentence"]],
    ignore_index=True,
)

y_train = pd.concat(
    [y_train, extra_df["Label"]],
    ignore_index=True,
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Held-out samples: {len(X_test)}")
print(f"Additional training examples: {len(extra_df)}")


# ==================================================
# TF-IDF FEATURE EXTRACTION
# ==================================================

print("\nTraining V4...")
print("Character-level + word-level TF-IDF")

features = FeatureUnion([
    (
        "char",
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            min_df=1,
            max_features=100000,
            sublinear_tf=True,
        ),
    ),
    (
        "word",
        TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
            max_features=50000,
            sublinear_tf=True,
        ),
    ),
])

X_train_features = features.fit_transform(X_train)
X_test_features = features.transform(X_test)


# ==================================================
# TRAIN CLASSIFIER
# ==================================================

model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42,
)

model.fit(X_train_features, y_train)


# ==================================================
# EVALUATE ON HELD-OUT ORIGINAL DATA
# ==================================================

predictions = model.predict(X_test_features)

accuracy = accuracy_score(y_test, predictions)

print("\n" + "=" * 60)
print("V4 HELD-OUT EVALUATION")
print("=" * 60)

print(f"\nAccuracy: {accuracy:.4%}")

print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Normal", "SQL Injection"],
        zero_division=0,
    )
)

print("Confusion matrix:")
print(
    confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    )
)


# ==================================================
# SAVE MODEL
# ==================================================

saved_pipeline = {
    "features": features,
    "model": model,
    "version": "V4",
    "labels": {
        0: "Normal Input",
        1: "SQL Injection",
    },
    "normalization": "URL decode up to twice; normalize whitespace",
}

joblib.dump(saved_pipeline, MODEL_PATH)

print(f"\nModel saved to:\n{MODEL_PATH}")


# ==================================================
# SANITY TESTS
# These are diagnostic examples, not independent evaluation.
# ==================================================

sanity_tests = [
    ("DROP TABLE users;", 1),
    ("'; DROP TABLE users; --", 1),
    ("' OR '1'='1", 1),
    ("%27%20OR%201%3D1--", 1),
    ("Please select a product", 0),
    ("Update my profile", 0),
    ("SELECT * FROM users WHERE id = 1", 0),
]

print("\nSanity checks:")

for original_text, expected in sanity_tests:
    text = normalize_text(original_text)
    vector = features.transform([text])

    prediction = int(model.predict(vector)[0])
    probabilities = model.predict_proba(vector)[0]
    class_index = list(model.classes_).index(prediction)
    confidence = float(probabilities[class_index]) * 100

    status = "PASS" if prediction == expected else "CHECK"

    print(
        f"[{status}] {original_text!r} -> "
        f"{saved_pipeline['labels'][prediction]} "
        f"({confidence:.2f}%)"
    )

print("\nTraining complete.")
