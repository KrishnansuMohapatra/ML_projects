from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "sqliv2.csv"
MODEL_PATH = BASE_DIR / "models" / "sql_injection_model_v3.pkl"


# --------------------------------------------------
# Load and clean dataset
# --------------------------------------------------

print("Loading SQLiV2 dataset...")

df = pd.read_csv(DATA_PATH, encoding="utf-16")

df = df[["Sentence", "Label"]].copy()

df["Sentence"] = df["Sentence"].fillna("").astype(str).str.strip()
df["Label"] = pd.to_numeric(df["Label"], errors="coerce")

df = df.dropna(subset=["Label"])
df = df[df["Label"].isin([0, 1])]
df = df[df["Sentence"] != ""]
df = df.drop_duplicates(subset=["Sentence"])

print(f"Cleaned dataset shape: {df.shape}")


# --------------------------------------------------
# Prepare X and y
# --------------------------------------------------

X = df["Sentence"]
y = df["Label"]


# --------------------------------------------------
# SAME TEST SPLIT AS TRAINING
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


# --------------------------------------------------
# Load V3
# --------------------------------------------------

print("\nLoading V3 model...")

model = joblib.load(MODEL_PATH)


# --------------------------------------------------
# Predict ONLY unseen test data
# --------------------------------------------------

print("\nEvaluating V3 on unseen test data...")

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"\nAccuracy: {accuracy:.4%}")


# --------------------------------------------------
# Classification report
# --------------------------------------------------

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        predictions,
        target_names=["Normal", "SQL Injection"],
    )
)


# --------------------------------------------------
# Confusion matrix
# --------------------------------------------------

print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

cm = confusion_matrix(y_test, predictions)

print(cm)


# --------------------------------------------------
# Build error dataframe
# --------------------------------------------------

results = pd.DataFrame(
    {
        "input": X_test.values,
        "expected": y_test.values,
        "predicted": predictions,
        "sqli_probability": probabilities[:, 1],
    }
)

errors = results[results["expected"] != results["predicted"]]


# --------------------------------------------------
# False Positives
# --------------------------------------------------

false_positives = errors[
    (errors["expected"] == 0) &
    (errors["predicted"] == 1)
]

print("\n" + "=" * 70)
print("FALSE POSITIVES")
print("=" * 70)

print(f"Count: {len(false_positives)}")

for _, row in false_positives.iterrows():

    print(f"\nInput: {row['input']}")
    print("Expected: Normal")
    print("Predicted: SQL Injection")
    print(f"SQLi Probability: {row['sqli_probability']:.2%}")


# --------------------------------------------------
# False Negatives
# --------------------------------------------------

false_negatives = errors[
    (errors["expected"] == 1) &
    (errors["predicted"] == 0)
]

print("\n" + "=" * 70)
print("FALSE NEGATIVES")
print("=" * 70)

print(f"Count: {len(false_negatives)}")

for _, row in false_negatives.iterrows():

    print(f"\nInput: {row['input']}")
    print("Expected: SQL Injection")
    print("Predicted: Normal")
    print(f"SQLi Probability: {row['sqli_probability']:.2%}")


# --------------------------------------------------
# Summary
# --------------------------------------------------

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"Test samples: {len(y_test)}")
print(f"Correct predictions: {(predictions == y_test).sum()}")
print(f"Total errors: {len(errors)}")
print(f"False positives: {len(false_positives)}")
print(f"False negatives: {len(false_negatives)}")

print("\nEvaluation complete.")