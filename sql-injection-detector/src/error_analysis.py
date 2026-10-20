from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "sqliv2.csv"
MODEL_PATH = BASE_DIR / "models" / "sql_injection_model_v3.pkl"


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

print("Loading dataset...")

df = pd.read_csv(DATA_PATH, encoding="utf-16")

df = df[["Sentence", "Label"]].copy()

df["Sentence"] = df["Sentence"].fillna("").astype(str).str.strip()
df["Label"] = pd.to_numeric(df["Label"], errors="coerce")

df = df.dropna(subset=["Label"])
df = df[df["Label"].isin([0, 1])]
df = df[df["Sentence"] != ""]
df = df.drop_duplicates(subset=["Sentence"])

print(f"Cleaned dataset: {df.shape}")


# --------------------------------------------------
# Load model
# --------------------------------------------------

print("\nLoading V3 model...")

model = joblib.load(MODEL_PATH)


# --------------------------------------------------
# Predictions
# --------------------------------------------------

X = df["Sentence"]
y = df["Label"]

predictions = model.predict(X)
probabilities = model.predict_proba(X)

df["prediction"] = predictions

# Probability of SQL Injection
df["sqli_probability"] = probabilities[:, 1]


# --------------------------------------------------
# Find errors
# --------------------------------------------------

errors = df[df["Label"] != df["prediction"]].copy()

false_positives = errors[
    (errors["Label"] == 0) &
    (errors["prediction"] == 1)
]

false_negatives = errors[
    (errors["Label"] == 1) &
    (errors["prediction"] == 0)
]


# --------------------------------------------------
# Results
# --------------------------------------------------

print("\n" + "=" * 70)
print("V3 ERROR ANALYSIS")
print("=" * 70)

print(f"\nTotal samples: {len(df)}")
print(f"Total errors: {len(errors)}")

print(f"\nFalse Positives: {len(false_positives)}")
print(f"False Negatives: {len(false_negatives)}")


# --------------------------------------------------
# False Positives
# --------------------------------------------------

print("\n" + "=" * 70)
print("FALSE POSITIVES")
print("(Normal input predicted as SQL Injection)")
print("=" * 70)

if len(false_positives) == 0:
    print("None 🎉")
else:
    for _, row in false_positives.iterrows():
        print(f"\nInput: {row['Sentence']}")
        print(f"Expected: Normal")
        print(f"Predicted: SQL Injection")
        print(f"SQLi Probability: {row['sqli_probability']:.2%}")


# --------------------------------------------------
# False Negatives
# --------------------------------------------------

print("\n" + "=" * 70)
print("FALSE NEGATIVES")
print("(SQL Injection predicted as Normal)")
print("=" * 70)

if len(false_negatives) == 0:
    print("None 🎉")
else:
    for _, row in false_negatives.iterrows():
        print(f"\nInput: {row['Sentence']}")
        print(f"Expected: SQL Injection")
        print(f"Predicted: Normal")
        print(f"SQLi Probability: {row['sqli_probability']:.2%}")


# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------

cm = confusion_matrix(y, predictions)

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(cm)

print("\nDone.")