from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "sqliv2.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "sql_injection_model.pkl"


# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

print("Loading SQLiV2 dataset...")

df = pd.read_csv(
    DATA_PATH,
    encoding="utf-16",

)

print(f"Original dataset shape: {df.shape}")

print("\nColumns:")
print(df.columns.tolist())


# --------------------------------------------------
# 3. Identify text and label columns
# --------------------------------------------------

# SQLiV2 normally uses Sentence and Label.
# This also handles capitalization differences.

column_map = {column.lower(): column for column in df.columns}

text_column = column_map.get("sentence")
label_column = column_map.get("label")

if text_column is None or label_column is None:
    raise ValueError(
        "Could not find 'Sentence' and 'Label' columns."
    )

print(f"\nText column: {text_column}")
print(f"Label column: {label_column}")


# --------------------------------------------------
# 4. Keep only required columns
# --------------------------------------------------

df = df[[text_column, label_column]].copy()

df.columns = ["input", "label"]


# --------------------------------------------------
# 5. Clean dataset
# --------------------------------------------------

print("\nCleaning dataset...")

# Remove missing values
df = df.dropna(subset=["input", "label"])

# Convert input to string
df["input"] = df["input"].astype(str).str.strip()

# Convert labels to numeric
df["label"] = pd.to_numeric(df["label"], errors="coerce")

# Remove rows with invalid labels
df = df.dropna(subset=["label"])

# Keep only binary labels
df = df[df["label"].isin([0, 1])]

# Remove empty inputs
df = df[df["input"] != ""]

# Remove duplicate inputs
df = df.drop_duplicates(subset=["input"])

# Convert labels to integer
df["label"] = df["label"].astype(int)


print(f"Cleaned dataset shape: {df.shape}")

print("\nLabel distribution:")
print(df["label"].value_counts())


# --------------------------------------------------
# 6. X and y
# --------------------------------------------------

X = df["input"]
y = df["label"]

print("\nX = input")
print("y = label")


# --------------------------------------------------
# 7. Train/Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")


# --------------------------------------------------
# 8. Build ML Pipeline
# --------------------------------------------------

model = Pipeline(
    [
        (
            "tfidf",
            TfidfVectorizer(
                analyzer="char",
                ngram_range=(2, 5),
                min_df=2,
                max_features=100000,
                sublinear_tf=True,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
            ),
        ),
    ]
)


# --------------------------------------------------
# 9. Train
# --------------------------------------------------

print("\nTraining model...")

model.fit(X_train, y_train)

print("Training completed.")


# --------------------------------------------------
# 10. Evaluate
# --------------------------------------------------

print("\nEvaluating model...")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy:.4f}")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Normal",
            "SQL Injection",
        ],
    )
)


# --------------------------------------------------
# 11. Confusion Matrix
# --------------------------------------------------

print("Confusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# --------------------------------------------------
# 12. Save model
# --------------------------------------------------

MODEL_DIR.mkdir(exist_ok=True)

joblib.dump(model, MODEL_PATH)

print("\nModel saved successfully:")

print(MODEL_PATH)