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
from sklearn.pipeline import FeatureUnion, Pipeline


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "sqliv2.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "sql_injection_model_v2.pkl"


# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

print("Loading SQLiV2 dataset...")

df = pd.read_csv(
    DATA_PATH,
    encoding="utf-16"
)

print(f"Original dataset shape: {df.shape}")

print("\nColumns:")
print(df.columns.tolist())


# --------------------------------------------------
# 3. Prepare columns
# --------------------------------------------------

df = df[["Sentence", "Label"]].copy()

df.columns = ["input", "label"]


# --------------------------------------------------
# 4. Clean dataset
# --------------------------------------------------

print("\nCleaning dataset...")

df = df.dropna(subset=["input", "label"])

df["input"] = df["input"].astype(str).str.strip()

df["label"] = pd.to_numeric(
    df["label"],
    errors="coerce"
)

df = df.dropna(subset=["label"])

df = df[df["label"].isin([0, 1])]

df = df[df["input"] != ""]

df = df.drop_duplicates(
    subset=["input"]
)

df["label"] = df["label"].astype(int)


print(f"Cleaned dataset shape: {df.shape}")

print("\nLabel distribution:")
print(df["label"].value_counts())


# --------------------------------------------------
# 5. X and y
# --------------------------------------------------

X = df["input"]
y = df["label"]


# --------------------------------------------------
# 6. Train/Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")


# --------------------------------------------------
# 7. Character + Word TF-IDF
# --------------------------------------------------

features = FeatureUnion(
    [
        (
            "character_features",
            TfidfVectorizer(
                analyzer="char",
                ngram_range=(2, 5),
                min_df=2,
                max_features=100000,
                sublinear_tf=True,
            ),
        ),
        (
            "word_features",
            TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                min_df=2,
                max_features=50000,
                sublinear_tf=True,
            ),
        ),
    ]
)


# --------------------------------------------------
# 8. Build Model
# --------------------------------------------------

model = Pipeline(
    [
        ("features", features),

        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced"
            ),
        ),
    ]
)


# --------------------------------------------------
# 9. Train
# --------------------------------------------------

print("\nTraining Model V2...")
print("Character TF-IDF + Word TF-IDF")

model.fit(
    X_train,
    y_train
)

print("Training completed.")


# --------------------------------------------------
# 10. Evaluate
# --------------------------------------------------

print("\nEvaluating Model V2...")

y_pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print(f"\nAccuracy: {accuracy:.4f}")


# --------------------------------------------------
# 11. Classification Report
# --------------------------------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Normal",
            "SQL Injection"
        ]
    )
)


# --------------------------------------------------
# 12. Confusion Matrix
# --------------------------------------------------

print("Confusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# --------------------------------------------------
# 13. Save Model
# --------------------------------------------------

MODEL_DIR.mkdir(
    exist_ok=True
)

joblib.dump(
    model,
    MODEL_PATH
)

print("\nModel V2 saved successfully:")

print(MODEL_PATH)