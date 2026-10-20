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
MODEL_PATH = MODEL_DIR / "sql_injection_model_v3.pkl"


# --------------------------------------------------
# 2. Load SQLiV2
# --------------------------------------------------

print("Loading SQLiV2 dataset...")

df = pd.read_csv(
    DATA_PATH,
    encoding="utf-16"
)

print(f"Original dataset shape: {df.shape}")


# --------------------------------------------------
# 3. Select columns
# --------------------------------------------------

df = df[["Sentence", "Label"]].copy()

df.columns = ["input", "label"]


# --------------------------------------------------
# 4. Clean dataset
# --------------------------------------------------

print("\nCleaning dataset...")

df = df.dropna(
    subset=["input", "label"]
)

df["input"] = (
    df["input"]
    .astype(str)
    .str.strip()
)

df["label"] = pd.to_numeric(
    df["label"],
    errors="coerce"
)

df = df.dropna(
    subset=["label"]
)

df = df[df["label"].isin([0, 1])]

df = df[df["input"] != ""]

df = df.drop_duplicates(
    subset=["input"]
)

df["label"] = df["label"].astype(int)


# --------------------------------------------------
# 5. Add hard-negative examples
# --------------------------------------------------

print("\nAdding hard-negative examples...")

hard_negative_texts = [
    "Please select a product",
    "Select your preferred option",
    "Select a country",
    "Select your language",
    "Select your payment method",
    "Select an item from the list",
    "Please select your location",
    "Select a category",
    "Select your preferred course",
    "Select a date",
    "Union membership benefits",
    "Union membership application",
    "Join the workers union",
    "Update my profile",
    "Update your contact details",
    "Update your account information",
    "Delete my account",
    "Delete this message",
    "Delete the selected file",
    "Insert your address",
    "Insert your phone number",
    "Insert a new row",
    "Drop-down menu",
    "Open the drop-down menu",
    "Administrator account",
    "Contact the administrator",
    "Administrator dashboard",
    "Select all files",
    "Update the application",
    "Delete the old record",
]

hard_negatives = pd.DataFrame(
    {
        "input": hard_negative_texts,
        "label": 0,
    }
)

df = pd.concat(
    [df, hard_negatives],
    ignore_index=True
)

df = df.drop_duplicates(
    subset=["input"]
)


print(
    f"Dataset after hard negatives: {df.shape}"
)


# --------------------------------------------------
# 6. Show distribution
# --------------------------------------------------

print("\nFinal label distribution:")

print(
    df["label"].value_counts()
)


# --------------------------------------------------
# 7. X and y
# --------------------------------------------------

X = df["input"]
y = df["label"]


# --------------------------------------------------
# 8. Train/Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(
    f"\nTraining samples: {len(X_train)}"
)

print(
    f"Testing samples: {len(X_test)}"
)


# --------------------------------------------------
# 9. Character + Word TF-IDF
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
# 10. Model
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
# 11. Train
# --------------------------------------------------

print("\nTraining Model V3...")

print(
    "Character TF-IDF + Word TF-IDF"
)

print(
    "With hard-negative examples"
)

model.fit(
    X_train,
    y_train
)

print("\nTraining completed.")


# --------------------------------------------------
# 12. Evaluate
# --------------------------------------------------

print("\nEvaluating Model V3...")

y_pred = model.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print(
    f"\nAccuracy: {accuracy:.4f}"
)


# --------------------------------------------------
# 13. Classification report
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
# 14. Confusion matrix
# --------------------------------------------------

print("Confusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# --------------------------------------------------
# 15. Save model
# --------------------------------------------------

MODEL_DIR.mkdir(
    exist_ok=True
)

joblib.dump(
    model,
    MODEL_PATH
)

print(
    "\nModel V3 saved successfully:"
)

print(MODEL_PATH)