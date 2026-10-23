"""
main.py — Fake News Detection: Training Pipeline
=================================================
Loads the True/Fake news dataset, prevents duplicate leakage with a
group-aware split, trains a Logistic Regression classifier on TF-IDF
features, and persists the model + vectorizer for the Streamlit frontend.
"""

import pandas as pd
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# ── paths ────────────────────────────────────────────────────────────
TRUE_CSV = "data/raw/True.csv"
FAKE_CSV = "data/raw/Fake.csv"
MODEL_PATH = "fake_news_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"


# ── 1. load & label ─────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    true_df = pd.read_csv(TRUE_CSV)
    fake_df = pd.read_csv(FAKE_CSV)

    true_df["label"] = 1  # Real
    fake_df["label"] = 0  # Fake

    df = pd.concat([true_df, fake_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Loaded  — True: {len(true_df):,}  |  Fake: {len(fake_df):,}  |  Total: {len(df):,}")
    return df


# ── 2. clean ─────────────────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # drop rows with blank text
    df = df[df["text"].fillna("").str.strip() != ""].copy()

    # safe string columns (prevent NaN propagation in concatenation)
    df["title"] = df["title"].fillna("").str.strip()
    df["text"] = df["text"].fillna("").str.strip()

    # combined feature
    df["combined_text"] = (df["title"] + " " + df["text"]).str.strip()

    # integer group id for every unique article body (memory-efficient)
    df["text_group"] = pd.factorize(df["text"])[0]

    print(f"Cleaned — rows: {len(df):,}  |  unique texts: {df['text_group'].nunique():,}")
    return df


# ── 3. duplicate diagnostics ────────────────────────────────────────
def report_duplicates(df: pd.DataFrame) -> None:
    dup_count = df["text"].duplicated().sum()
    unique_dup_texts = df["text"].value_counts().loc[lambda x: x > 1]
    conflicts = df.groupby("text")["label"].nunique()
    conflict_count = (conflicts > 1).sum()

    print("\n── Duplicate Diagnostics ──")
    print(f"  Duplicate text rows          : {dup_count:,}")
    print(f"  Texts appearing >1 time      : {len(unique_dup_texts):,}")
    print(f"  Conflicting-label groups     : {conflict_count:,}")

    if conflict_count > 0:
        print("\n  ⚠  Some articles appear in BOTH Fake and Real sets.")
        print("     GroupShuffleSplit will keep each article in one split only.\n")


# ── 4. group-aware split (prevents data leakage) ────────────────────
def group_split(df: pd.DataFrame, test_size: float = 0.2):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)

    train_idx, test_idx = next(
        gss.split(df["combined_text"], df["label"], groups=df["text_group"])
    )

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    # sanity: zero text overlap
    overlap = set(train_df["text"]) & set(test_df["text"])
    assert len(overlap) == 0, f"Leakage detected! {len(overlap)} overlapping texts."

    print("\n── Group-Aware Split ──")
    print(f"  Train: {len(train_df):,}  |  Test: {len(test_df):,}")
    print(f"  Train label balance (real %): {train_df['label'].mean():.2%}")
    print(f"  Test  label balance (real %): {test_df['label'].mean():.2%}")
    print(f"  Text overlap between splits : {len(overlap)}  ✓")

    return train_df, test_df


# ── 5. vectorize ─────────────────────────────────────────────────────
def build_vectorizer():
    return TfidfVectorizer(
        max_features=50_000,
        ngram_range=(1, 2),
        stop_words="english",
        min_df=2,
        max_df=0.95,
    )


# ── 6. train & evaluate ─────────────────────────────────────────────
def train_and_evaluate(train_df, test_df):
    vectorizer = build_vectorizer()

    X_train_tfidf = vectorizer.fit_transform(train_df["combined_text"])
    X_test_tfidf = vectorizer.transform(test_df["combined_text"])

    print(f"\n  TF-IDF features: {X_train_tfidf.shape[1]:,}")

    model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    model.fit(X_train_tfidf, train_df["label"])

    y_pred = model.predict(X_test_tfidf)

    print("\n── Results (Group-Split, No Leakage) ──")
    print(f"  Accuracy: {accuracy_score(test_df['label'], y_pred):.4f}\n")
    print(classification_report(
        test_df["label"], y_pred, target_names=["Fake", "Real"]
    ))
    print("Confusion Matrix:")
    print(confusion_matrix(test_df["label"], y_pred))

    return model, vectorizer


# ── 7. error analysis ───────────────────────────────────────────────
def error_analysis(model, vectorizer, test_df):
    results = test_df[["title", "text", "combined_text", "label"]].copy()
    results["prediction"] = model.predict(
        vectorizer.transform(results["combined_text"])
    )
    results["prob_real"] = model.predict_proba(
        vectorizer.transform(results["combined_text"])
    )[:, 1]
    results["confidence"] = results["prob_real"].apply(
        lambda p: p if p >= 0.5 else 1 - p
    )

    errors = results[results["label"] != results["prediction"]]
    print(f"\n── Error Analysis ──")
    print(f"  Total errors: {len(errors):,} / {len(results):,}")

    fake_as_real = errors[(errors["label"] == 0) & (errors["prediction"] == 1)]
    real_as_fake = errors[(errors["label"] == 1) & (errors["prediction"] == 0)]
    print(f"  Fake → Real : {len(fake_as_real):,}")
    print(f"  Real → Fake : {len(real_as_fake):,}")

    print("\n  Top high-confidence mistakes:")
    print(
        errors[["label", "prediction", "confidence", "title"]]
        .sort_values("confidence", ascending=False)
        .head(10)
        .to_string(index=False)
    )


# ── 8. persist ───────────────────────────────────────────────────────
def save_artifacts(model, vectorizer):
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"\n✓ Model saved     → {MODEL_PATH}")
    print(f"✓ Vectorizer saved → {VECTORIZER_PATH}")


# ── main ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data()
    df = clean_data(df)
    report_duplicates(df)

    train_df, test_df = group_split(df)
    model, vectorizer = train_and_evaluate(train_df, test_df)
    error_analysis(model, vectorizer, test_df)
    save_artifacts(model, vectorizer)