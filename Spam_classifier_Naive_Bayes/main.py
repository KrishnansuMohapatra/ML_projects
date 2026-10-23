import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42
DATA_PATH = "spam.csv"
MODEL_PATH = "spam_nb_model.joblib"
ERRORS_PATH = "errors.csv"
LABELS = ["ham", "spam"]


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin1")
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df = df.rename(columns={"v1": "label", "v2": "message"})
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["message"] = df["message"].fillna("").astype(str)
    df = df[df["message"].str.len() > 0]
    df = df.drop_duplicates(subset="message").reset_index(drop=True)
    df["target"] = (df["label"] == "spam").astype(int)
    return df


def build_model() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.98,
                    sublinear_tf=True,
                    token_pattern=r"(?u)\b\w+\b",
                ),
            ),
            ("nb", MultinomialNB(alpha=0.1)),
        ]
    )


def tune_alpha(X_train, y_train) -> float:
    best_alpha, best_score = 0.1, -1.0
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for alpha in [0.01, 0.03, 0.1, 0.3, 0.5, 1.0]:
        score = cross_val_score(
            build_model().set_params(nb__alpha=alpha),
            X_train,
            y_train,
            cv=cv,
            scoring="f1_macro",
        ).mean()
        if score > best_score:
            best_alpha, best_score = alpha, score
    return best_alpha


def build_detail(model: Pipeline, X_test, y_test) -> pd.DataFrame:
    y_pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    detail = pd.DataFrame(
        {
            "message": X_test.to_numpy(),
            "actual": np.where(y_test.to_numpy() == 1, "spam", "ham"),
            "predicted": np.where(y_pred == 1, "spam", "ham"),
            "p_spam": proba,
        }
    )
    detail["error"] = detail["actual"] != detail["predicted"]
    detail["error_type"] = np.select(
        [
            detail["actual"].eq("ham") & detail["predicted"].eq("spam"),
            detail["actual"].eq("spam") & detail["predicted"].eq("ham"),
        ],
        ["false_positive", "false_negative"],
        default="correct",
    )
    text = detail["message"]
    detail["chars"] = text.str.len()
    detail["words"] = text.str.split().str.len()
    detail["digits"] = text.str.count(r"\d")
    detail["exclaims"] = text.str.count("!")
    detail["shout_runs"] = text.str.count(r"[A-Z]{3,}")
    detail["has_url"] = text.str.contains(r"https?://|www\.|\.com|\.co\.uk", case=False)
    detail["has_long_number"] = text.str.contains(r"\+?\d[\d\s\-]{8,}\d")
    detail["has_currency"] = text.str.contains(r"[$£€]|\bfree\b|\bwin\b|\bcash\b", case=False)
    return detail


def print_classification_report(model: Pipeline, detail: pd.DataFrame) -> None:
    y_test = (detail["actual"] == "spam").astype(int).to_numpy()
    y_pred = (detail["predicted"] == "spam").astype(int).to_numpy()
    report = classification_report(
        y_test, y_pred, target_names=LABELS, output_dict=True, zero_division=0
    )

    rows = [
        {
            "class": name,
            "precision": round(report[name]["precision"], 4),
            "recall": round(report[name]["recall"], 4),
            "f1-score": round(report[name]["f1-score"], 4),
            "support": int(report[name]["support"]),
        }
        for name in LABELS + ["macro avg", "weighted avg"]
    ]
    print("\n=== Classification report ===")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\naccuracy: {report['accuracy']:.4f}   n = {len(y_test)}")

    print("\n=== Confusion matrix (rows = actual, cols = predicted) ===")
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    cm_df = pd.DataFrame(cm, index=pd.Index(LABELS, name="actual\\predicted"), columns=LABELS)
    print(cm_df.to_string())
    tn, fp, fn, tp = cm.ravel()
    print(f"\ntrue negatives  (ham  -> ham) : {tn}")
    print(f"false positives (ham  -> spam): {fp}")
    print(f"false negatives (spam -> ham) : {fn}")
    print(f"true positives  (spam -> spam): {tp}")
    print(f"overall accuracy              : {accuracy_score(y_test, y_pred):.4f}")


def print_error_gallery(detail: pd.DataFrame, limit: int = 12) -> None:
    for kind, title in [
        ("false_positive", "FALSE POSITIVES - legit mail blocked as spam"),
        ("false_negative", "FALSE NEGATIVES - spam that got through"),
    ]:
        rows = detail[detail["error_type"].eq(kind)]
        print(f"\n=== {title} ({len(rows)} of {len(detail)} test messages) ===")
        if rows.empty:
            print("none")
            continue
        print(
            rows.sort_values("p_spam", ascending=kind == "false_positive")
            .head(limit)[["p_spam", "chars", "message"]]
            .to_string(
                index=False,
                formatters={"p_spam": lambda v: f"{v:.4f}"},
                max_colwidth=88,
            )
        )


def print_error_profile(detail: pd.DataFrame) -> None:
    print("\n=== Error distribution ===")
    counts = detail["error_type"].value_counts().reindex(
        ["correct", "false_positive", "false_negative"]
    )
    dist = pd.DataFrame(
        {
            "count": counts,
            "share": (counts / len(detail)).round(4),
        }
    )
    print(dist.to_string())

    for label in LABELS:
        block = detail[detail["actual"].eq(label)]
        missed = int(block["error"].sum())
        print(
            f"\n{label}: {missed}/{len(block)} wrong "
            f"({missed / len(block):.4f} error rate)"
        )

    print("\n=== Message profile by outcome (means) ===")
    cols = ["chars", "words", "digits", "exclaims", "shout_runs"]
    profile = detail.groupby("error_type")[cols].mean().round(1)
    profile = profile.reindex(["correct", "false_positive", "false_negative"])
    print(profile.to_string())

    print("\n=== Spam-style markers by outcome (row share) ===")
    flags = ["has_url", "has_long_number", "has_currency", "shout_runs"]
    marker = detail.groupby("error_type")[flags].mean().reindex(
        ["correct", "false_positive", "false_negative"]
    )
    print(marker.round(3).to_string())

    errors = detail[detail["error"]]
    if not errors.empty:
        print("\n=== Model confidence on errors ===")
        print(
            f"mean p(spam) on errors        : {errors['p_spam'].mean():.4f}\n"
            f"errors within 0.10-0.90 band : {int(errors['p_spam'].between(0.10, 0.90).sum())}"
            f"/{len(errors)}\n"
            f"errors beyond 0.10-0.90 band : "
            f"{int((errors['p_spam'] <= 0.10).sum() + (errors['p_spam'] >= 0.90).sum())}"
            f"/{len(errors)} (confidently wrong)"
        )


def threshold_sweep(detail: pd.DataFrame) -> pd.DataFrame:
    actual = (detail["actual"] == "spam").to_numpy().astype(int)
    proba = detail["p_spam"].to_numpy()
    rows = []
    for t in np.arange(0.05, 0.96, 0.05):
        pred = (proba >= t).astype(int)
        rows.append(
            {
                "threshold": round(float(t), 2),
                "flagged": int(pred.sum()),
                "precision": round(precision_score(actual, pred, zero_division=0), 4),
                "recall": round(recall_score(actual, pred, zero_division=0), 4),
                "f1": round(f1_score(actual, pred, zero_division=0), 4),
                "false_positives": int(((pred == 1) & (actual == 0)).sum()),
                "missed_spam": int(((pred == 0) & (actual == 1)).sum()),
            }
        )
    return pd.DataFrame(rows)


def print_threshold_analysis(detail: pd.DataFrame) -> None:
    sweep = threshold_sweep(detail)
    print("\n=== Decision-threshold sweep (current operating point = 0.50) ===")
    print(sweep.to_string(index=False))

    best = sweep.loc[sweep["f1"].idxmax()]
    print(f"\nhighest f1: {best['f1']:.4f} at threshold {best['threshold']:.2f} "
          f"(precision {best['precision']:.4f}, recall {best['recall']:.4f})")

    min_fp = int(sweep["false_positives"].min())
    if min_fp == 0:
        row = sweep.loc[sweep["recall"].idxmax()]
        print(
            f"zero false positives: threshold {row['threshold']:.2f} "
            f"-> recall {row['recall']:.4f}, {int(row['missed_spam'])} spam missed"
        )
    else:
        reachable = sweep[sweep["false_positives"].eq(min_fp)]
        row = reachable.loc[reachable["recall"].idxmax()]
        print(
            f"cleanest setting: {min_fp} false positive(s) minimum, best recall there is "
            f"{row['recall']:.4f} at threshold {row['threshold']:.2f} "
            f"({int(row['missed_spam'])} spam missed)"
        )


def error_analysis(model: Pipeline, X_test, y_test, save: bool = True) -> pd.DataFrame:
    detail = build_detail(model, X_test, y_test)

    print_classification_report(model, detail)
    print_error_gallery(detail)
    print_error_profile(detail)
    print_threshold_analysis(detail)

    errors = detail[detail["error"]].copy()
    if save and not errors.empty:
        errors.to_csv(ERRORS_PATH, index=False)
        print(f"\nsaved {len(errors)} misclassified messages -> {ERRORS_PATH}")

    return detail


def top_features(model: Pipeline, n: int = 20) -> pd.DataFrame:
    names = np.asarray(model.named_steps["tfidf"].get_feature_names_out())
    log_prob = model.named_steps["nb"].feature_log_prob_
    weight = log_prob[1] - log_prob[0]
    top = pd.DataFrame({"feature": names, "log_ratio": weight})
    top = pd.concat([top.nlargest(n, "log_ratio"), top.nsmallest(n, "log_ratio")])
    return top.sort_values("log_ratio", ascending=False)


def predict(model: Pipeline, messages: list[str], threshold: float = 0.5) -> list[dict]:
    proba = model.predict_proba(messages)[:, 1]
    labels = (proba >= threshold).astype(int)
    return [
        {"message": m, "label": "spam" if y else "ham", "spam_probability": round(float(p), 4)}
        for m, y, p in zip(messages, labels, proba)
    ]


def main() -> None:
    df = load_data()
    print("=== Dataset ===")
    print(df["label"].value_counts().to_string())
    print(f"total: {len(df)} messages")
    print(df.head().to_string())

    X_train, X_test, y_train, y_test = train_test_split(
        df["message"],
        df["target"],
        test_size=0.2,
        stratify=df["target"],
        random_state=RANDOM_STATE,
    )

    alpha = tune_alpha(X_train, y_train)
    print(f"\nselected alpha: {alpha}")

    model = build_model().set_params(nb__alpha=alpha)
    model.fit(X_train, y_train)

    error_analysis(model, X_test, y_test)

    print("\n=== 5-fold CV on full data (f1_macro) ===")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, df["message"], df["target"], cv=cv, scoring="f1_macro")
    print(f"mean: {scores.mean():.4f}  std: {scores.std():.4f}  folds: {scores}")

    print("\n=== Strongest features (top 15 spam / bottom 15 ham) ===")
    print(top_features(model, n=15).to_string(index=False))

    joblib.dump(model, MODEL_PATH)
    print(f"\nsaved pipeline -> {MODEL_PATH}")

    print("\n=== Sample predictions ===")
    for row in predict(
        model,
        [
            "Congratulations! You have won a $1000 gift card. Click here to claim now!",
            "Hey are we still meeting for lunch at 1pm?",
            "FREE entry to a prize draw, txt WIN to 80086 now",
            "Can you pick up milk on your way home?",
        ],
    ):
        print(f"{row['label']:>4}  p(spam)={row['spam_probability']:<7}  {row['message']}")


if __name__ == "__main__":
    main()
