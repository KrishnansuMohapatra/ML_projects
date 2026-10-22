import json

import pandas as pd
import matplotlib.pyplot as plt
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)


# ============================================
# 1. PROJECT PATHS
# ============================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "instagram_fake_account_enhanced_dataset.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)


# ============================================
# 2. LOAD DATASET
# ============================================

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# ============================================
# 3. SELECT FEATURES AND TARGET
# ============================================

features = [
    "userFollowerCount",
    "userFollowingCount",
    "userBiographyLength",
    "userMediaCount",
    "usernameDigitCount",
    "usernameLength"
]

X = df[features]
y = df["isFake"]


# ============================================
# 4. TRAIN / TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================
# 5. FEATURE SCALING
# ============================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================
# 6. TRAIN LOGISTIC REGRESSION
# ============================================

model = LogisticRegression(
    random_state=42,
    max_iter=1000
)

model.fit(X_train_scaled, y_train)


# ============================================
# 7. NORMAL PREDICTIONS
# ============================================

y_pred = model.predict(X_test_scaled)


# ============================================
# 8. MODEL EVALUATION
# ============================================

print("\n==============================")
print("MODEL EVALUATION")
print("==============================")

accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:")
print(f"{accuracy:.4f}")

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Non-Fake", "Fake"]
    )
)


# ============================================
# 9. PREDICTION PROBABILITIES
# ============================================

probabilities = model.predict_proba(X_test_scaled)[:, 1]


# ============================================
# 10. THRESHOLD EXPERIMENT
# ============================================

print("\n==============================")
print("THRESHOLD EXPERIMENT")
print("==============================")

for threshold in [0.5, 0.4, 0.3, 0.2]:
    y_threshold = (probabilities >= threshold).astype(int)
    cm_threshold = confusion_matrix(y_test, y_threshold)
    tn, fp, fn, tp = cm_threshold.ravel()

    fake_recall = tp / (tp + fn)
    fake_precision = tp / (tp + fp)

    print(f"\nThreshold: {threshold}")
    print(f"TN: {tn}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")
    print(f"TP: {tp}")
    print(f"Fake Precision: {fake_precision:.2%}")
    print(f"Fake Recall: {fake_recall:.2%}")


# ============================================
# 11. ROC CURVE + ROC-AUC
# ============================================

fpr, tpr, roc_thresholds = roc_curve(y_test, probabilities)
auc_score = roc_auc_score(y_test, probabilities)

print("\n==============================")
print("ROC-AUC")
print("==============================")
print(f"ROC-AUC: {auc_score:.4f}")

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {auc_score:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Instagram Fake Account Detection")
plt.legend()
plt.grid()
plt.savefig(MODEL_DIR / "roc_curve.png", dpi=150, bbox_inches="tight")
plt.close()


# ============================================
# 12. PRECISION-RECALL CURVE
# ============================================

pr_precision, pr_recall, pr_thresholds = precision_recall_curve(
    y_test,
    probabilities
)

average_precision = average_precision_score(y_test, probabilities)

print("\n==============================")
print("PRECISION-RECALL")
print("==============================")
print(f"Average Precision: {average_precision:.4f}")

plt.figure(figsize=(8, 6))
plt.plot(pr_recall, pr_precision, label=f"Logistic Regression (AP = {average_precision:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve - Instagram Fake Account Detection")
plt.legend()
plt.grid()
plt.savefig(MODEL_DIR / "precision_recall_curve.png", dpi=150, bbox_inches="tight")
plt.close()


# ============================================
# 13. FIND BEST THRESHOLD USING F1
# ============================================

# precision and recall contain one extra
# value compared with thresholds, so slice both to align.

f1_scores = (
    2 * pr_precision[:-1] * pr_recall[:-1]
    / (pr_precision[:-1] + pr_recall[:-1] + 1e-10)
)

best_index = f1_scores.argmax()

best_threshold = pr_thresholds[best_index]
best_precision = pr_precision[best_index]
best_recall = pr_recall[best_index]
best_f1 = f1_scores[best_index]

print("\n==============================")
print("BEST THRESHOLD")
print("==============================")
print(f"Threshold : {best_threshold:.4f}")
print(f"Precision : {best_precision:.4f}")
print(f"Recall    : {best_recall:.4f}")
print(f"F1 Score  : {best_f1:.4f}")


# ============================================
# 14. EVALUATE BEST THRESHOLD
# ============================================

y_best = (probabilities >= best_threshold).astype(int)
cm_best = confusion_matrix(y_test, y_best)
tn, fp, fn, tp = cm_best.ravel()

print("\n==============================")
print("BEST THRESHOLD EVALUATION")
print("==============================")
print(f"TN: {tn}")
print(f"FP: {fp}")
print(f"FN: {fn}")
print(f"TP: {tp}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_best,
        target_names=["Non-Fake", "Fake"]
    )
)


# ============================================
# 15. SAVE MODEL, SCALER, AND THRESHOLD
# ============================================

model_path = MODEL_DIR / "instagram_fake_model.pkl"
scaler_path = MODEL_DIR / "instagram_fake_scaler.pkl"
metadata_path = MODEL_DIR / "model_metadata.json"

joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)

metadata = {
    "features": features,
    "best_threshold": float(best_threshold),
    "best_precision": float(best_precision),
    "best_recall": float(best_recall),
    "best_f1": float(best_f1),
    "roc_auc": float(auc_score),
    "average_precision": float(average_precision)
}

with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)

print("\n==============================")
print("MODEL SAVING")
print("==============================")
print(f"Model saved to:\n{model_path}")
print(f"Scaler saved to:\n{scaler_path}")
print(f"Metadata (incl. best_threshold) saved to:\n{metadata_path}")
print(f"ROC curve saved to:\n{MODEL_DIR / 'roc_curve.png'}")
print(f"PR curve saved to:\n{MODEL_DIR / 'precision_recall_curve.png'}")

print("\nProject training completed successfully.")