"""Model evaluation: standard holdout, strict duplicate-aware holdout,
cross-validation, and error analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)

from .config import FEATURES, N_ESTIMATORS, RANDOM_STATE, TARGET_COLUMN, TEST_SIZE


# ---------------------------------------------------------------
# Standard evaluation
# ---------------------------------------------------------------

def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Evaluate a fitted model on a holdout set."""
    predictions = model.predict(X_test)

    matrix = confusion_matrix(y_test, predictions, labels=[-1, 1])

    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "confusion_matrix": matrix,
        "report": classification_report(
            y_test,
            predictions,
            labels=[-1, 1],
            target_names=["Legitimate", "Phishing"],
            digits=4,
            zero_division=0,
        ),
        "predictions": predictions,
        "false_negatives": int(matrix[1][0]),
        "false_positives": int(matrix[0][1]),
    }


def feature_importance(model, X: pd.DataFrame) -> pd.Series:
    """Return feature importances sorted from most to least important."""
    importances = pd.Series(
        model.feature_importances_,
        index=X.columns,
    )
    return importances.sort_values(ascending=False)


# ---------------------------------------------------------------
# Duplicate-aware (strict) evaluation
# ---------------------------------------------------------------

def find_conflicting_groups(frame: pd.DataFrame) -> pd.Index:
    """Return the feature vectors that carry more than one label.

    The UCI Phishing Websites dataset contains repeated feature patterns, and
    some patterns are labelled inconsistently. Those groups are ambiguous and
    make any accuracy score unreliable, so a strict evaluation removes them.
    """
    label_counts = frame.groupby(FEATURES, sort=False)[TARGET_COLUMN].nunique()
    conflicting = label_counts[label_counts > 1].index
    return conflicting


def deduplicate_dataset(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Index]:
    """Drop ambiguous groups and collapse identical feature vectors.

    Returns the cleaned frame together with the conflicting feature vectors
    that were removed.
    """
    conflicting = find_conflicting_groups(frame)

    indexed = frame.set_index(FEATURES)
    keep = ~indexed.index.isin(conflicting)
    cleaned = indexed[keep].reset_index()

    cleaned = cleaned.drop_duplicates(subset=FEATURES, keep="first")

    return cleaned, conflicting


def duplicate_aware_holdout(
    frame: pd.DataFrame,
    n_estimators: int = N_ESTIMATORS,
    random_state: int = RANDOM_STATE,
    test_size: float = TEST_SIZE,
) -> dict:
    """Train and evaluate on deduplicated, unambiguous feature vectors.

    This is the honest evaluation: no test row shares its exact feature
    vector with a training row.
    """
    cleaned, conflicting = deduplicate_dataset(frame)

    X = cleaned[FEATURES]
    y = cleaned[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    results = evaluate_model(model, X_test, y_test)
    results["model"] = model
    results["conflicting_groups_removed"] = len(conflicting)
    results["rows_before"] = len(frame)
    results["rows_after"] = len(cleaned)

    return results


def standard_cross_validate(
    frame: pd.DataFrame,
    n_estimators: int = N_ESTIMATORS,
    random_state: int = RANDOM_STATE,
    cv: int = 5,
) -> np.ndarray:
    """Five-fold cross-validation over the full dataset.

    This is the optimistic number: identical feature vectors can appear in
    both the training and the validation fold. Use `strict_cross_validate`
    for the honest estimate.
    """
    X = frame[FEATURES]
    y = frame[TARGET_COLUMN]

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
    )

    return cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy",
    )


def strict_cross_validate(
    frame: pd.DataFrame,
    n_estimators: int = N_ESTIMATORS,
    random_state: int = RANDOM_STATE,
    cv: int = 5,
) -> np.ndarray:
    """Five-fold cross-validation over the cleaned unique feature patterns."""
    cleaned, _ = deduplicate_dataset(frame)

    X = cleaned[FEATURES]
    y = cleaned[TARGET_COLUMN]

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
    )

    return cross_val_score(
        model,
        X,
        y,
        cv=StratifiedKFold(
            n_splits=cv,
            shuffle=True,
            random_state=random_state,
        ),
        scoring="accuracy",
    )


# ---------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------

def error_analysis(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    predictions: np.ndarray,
) -> dict:
    """Compare the feature profile of false negatives against all phishing rows.

    A false negative (a phishing site judged legitimate) is the most costly
    mistake for a security classifier, so it is worth knowing what those rows
    look like.
    """
    is_phishing = y_test == 1
    is_false_negative = is_phishing & (predictions == -1)

    false_negatives = X_test[is_false_negative]
    phishing_rows = X_test[is_phishing]

    if false_negatives.empty:
        return {
            "false_negative_count": 0,
            "comparison": pd.DataFrame(),
            "false_negative_rows": false_negatives,
        }

    comparison = pd.DataFrame(
        {
            "All_Phishing": phishing_rows.mean(),
            "False_Negatives": false_negatives.mean(),
        }
    )
    comparison["Difference"] = (
        comparison["False_Negatives"] - comparison["All_Phishing"]
    )

    return {
        "false_negative_count": int(is_false_negative.sum()),
        "comparison": comparison.sort_values("Difference"),
        "false_negative_rows": false_negatives,
    }


def build_model(
    n_estimators: int = N_ESTIMATORS,
    random_state: int = RANDOM_STATE,
) -> RandomForestClassifier:
    """Create an unfitted Random Forest classifier with the project defaults."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
    )


def refit_on_all_data(model, X: pd.DataFrame, y: pd.Series):
    """Refit a model on the full dataset using the same hyper-parameters."""
    final_model = clone(model)
    final_model.fit(X, y)
    return final_model
