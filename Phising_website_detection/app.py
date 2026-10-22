"""Streamlit front-end for the phishing website detector.

Run with:
    uv run streamlit run app.py
"""

import re

import pandas as pd
import streamlit as st

from phishing_detection.config import (
    BATCH_DIR,
    FEATURES,
    LABEL_NAMES,
    LEGITIMATE_LABEL,
    MODEL_PATH,
    PHISHING_LABEL,
)
from phishing_detection.predict import load_model


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="Phishing Website Detector",
    page_icon="🎣",
    layout="wide",
)


# ============================================================
# Load model
# ============================================================

@st.cache_resource
def get_model():
    """Load and cache the trained model."""
    if not MODEL_PATH.exists():
        return None
    return load_model(MODEL_PATH)


model = get_model()


# ============================================================
# Header
# ============================================================

st.title("🎣 Phishing Website Detector")

st.write(
    "A Random Forest machine-learning model for classifying "
    "websites using 30 encoded security-related features."
)

st.warning(
    "⚠️ Educational/portfolio project. This model does not inspect a raw URL "
    "directly and should not be used as the sole basis for a security decision."
)

if model is None:
    st.error(
        "Model file not found. Train one with `uv run phishing-detector train`, "
        "or place a trained model at `models/phishing_model.pkl`."
    )
    st.stop()


# ============================================================
# Tabs
# ============================================================

single_tab, batch_tab, info_tab = st.tabs(
    ["🔍 Single Test", "📁 Batch CSV", "ℹ️ About"]
)


# ============================================================
# Single prediction
# ============================================================

with single_tab:

    st.subheader("Test Box")

    st.write(
        "Paste exactly 30 feature values. The app accepts commas, spaces, "
        "new lines, and square brackets."
    )

    example = (
        "-1, -1, 1, 1, -1, -1, 1, 1, 1, 1, "
        "0, 1, 1, 1, 1, 1, -1, -1, -1, -1, "
        "-1, -1, -1, 1, 1, 1, 1, 1, 1, 1"
    )

    test_input = st.text_area(
        "Enter 30 feature values",
        value=example,
        height=130,
    )

    if st.button(
        "🔍 Analyze",
        type="primary",
        width="stretch",
    ):

        # Extract integers robustly.
        values = [
            int(value)
            for value in re.findall(r"-?\d+", test_input)
        ]

        st.write(f"**Features detected:** {len(values)} / 30")

        if len(values) != 30:

            st.error(
                f"❌ Expected 30 features, but detected {len(values)}."
            )

        elif not all(value in (-1, 0, 1) for value in values):

            st.error(
                "❌ Invalid values. Every feature must be -1, 0, or 1."
            )

        else:

            input_df = pd.DataFrame([values], columns=FEATURES)

            prediction = model.predict(input_df)[0]

            st.divider()

            if prediction == PHISHING_LABEL:
                st.error(f"🚨 Prediction: {LABEL_NAMES[PHISHING_LABEL].upper()}")
            elif prediction == LEGITIMATE_LABEL:
                st.success(
                    f"✅ Prediction: {LABEL_NAMES[LEGITIMATE_LABEL].upper()}"
                )
            else:
                st.info(f"Model returned class: {prediction}")

            # Probability
            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(input_df)[0]

                class_probabilities = dict(
                    zip(model.classes_, probabilities)
                )

                phishing_probability = float(
                    class_probabilities.get(PHISHING_LABEL, 0)
                )

                legitimate_probability = float(
                    class_probabilities.get(LEGITIMATE_LABEL, 0)
                )

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Phishing probability",
                        f"{phishing_probability:.2%}",
                    )

                with col2:
                    st.metric(
                        "Legitimate probability",
                        f"{legitimate_probability:.2%}",
                    )

                st.progress(phishing_probability)

            with st.expander("View input features"):
                st.dataframe(
                    input_df.T.rename(columns={0: "value"}),
                    width="stretch",
                )


# ============================================================
# Batch CSV prediction
# ============================================================

with batch_tab:

    st.subheader("📁 Batch Prediction")

    st.write(
        "Upload a CSV containing the same 30 feature columns used during "
        "training. Every row predicted as phishing is flagged below."
    )

    # Choose a CSV either from the project's batch/ folder or upload one.
    batch_files = (
        sorted(BATCH_DIR.glob("*.csv")) if BATCH_DIR.exists() else []
    )

    source_mode = st.radio(
        "Choose input method",
        ["📂 Use file from batch folder", "⬆️ Upload CSV"],
        horizontal=True,
    )

    input_file = None

    if source_mode == "📂 Use file from batch folder":
        if batch_files:
            input_file = st.selectbox(
                "Select a batch CSV",
                batch_files,
                format_func=lambda p: p.name,
            )
        else:
            st.info(
                "No CSV files found in the `batch` folder. "
                "Add your sample CSV files there or choose Upload CSV."
            )
    else:
        input_file = st.file_uploader("Upload CSV", type=["csv"])

    # Only process when a real file has been selected
    if input_file is not None:
        try:
            data = pd.read_csv(input_file)
            file_name = getattr(input_file, "name", str(input_file))

            st.success(f"Loaded **{file_name}** — {len(data)} rows")

            missing = [f for f in FEATURES if f not in data.columns]

            if missing:
                st.error("❌ The CSV is missing required features:")
                st.code("\n".join(missing))
                st.stop()

            # Convert to numbers; anything non-numeric becomes NaN.
            X = data[FEATURES].apply(pd.to_numeric, errors="coerce")

            if X.isna().any().any():
                bad_rows = (X[X.isna().any(axis=1)].index + 2).tolist()
                st.error(
                    "❌ The CSV has missing or non-numeric feature values "
                    f"in row(s): {bad_rows[:20]}"
                    + (" ..." if len(bad_rows) > 20 else "")
                )
                st.stop()

            invalid_values = sorted(set(X.to_numpy().ravel()) - {-1, 0, 1})

            if invalid_values:
                st.error(
                    f"❌ Invalid feature values found: {invalid_values}. "
                    "Only -1, 0, and 1 are allowed."
                )
                st.stop()

            X = X.astype(int)

            # ---------------- Predict ----------------
            predictions = model.predict(X)
            is_phishing = predictions == PHISHING_LABEL
            is_legitimate = predictions == LEGITIMATE_LABEL

            result = data.copy()

            # row_number matches the row number you see in Excel
            # (row 1 is the header, so the first data row is row 2).
            result.insert(0, "row_number", data.index + 2)

            result.insert(
                1,
                "prediction",
                [
                    LABEL_NAMES.get(p, str(p))
                    for p in predictions
                ],
            )

            has_proba = False
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(X)
                class_to_index = {c: i for i, c in enumerate(model.classes_)}

                if PHISHING_LABEL in class_to_index:
                    result.insert(
                        2,
                        "phishing_probability",
                        probabilities[:, class_to_index[PHISHING_LABEL]],
                    )
                    has_proba = True

            # ---------------- Summary ----------------
            st.divider()

            total = len(result)
            phishing_count = int(is_phishing.sum())
            legitimate_count = int(is_legitimate.sum())

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total rows", total)
            c2.metric("🚨 Phishing", phishing_count)
            c3.metric("✅ Legitimate", legitimate_count)
            c4.metric(
                "Phishing share",
                f"{phishing_count / total:.1%}" if total else "0%",
            )

            # ---------------- Flagged phishing rows ----------------
            st.subheader("🚨 Flagged Phishing Rows")

            phishing_df = result[is_phishing]

            if phishing_df.empty:
                st.success("No phishing rows detected in this file.")
            else:
                if has_proba:
                    phishing_df = phishing_df.sort_values(
                        "phishing_probability", ascending=False
                    )

                st.error(
                    f"{phishing_count} row(s) flagged as PHISHING. "
                    f"Row numbers: "
                    f"{phishing_df['row_number'].head(50).tolist()}"
                    + (" ..." if phishing_count > 50 else "")
                )

                phishing_config = {}
                if has_proba:
                    phishing_config["phishing_probability"] = (
                        st.column_config.ProgressColumn(
                            "phishing_probability",
                            min_value=0.0,
                            max_value=1.0,
                            format="%.2f",
                        )
                    )

                st.dataframe(
                    phishing_df,
                    width="stretch",
                    hide_index=True,
                    column_config=phishing_config,
                )

                st.download_button(
                    "⬇️ Download Phishing Rows Only",
                    data=phishing_df.to_csv(index=False).encode("utf-8"),
                    file_name="phishing_rows_only.csv",
                    mime="text/csv",
                    key="download_phishing_only",
                )

            # ---------------- All rows, phishing highlighted ----------------
            st.subheader("📋 All Predictions (phishing rows highlighted)")

            def highlight_phishing(row):
                color = (
                    "background-color: #ffd6d6; color: #8b0000"
                    if row["prediction"] == "Phishing"
                    else ""
                )
                return [color] * len(row)

            # Styling very large tables is slow, so skip it above ~200k cells.
            if result.size <= 200_000:
                styled = result.style.apply(highlight_phishing, axis=1)
                if has_proba:
                    styled = styled.format({"phishing_probability": "{:.2%}"})
                st.dataframe(styled, width="stretch", hide_index=True)
            else:
                st.caption(
                    "File is large, so row highlighting is turned off. "
                    "Use the flagged table above."
                )
                st.dataframe(result, width="stretch", hide_index=True)

            st.download_button(
                "⬇️ Download All Predictions",
                data=result.to_csv(index=False).encode("utf-8"),
                file_name="phishing_predictions.csv",
                mime="text/csv",
                key="download_all",
            )

        except Exception as e:
            st.error(f"❌ Could not process the CSV: {e}")


# ============================================================
# About
# ============================================================

with info_tab:

    st.subheader("About the project")

    st.markdown(
        """
### Model

**Random Forest Classifier**

The model was trained using 30 encoded website features.

### Important dataset limitation

The dataset contains repeated feature patterns. Therefore, a normal
random train/test split can place identical feature vectors in both
training and testing data.

A stricter duplicate-aware evaluation was also performed during the
project. The resulting performance was approximately **96-97% accuracy**.

### What this app does

```text
30 encoded features
        ↓
Pandas DataFrame
        ↓
Random Forest
        ↓
Prediction
        ↓
Phishing / Legitimate
```

### What this app does NOT do

This version does **not** take a raw URL and automatically extract the
30 features.

Therefore, entering a URL such as:

```text
https://example.com
```

is not valid input for the Test Box.

The Test Box expects values such as:

```text
-1, -1, 1, 1, -1, ...
```

with exactly 30 values.
"""
    )

    st.info(
        "For a production security system, use this model only as one "
        "component of a larger security pipeline."
    )


# ============================================================
# Footer
# ============================================================

st.divider()

st.caption(
    "Phishing Website Detection • Random Forest • "
    "Educational/Portfolio Demonstration"
)
