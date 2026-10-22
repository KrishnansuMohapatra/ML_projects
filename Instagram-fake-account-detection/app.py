import json
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "instagram_fake_model.pkl"
SCALER_PATH = MODEL_DIR / "instagram_fake_scaler.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

FEATURE_COLUMNS = [
    "userFollowerCount",
    "userFollowingCount",
    "userBiographyLength",
    "userMediaCount",
    "usernameDigitCount",
    "usernameLength",
]

st.set_page_config(
    page_title="Instagram Fake Account Detector",
    page_icon="📱",
    layout="wide",
)


# --------------------------------------------------
# CHART HELPERS
# --------------------------------------------------
def probability_vs_threshold_chart(fake_probability, threshold):
    """Bar for the fake probability with a dashed line at the threshold."""
    df = pd.DataFrame({"Label": ["Fake probability"], "Value": [fake_probability]})
    prob_scale = alt.Scale(domain=[0, 1])

    bars = alt.Chart(df).mark_bar(size=40).encode(
        y=alt.Y("Label:N", title=None),
        x=alt.X("Value:Q", scale=prob_scale, title="Probability"),
        color=alt.condition(
            alt.datum.Value >= threshold,
            alt.value("#e5484d"),
            alt.value("#30a46c"),
        ),
        tooltip=[alt.Tooltip("Value:Q", format=".2%", title="Fake probability")],
    )
    rule = (
        alt.Chart(pd.DataFrame({"t": [threshold]}))
        .mark_rule(color="orange", strokeDash=[6, 4], size=3)
        .encode(
            x=alt.X("t:Q", scale=prob_scale),
            tooltip=[alt.Tooltip("t:Q", format=".4f", title="Threshold")],
        )
    )
    return (bars + rule).properties(height=100)


def relative_feature_chart(X_scaled):
    """Scaled values = how far each feature is from the training average."""
    df = pd.DataFrame({
        "Feature": FEATURE_COLUMNS,
        "Deviation": X_scaled[0],
    })
    bars = alt.Chart(df).mark_bar().encode(
        y=alt.Y("Feature:N", sort=None, title=None),
        x=alt.X("Deviation:Q", title="Std. deviations from training average"),
        color=alt.condition(
            alt.datum.Deviation > 0,
            alt.value("#3e63dd"),
            alt.value("#f76b15"),
        ),
        tooltip=["Feature", alt.Tooltip("Deviation:Q", format=".2f")],
    )
    zero = (
        alt.Chart(pd.DataFrame({"z": [0]}))
        .mark_rule(color="gray")
        .encode(x="z:Q")
    )
    return (bars + zero).properties(height=220)


def probability_distribution_chart(probs, threshold):
    """Histogram of fake probabilities with a dashed threshold line."""
    df = pd.DataFrame({"fake_probability": probs})
    prob_scale = alt.Scale(domain=[0, 1])

    hist = alt.Chart(df).mark_bar().encode(
        x=alt.X(
            "fake_probability:Q",
            bin=alt.Bin(maxbins=20, extent=[0, 1]),
            scale=prob_scale,
            title="Fake probability",
        ),
        y=alt.Y("count():Q", title="Accounts"),
    )
    rule = (
        alt.Chart(pd.DataFrame({"t": [threshold]}))
        .mark_rule(color="orange", strokeDash=[6, 4], size=3)
        .encode(x=alt.X("t:Q", scale=prob_scale))
    )
    return (hist + rule).properties(height=250)


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    threshold = 0.1935

    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            threshold = json.load(f).get("best_threshold", threshold)

    return model, scaler, threshold


st.title("📱 Instagram Fake Account Detector")
st.write(
    "Estimate whether an Instagram account appears Fake or Non-Fake "
    "using Logistic Regression."
)
st.info(
    "Educational ML demonstration only. A prediction does not prove "
    "that an account or person is fake."
)

single_tab, bulk_tab = st.tabs(
    ["🔍 Single Account Detection", "📁 Bulk Detection (CSV Upload)"]
)

# --------------------------------------------------
# TAB 1: SINGLE ACCOUNT DETECTION
# --------------------------------------------------
with single_tab:
    st.subheader("Enter Account Information")

    col1, col2 = st.columns(2)

    with col1:
        followers = st.number_input(
            "Followers", min_value=0, value=100
        )
        following = st.number_input(
            "Following", min_value=0, value=100
        )
        bio_length = st.number_input(
            "Biography Length", min_value=0, value=50
        )

    with col2:
        media_count = st.number_input(
            "Media Count", min_value=0, value=20
        )
        username_digit_count = st.number_input(
            "Username Digit Count", min_value=0, value=1
        )
        username_length = st.number_input(
            "Username Length", min_value=1, value=10
        )

    if st.button(
        "🔍 Detect Account", key="single_detect", width="stretch"
    ):
        try:
            model, scaler, threshold = load_artifacts()

            X = pd.DataFrame({
                "userFollowerCount": [followers],
                "userFollowingCount": [following],
                "userBiographyLength": [bio_length],
                "userMediaCount": [media_count],
                "usernameDigitCount": [username_digit_count],
                "usernameLength": [username_length],
            })

            X_scaled = scaler.transform(X)
            probs = model.predict_proba(X_scaled)[0]

            fake_probability = float(probs[1])
            non_fake_probability = float(probs[0])

            st.divider()
            st.subheader("Prediction")

            if fake_probability >= threshold:
                st.error("🚨 Suspicious / Likely Fake")
            else:
                st.success("✅ Likely Non-Fake")

            c1, c2 = st.columns(2)
            c1.metric("Fake Probability", f"{fake_probability:.2%}")
            c2.metric("Non-Fake Probability", f"{non_fake_probability:.2%}")

            st.write(f"**Decision threshold:** `{threshold:.4f}`")
            st.progress(fake_probability)

            st.markdown(
                "**Fake probability vs. threshold** (dashed line = threshold)"
            )
            st.altair_chart(
                probability_vs_threshold_chart(fake_probability, threshold),
                width="stretch",
            )

            st.markdown("**How this account compares to the training average**")
            st.altair_chart(
                relative_feature_chart(X_scaled),
                width="stretch",
            )
            st.caption(
                "Bars to the right are above the average account in the "
                "training data, bars to the left are below it."
            )

            with st.expander("View input data"):
                st.dataframe(X, width="stretch")

        except FileNotFoundError:
            st.error(
                "Model files were not found. Run train_real.py first "
                "and ensure the model and scaler are in the models folder."
            )
        except Exception as e:
            st.error(f"Prediction failed: {e}")


# --------------------------------------------------
# TAB 2: BULK CSV UPLOAD
# --------------------------------------------------
with bulk_tab:
    st.subheader("Upload Multiple Instagram Accounts")
    st.write(
        "Upload a CSV containing the required account features. "
        "The app will predict each row and let you download the results."
    )

    with st.expander("View required CSV format"):
        example = pd.DataFrame({
            "userFollowerCount": [100, 2500],
            "userFollowingCount": [900, 300],
            "userBiographyLength": [20, 150],
            "userMediaCount": [2, 120],
            "usernameDigitCount": [5, 0],
            "usernameLength": [14, 9],
        })
        st.dataframe(example, width="stretch")

        st.download_button(
            "⬇️ Download Sample CSV",
            data=example.to_csv(index=False).encode("utf-8"),
            file_name="sample_instagram_accounts.csv",
            mime="text/csv",
            width="stretch",
        )

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        key="bulk_csv_upload",
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            if df.empty:
                st.warning("The uploaded CSV contains no rows.")
            else:
                missing = [
                    col for col in FEATURE_COLUMNS
                    if col not in df.columns
                ]

                if missing:
                    st.error(
                        f"Missing required columns: {', '.join(missing)}"
                    )
                    st.info(
                        "Required columns: "
                        + ", ".join(FEATURE_COLUMNS)
                    )
                else:
                    st.write(f"**Accounts loaded:** {len(df)}")
                    st.dataframe(
                        df[FEATURE_COLUMNS],
                        width="stretch",
                    )

                    if st.button(
                        "🚀 Run Bulk Detection",
                        key="bulk_detect",
                        width="stretch",
                    ):
                        model, scaler, threshold = load_artifacts()

                        X = df[FEATURE_COLUMNS].apply(
                            pd.to_numeric, errors="coerce"
                        )

                        invalid_rows = X.isna().any(axis=1)

                        if invalid_rows.any():
                            st.error(
                                f"{int(invalid_rows.sum())} row(s) contain "
                                "missing or non-numeric feature values. "
                                "Correct the CSV and upload it again."
                            )
                        else:
                            if (X < 0).any().any():
                                st.error(
                                    "Feature values cannot be negative. "
                                    "Correct the CSV and upload it again."
                                )
                            else:
                                probs = model.predict_proba(
                                    scaler.transform(X)
                                )[:, 1]

                                results = df.copy()
                                results["fake_probability"] = probs
                                results["non_fake_probability"] = 1 - probs
                                results["prediction"] = [
                                    "Fake" if p >= threshold
                                    else "Non-Fake"
                                    for p in probs
                                ]

                                st.divider()
                                st.subheader("Bulk Prediction Results")

                                c1, c2, c3 = st.columns(3)
                                c1.metric(
                                    "Total Accounts", len(results)
                                )
                                c2.metric(
                                    "Suspicious / Likely Fake",
                                    int((results["prediction"] == "Fake").sum()),
                                )
                                c3.metric(
                                    "Likely Non-Fake",
                                    int((results["prediction"] == "Non-Fake").sum()),
                                )

                                st.markdown(
                                    "**Distribution of fake probabilities** "
                                    "(dashed line = threshold)"
                                )
                                st.altair_chart(
                                    probability_distribution_chart(
                                        probs, threshold
                                    ),
                                    width="stretch",
                                )

                                st.dataframe(
                                    results,
                                    width="stretch",
                                )

                                csv_data = results.to_csv(
                                    index=False
                                ).encode("utf-8")

                                st.download_button(
                                    "⬇️ Download Results CSV",
                                    data=csv_data,
                                    file_name="fake_account_predictions.csv",
                                    mime="text/csv",
                                    width="stretch",
                                )

        except pd.errors.EmptyDataError:
            st.error("The uploaded CSV is empty.")
        except FileNotFoundError:
            st.error(
                "Model files were not found. Run train_real.py first "
                "and ensure the model and scaler are in the models folder."
            )
        except Exception as e:
            st.error(f"Bulk prediction failed: {e}")


st.divider()
st.caption(
    "Educational ML project • Logistic Regression • "
    "Threshold-based classification"
)