
from pathlib import Path
from datetime import datetime
from io import BytesIO

import joblib
import pandas as pd
import streamlit as st


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "sql_injection_model_v4.pkl"

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
MAX_ROWS = 10000

st.set_page_config(
    page_title="SQL Injection Detector",
    page_icon="🛡️",
    layout="wide",
)


# --------------------------------------------------
# Load trained V4 model
# --------------------------------------------------

@st.cache_resource
def load_model():
    saved = joblib.load(MODEL_PATH)
    return saved["features"], saved["model"], saved["labels"]


st.title("🛡️ SQL Injection Detector")
st.caption(
    "Machine learning-based analysis using TF-IDF and Logistic Regression"
)

if not MODEL_PATH.exists():
    st.error("V4 model not found. Train it using src/train_v4.py.")
    st.stop()

try:
    features, model, labels = load_model()
except Exception as exc:
    st.error(f"Unable to load model: {exc}")
    st.stop()


# --------------------------------------------------
# Prediction utility
# --------------------------------------------------

def analyze_queries(queries):
    cleaned = [str(q).strip() for q in queries]
    transformed = features.transform(cleaned)

    predictions = model.predict(transformed)
    probabilities = model.predict_proba(transformed)
    classes = list(model.classes_)

    results = []

    for query, prediction, probs in zip(
        cleaned, predictions, probabilities
    ):
        prediction = int(prediction)
        class_index = classes.index(prediction)
        confidence = float(probs[class_index]) * 100

        results.append({
            "Input": query,
            "Prediction": labels[prediction],
            "Confidence (%)": round(confidence, 2),
            "Status": (
                "Flagged"
                if prediction == 1
                else "Not flagged"
            ),
        })

    return pd.DataFrame(results)


def read_uploaded_file(uploaded_file):
    raw = uploaded_file.getvalue()

    if len(raw) > MAX_FILE_SIZE:
        raise ValueError("File exceeds the 10 MB size limit.")

    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix == ".txt":
        text = raw.decode("utf-8-sig")
        lines = [line.strip() for line in text.splitlines()]
        lines = [line for line in lines if line]

        if not lines:
            raise ValueError("The text file contains no queries.")

        if len(lines) > MAX_ROWS:
            raise ValueError(f"Maximum {MAX_ROWS:,} queries per file.")

        return pd.DataFrame({"Input": lines})

    if suffix == ".csv":
        df = pd.read_csv(BytesIO(raw), encoding="utf-8-sig")

        if df.empty or len(df.columns) == 0:
            raise ValueError("The CSV file is empty.")

        if len(df) > MAX_ROWS:
            raise ValueError(f"Maximum {MAX_ROWS:,} rows per file.")

        if len(df.columns) > 100:
            raise ValueError("The CSV has too many columns.")

        if df.columns.duplicated().any():
            raise ValueError("The CSV contains duplicate column names.")

        return df

    raise ValueError("Upload a CSV or TXT file.")


def make_text_report(results):
    total = len(results)
    flagged = int((results["Prediction"] == labels[1]).sum())
    normal = total - flagged

    generated = datetime.now().astimezone().strftime(
        "%Y-%m-%d %H:%M:%S %Z"
    )

    lines = [
        "SQL INJECTION DETECTOR — ANALYSIS REPORT",
        "=" * 48,
        f"Generated: {generated}",
        "Model: SQL Injection Detector V4",
        "Method: TF-IDF + Logistic Regression",
        "",
        "SUMMARY",
        "-" * 48,
        f"Total inputs analyzed: {total}",
        f"Flagged as SQL injection: {flagged}",
        f"Classified as normal: {normal}",
        f"Flagged percentage: {flagged / total * 100:.2f}%",
        "",
        "FINDINGS",
        "-" * 48,
    ]

    for index, row in results.iterrows():
        lines.extend([
            f"Input #{index + 1}: {row['Input']}",
            f"Prediction: {row['Prediction']}",
            f"Model confidence: {row['Confidence (%)']:.2f}%",
            f"Status: {row['Status']}",
            "",
        ])

    lines.extend([
        "LIMITATIONS",
        "-" * 48,
        "Predictions are based on a machine learning text classifier.",
        "Confidence is not a guarantee of correctness.",
        "A normal prediction does not prove an input is safe.",
        "A flagged prediction does not prove malicious intent.",
        "Use established security controls, including parameterized queries.",
        "",
    ])

    return "\n".join(lines)


# --------------------------------------------------
# Single query analysis
# --------------------------------------------------

single_tab, batch_tab, about_tab = st.tabs([
    "🔎 Single Query",
    "📂 File Analysis",
    "ℹ️ About",
])

with single_tab:
    st.subheader("Analyze an individual input")

    user_input = st.text_area(
        "Enter a query or text",
        placeholder="Example: ' OR '1'='1",
        height=120,
        key="single_query",
    )

    if st.button("Analyze Query", type="primary"):
        if not user_input.strip():
            st.warning("Enter some text first.")
        else:
            result = analyze_queries([user_input])
            row = result.iloc[0]

            if row["Prediction"] == labels[1]:
                st.error("🚨 Potential SQL Injection Detected")
            else:
                st.success("✅ Classified as Normal Input")

            col1, col2 = st.columns(2)
            col1.metric("Prediction", row["Prediction"])
            col2.metric(
                "Model confidence",
                f"{row['Confidence (%)']:.2f}%",
            )

            with st.expander("View detailed result"):
                st.json(row.to_dict())


# --------------------------------------------------
# Batch file analysis
# --------------------------------------------------

with batch_tab:
    st.subheader("Upload a query dataset")
    st.write(
        "Upload a CSV with a query column or a TXT file "
        "containing one query per line."
    )

    uploaded = st.file_uploader(
        "Choose a CSV or TXT file",
        type=["csv", "txt"],
        help="Maximum 10 MB and 10,000 queries per file.",
        key="batch_upload",
    )

    if uploaded is not None:
        try:
            source_df = read_uploaded_file(uploaded)

            if uploaded.name.lower().endswith(".txt"):
                selected_column = "Input"
                st.info(
                    f"Loaded {len(source_df):,} non-empty lines."
                )
            else:
                suggested = next(
                    (
                        col for col in source_df.columns
                        if str(col).strip().lower()
                        in {
                            "sentence", "query", "sql",
                            "text", "input", "payload"
                        }
                    ),
                    source_df.columns[0],
                )

                selected_column = st.selectbox(
                    "Select the column containing queries",
                    options=list(source_df.columns),
                    index=list(source_df.columns).index(suggested),
                )

            if st.button("Analyze Uploaded File", type="primary"):
                source_queries = (
                    source_df[selected_column]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                )

                valid_mask = source_queries.ne("")
                valid_queries = source_queries[valid_mask]

                if valid_queries.empty:
                    st.warning("No non-empty queries found.")
                else:
                    with st.spinner("Analyzing queries..."):
                        results = analyze_queries(
                            valid_queries.tolist()
                        )

                    st.session_state["batch_results"] = results
                    st.session_state["batch_filename"] = uploaded.name
                    st.success(
                        f"Analysis complete: {len(results):,} inputs."
                    )

        except Exception as exc:
            st.error(f"Could not process file: {exc}")

    results = st.session_state.get("batch_results")

    if results is not None:
        st.divider()
        st.subheader("📊 Analysis Summary")

        total = len(results)
        flagged = int((results["Prediction"] == labels[1]).sum())
        normal = total - flagged

        c1, c2, c3 = st.columns(3)
        c1.metric("Total analyzed", f"{total:,}")
        c2.metric("Potential SQL injections", f"{flagged:,}")
        c3.metric("Classified as normal", f"{normal:,}")

        st.caption(
            "Flagged means the model predicted SQL Injection. "
            "It is not a confirmed security incident."
        )

        st.subheader("Detailed Findings")

        filter_choice = st.selectbox(
            "Filter results",
            [
                "All results",
                "Flagged only",
                "Normal only",
            ],
        )

        if filter_choice == "Flagged only":
            display_df = results[
                results["Prediction"] == labels[1]
            ]
        elif filter_choice == "Normal only":
            display_df = results[
                results["Prediction"] == labels[0]
            ]
        else:
            display_df = results

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Download Reports")

        csv_bytes = results.to_csv(index=False).encode("utf-8-sig")
        report_text = make_text_report(results)

        col1, col2 = st.columns(2)

        with col1:
            st.download_button(
                "⬇️ Download Results (CSV)",
                data=csv_bytes,
                file_name="sql_injection_findings.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with col2:
            st.download_button(
                "⬇️ Download Summary Report (TXT)",
                data=report_text,
                file_name="sql_injection_report.txt",
                mime="text/plain",
                use_container_width=True,
            )


# --------------------------------------------------
# About
# --------------------------------------------------

with about_tab:
    st.subheader("About this detector")

    st.markdown("""
    **Model:** V4

    **Feature extraction:** Word-level and character-level TF-IDF

    **Classifier:** Logistic Regression

    **Input support:** Individual text, CSV and TXT files

    **Reporting:** Per-input predictions, confidence scores,
    summary counts, downloadable CSV results and TXT report.

    ### Important limitations

    - This is a text classifier, not a database firewall.
    - It does not execute uploaded queries.
    - It can miss novel attacks or flag legitimate SQL.
    - Confidence scores are not guarantees of correctness.
    - Use parameterized queries and prepared statements
      to prevent SQL injection in real applications.
    """)
