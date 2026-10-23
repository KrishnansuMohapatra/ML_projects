"""
app.py — Fake News Detector  (Streamlit Frontend)
==================================================
Run:  streamlit run app.py
"""

import streamlit as st
import joblib
import os

# ── config ───────────────────────────────────────────────────────────
MODEL_PATH = "fake_news_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide",
)


# ── load artifacts (cached) ─────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        return None, None
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


model, vectorizer = load_artifacts()


# ── demo articles ────────────────────────────────────────────────────
DEMO_ARTICLES = {
    "— Select an example —": {
        "title": "",
        "body": "",
        "expected": None,
    },
    "✅ Easy Real — Jobless Claims": {
        "title": "U.S. jobless claims fall to near 45-year low",
        "body": (
            "WASHINGTON (Reuters) - The number of Americans filing for "
            "unemployment benefits fell to its lowest level in nearly 45 years "
            "last week, pointing to sustained labor market strength that could "
            "encourage the Federal Reserve to raise interest rates in March.\n\n"
            "Initial claims for state unemployment benefits dropped 22,000 to a "
            "seasonally adjusted 216,000 for the week ended Feb. 10, the Labor "
            "Department said on Thursday. That was the lowest reading since "
            "January 1973 and well below the 230,000 that economists polled by "
            "Reuters had forecast.\n\n"
            "The four-week moving average of initial claims, considered a better "
            "measure of labor market trends as it irons out week-to-week "
            "volatility, fell 7,250 to 226,000 last week, the lowest level since "
            "March 1973.\n\n"
            "The labor market is near full employment, with the jobless rate at "
            "a 17-year low of 4.1 percent."
        ),
        "expected": "Real",
    },
    "✅ Easy Real — European Stocks": {
        "title": "European stocks close higher as earnings boost sentiment",
        "body": (
            "LONDON (Reuters) - European shares closed higher on Tuesday, lifted "
            "by strong corporate earnings and positive economic data from Germany, "
            "while the euro held steady against the dollar.\n\n"
            "The pan-European STOXX 600 index rose 0.6 percent to 384.52 points, "
            "its highest close in two weeks. Germany's DAX gained 0.8 percent "
            "after data showed German business morale rose to a record high in "
            "February.\n\n"
            "Banking stocks were among the biggest gainers, with the sector index "
            "up 1.2 percent. Deutsche Bank rose 2.3 percent after reporting a "
            "narrower-than-expected quarterly loss.\n\n"
            '"This is a market that is being driven by fundamentals," said Michael '
            "Hewson, chief market analyst at CMC Markets in London."
        ),
        "expected": "Real",
    },
    "🚨 Easy Fake — Wiretapping": {
        "title": "BREAKING: Obama Caught Wiretapping Trump Tower During Election!",
        "body": (
            "IT'S FINALLY BEEN CONFIRMED! The mainstream media doesn't want you "
            "to see this, but the truth is coming out!\n\n"
            "President Obama personally ordered the wiretapping of Trump Tower "
            "during the 2016 election campaign, according to explosive new reports "
            "that the fake news media is desperately trying to suppress!\n\n"
            "Sources inside the intelligence community have confirmed that Obama "
            'was "furious" about Trump\'s rising poll numbers and took matters '
            "into his own hands. This is the biggest scandal in American history "
            "and the Democrats are TERRIFIED!\n\n"
            "Share this before it gets taken down! The liberal media will never "
            "report this! We are being silenced but the American people deserve "
            "to know the TRUTH!\n\n"
            "Wake up, America! The deep state is real and they are coming for "
            "our freedoms!"
        ),
        "expected": "Fake",
    },
    "🚨 Easy Fake — Email Server": {
        "title": "SHOCKING: Hillary Clinton's Secret Email Server Found in Abandoned Warehouse!",
        "body": (
            "YOU WON'T BELIEVE THIS! FBI agents just raided an abandoned warehouse "
            "in Virginia and discovered Hillary Clinton's MISSING email server "
            "hidden behind a false wall!\n\n"
            "The server reportedly contains over 650,000 classified emails that "
            "Hillary thought she had destroyed! This is the smoking gun that "
            "proves she has been lying to the American people for YEARS!\n\n"
            '"Everything we suspected is true and it\'s even worse than we '
            'imagined," said a senior FBI source who asked to remain anonymous. '
            '"The level of corruption is absolutely staggering."\n\n'
            "The mainstream media is refusing to cover this story because they "
            "are in Hillary's pocket! Don't let them bury this!\n\n"
            "SHARE SHARE SHARE before the deep state takes this down! The "
            "revolution is HERE and nothing can stop it! God bless America!"
        ),
        "expected": "Fake",
    },
    "⚠️ Tricky Fake — Fed Rate Hike": {
        "title": "Federal Reserve signals potential rate hike amid inflation concerns",
        "body": (
            "WASHINGTON — Federal Reserve officials indicated on Wednesday that "
            "they may accelerate the pace of interest rate increases in the coming "
            "months, citing growing concerns about persistent inflation pressures "
            "across the U.S. economy.\n\n"
            "In a statement released after their two-day policy meeting, members "
            "of the Federal Open Market Committee noted that core inflation has "
            "remained above the central bank's 2 percent target for the third "
            "consecutive quarter. The committee voted unanimously to hold the "
            "benchmark federal funds rate steady at its current range of 1.50 to "
            "1.75 percent.\n\n"
            '"Recent data suggests that inflationary pressures are more entrenched '
            'than previously anticipated," said Fed Chair Jerome Powell during a '
            "press conference following the announcement.\n\n"
            "Treasury yields rose sharply in response to the statement, with the "
            "10-year note climbing 8 basis points to 2.95 percent. The S&P 500 "
            "fell 0.4 percent in afternoon trading."
        ),
        "expected": "Fake",
    },
    "⚠️ Tricky Real — ECB Interview": {
        "title": "Highlights of Reuters interview with ECB President Mario Draghi",
        "body": (
            "FRANKFURT (Reuters) - Following are highlights of a Reuters interview "
            "with European Central Bank President Mario Draghi conducted on "
            "Thursday:\n\n"
            "ON MONETARY POLICY:\n"
            '"We have expressed a lot of confidence that inflation will converge '
            "to our target. The question is when, and the answer to that depends "
            'on the persistence of the underlying inflation pressures."\n\n'
            "ON THE EURO:\n"
            '"The recent volatility in the exchange rate represents a source of '
            "uncertainty which requires monitoring with regard to its possible "
            'implications for the medium-term outlook for price stability."\n\n'
            "ON ECONOMIC GROWTH:\n"
            '"The euro area economy is growing at a robust pace. The risks '
            "surrounding the growth outlook remain broadly balanced but have "
            'moved to the downside due to global factors."'
        ),
        "expected": "Real",
    },
}


# ── sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown(
        """
        This tool classifies news articles as **Real** or **Fake**
        using a Logistic Regression model trained on TF-IDF features.

        **Dataset:** Fake and Real News (Kaggle)  
        **Model:** Logistic Regression  
        **Features:** Unigrams + Bigrams (TF-IDF)  
        **Accuracy:** 98.61% (group-aware split)

        > ⚠️ This is a demonstration tool.
        > Do not rely on it for critical decisions.
        """
    )
    st.divider()
    st.caption("Trained with group-aware splitting to prevent data leakage.")


# ── main UI ──────────────────────────────────────────────────────────
st.title("📰 Fake News Detector")
st.caption("Paste a news article or choose a demo example below.")

if model is None or vectorizer is None:
    st.error(
        "Model files not found. "
        "Download the dataset, then run `python main.py` to generate "
        f"`{MODEL_PATH}` and `{VECTORIZER_PATH}`."
    )
    st.stop()

# ── demo selector ────────────────────────────────────────────────────
st.markdown("### 🧪 Demo Examples")

selected = st.selectbox(
    "Pick a pre-loaded article to test:",
    options=list(DEMO_ARTICLES.keys()),
    index=0,
)

# populate from selection
demo = DEMO_ARTICLES[selected]
default_title = demo["title"]
default_body = demo["body"]
expected_label = demo["expected"]

st.divider()

# ── input fields ─────────────────────────────────────────────────────
st.markdown("### ✏️ Article Input")

title_input = st.text_input(
    "Article Title  *(optional but improves accuracy)*",
    value=default_title,
    placeholder="e.g.  U.S. jobless claims fall to near 45-year low",
)

body_input = st.text_area(
    "Article Body",
    value=default_body,
    height=250,
    placeholder="Paste the full article text here…",
)

# ── predict ──────────────────────────────────────────────────────────
if st.button("🔍 Analyze Article", type="primary", use_container_width=True):

    combined = f"{title_input} {body_input}".strip()

    if not combined:
        st.warning("Please enter at least some article text or select a demo example.")
    else:
        # inference
        tfidf_vec = vectorizer.transform([combined])
        prob_real = model.predict_proba(tfidf_vec)[0, 1]
        prob_fake = 1 - prob_real

        is_real = prob_real >= 0.5
        confidence = max(prob_real, prob_fake)

        # ── result card ──────────────────────────────────────────────
        if is_real:
            label = "REAL"
            color = "#28a745"
            emoji = "✅"
        else:
            label = "FAKE"
            color = "#dc3545"
            emoji = "🚨"

        st.markdown(
            f"""
            <div style="
                background-color: {color}15;
                border-left: 6px solid {color};
                padding: 20px 24px;
                border-radius: 8px;
                margin: 16px 0;
            ">
                <h2 style="margin:0; color:{color};">
                    {emoji}  {label}
                </h2>
                <p style="margin:6px 0 0; font-size:1.1rem;">
                    Confidence: <strong>{confidence:.1%}</strong>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── probability breakdown ────────────────────────────────────
        st.markdown("**Probability Breakdown**")
        col1, col2, col3 = st.columns(3)
        col1.metric("Real", f"{prob_real:.1%}")
        col2.metric("Fake", f"{prob_fake:.1%}")

        if expected_label:
            correct = (expected_label == "Real" and is_real) or (
                expected_label == "Fake" and not is_real
            )
            verdict = "✅ Correct" if correct else "❌ Wrong"
            col3.metric("Expected", expected_label, delta=verdict)
        else:
            col3.metric("Expected", "N/A")

        st.progress(prob_real, text=f"P(Real) = {prob_real:.1%}")

        # ── disclaimer ───────────────────────────────────────────────
        st.caption(
            "This prediction is based on statistical patterns in the "
            "training data and may not reflect ground truth."
        )