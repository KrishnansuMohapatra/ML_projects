
import streamlit as st
import joblib

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="SpamShield",
    page_icon="🛡️",
    layout="centered"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

    .stApp {
        background: linear-gradient(135deg, #0f172a, #111827);
    }

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 800;
        margin-top: 20px;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 16px;
        margin-bottom: 35px;
    }

    .card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 20px;
        padding: 25px;
        margin-top: 20px;
    }

    .result {
        text-align: center;
        padding: 25px;
        border-radius: 18px;
        margin-top: 25px;
        border: 1px solid rgba(255,255,255,0.12);
    }

    .spam {
        background: rgba(239, 68, 68, 0.12);
    }

    .ham {
        background: rgba(34, 197, 94, 0.12);
    }

    .result-title {
        font-size: 30px;
        font-weight: 800;
        margin-bottom: 10px;
    }

    .probability {
        font-size: 18px;
        color: #cbd5e1;
    }

    .footer {
        text-align: center;
        color: #64748b;
        font-size: 13px;
        margin-top: 40px;
    }

    textarea {
        border-radius: 14px !important;
    }

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("spam_nb_model.joblib")


try:
    model = load_model()
except Exception as e:
    st.error(
        "Model could not be loaded. Make sure "
        "`spam_nb_model.joblib` is in the same folder as this app."
    )
    st.stop()


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🛡️ SpamShield</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered SMS spam detection using TF-IDF + Naive Bayes'
    '</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# REAL-LIFE EXAMPLES
# --------------------------------------------------

EXAMPLES = {
    "spam": [
        (
            "🏦 Bank alert",
            "Dear Customer, your bank account has been suspended due to "
            "suspicious activity. Verify your identity immediately at "
            "http://secure-bank-verify.com"
        ),
        (
            "📦 Parcel fee",
            "Your parcel is awaiting delivery. Pay the £1.99 redelivery fee "
            "here: http://bit.ly/parcel-fee"
        ),
        (
            "🎁 Prize win",
            "Congratulations! You have been selected for a $500 Amazon gift "
            "card. Claim now: http://win-prize.co"
        ),
        (
            "🔒 Account locked",
            "FINAL NOTICE: Your iCloud account has been locked due to unusual "
            "activity. Verify now at http://apple-id-verify.co"
        ),
        (
            "💸 Loan offer",
            "Get a cheap loan today with no credit check! Apply now and receive "
            "cash within 24 hours. Reply STOP to opt out."
        ),
    ],
    "ham": [
        (
            "☕ Coffee",
            "Hey, are we still on for coffee tomorrow morning?"
        ),
        (
            "🏢 Work meeting",
            "Team, the project meeting has been moved to 3pm in conference room B."
        ),
        (
            "📎 Report",
            "Hi Sarah, please find the quarterly report attached. Let me know if "
            "you have any questions before tomorrow's meeting."
        ),
        (
            "🍽️ Lunch",
            "Lunch tomorrow at the usual place? My treat."
        ),
    ],
}


def use_example(text: str) -> None:
    st.session_state["message"] = text
    st.session_state["pending_check"] = True


if "message" not in st.session_state:
    st.session_state["message"] = ""


# --------------------------------------------------
# INPUT
# --------------------------------------------------

st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("📩 Check your message")

st.caption("Click a real-life email to demo instantly, or type your own.")

spam_col, ham_col = st.columns(2)

with spam_col:
    st.markdown("**🚨 Spam examples**")
    for name, text in EXAMPLES["spam"]:
        st.button(
            name,
            use_container_width=True,
            on_click=use_example,
            args=(text,),
        )

with ham_col:
    st.markdown("**✅ Ham examples**")
    for name, text in EXAMPLES["ham"]:
        st.button(
            name,
            use_container_width=True,
            on_click=use_example,
            args=(text,),
        )

message = st.text_area(
    "Enter an SMS or email message",
    placeholder="Example: Congratulations! You won a free prize. Click here to claim...",
    height=150,
    label_visibility="collapsed",
    key="message",
)

check = st.button(
    "🔍 Analyze Message",
    use_container_width=True
)

st.markdown('</div>', unsafe_allow_html=True)


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

run_check = check or st.session_state.pop("pending_check", False)

if run_check:

    if not message.strip():
        st.warning("Please enter a message first.")

    else:

        probability = float(
            model.predict_proba([message])[0, 1]
        )

        # 0.5 is the normal classification threshold
        threshold = 0.5

        if probability >= threshold:

            st.markdown(
                f"""
                <div class="result spam">
                    <div class="result-title">
                        🚨 SPAM DETECTED
                    </div>
                    <div class="probability">
                        Spam probability: <b>{probability:.2%}</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class="result ham">
                    <div class="result-title">
                        ✅ LEGITIMATE MESSAGE
                    </div>
                    <div class="probability">
                        Spam probability: <b>{probability:.2%}</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown(
    '<div class="footer">'
    'Built with Python • Scikit-learn • Streamlit'
    '</div>',
    unsafe_allow_html=True
)

