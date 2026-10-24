import re
import time

import streamlit as st

from main import FALLBACK_MESSAGE, MATCH_THRESHOLD, find_best_match

st.set_page_config(
    page_title="E-commerce FAQ Bot",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

SUGGESTIONS = [
    "Where is my order?",
    "How long does delivery take?",
    "How can I return my product?",
    "Can I cancel my order?",
]

INTENT_LABELS = {
    "order_tracking": "Order tracking",
    "delivery": "Delivery",
    "return": "Returns & refunds",
    "cancellation": "Cancellation",
}

STYLES = """
<style>
    .block-container {
        max-width: 54rem;
        padding-top: 2.5rem;
        padding-bottom: 6rem;
    }
    h1 {
        letter-spacing: -0.02em;
        font-weight: 700;
    }
    .eyebrow {
        text-transform: uppercase;
        letter-spacing: 0.09em;
        font-size: 0.7rem;
        font-weight: 600;
        color: #60a5fa;
        margin: 0 0 0.35rem;
    }
    .hero {
        border: 1px solid #1f2937;
        border-radius: 16px;
        padding: 1.5rem 1.75rem;
        background: #0b1120;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.35);
        margin-bottom: 1.25rem;
    }
    .hero h2 {
        margin: 0 0 0.5rem;
        font-size: 1.4rem;
        letter-spacing: -0.01em;
        color: #f8fafc;
    }
    .hero p {
        margin: 0;
        color: #cbd5e1;
        line-height: 1.55;
    }
    [data-testid="stChatMessage"] {
        background: #0b1120;
        border: 1px solid #1f2937;
        border-radius: 14px;
        padding: 0.6rem 1rem;
    }
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] strong,
    [data-testid="stChatMessage"] em,
    [data-testid="stChatMessage"] a,
    [data-testid="stChatMessage"] code,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
        color: #f1f5f9;
    }
    .meta {
        color: #94a3b8;
        font-size: 0.78rem;
        letter-spacing: 0.01em;
        margin-top: 0.4rem;
    }
    .meta.miss {
        color: #fbbf24;
    }
    [data-testid="stChatInput"] {
        border-radius: 999px;
    }
    .stButton > button {
        border-radius: 10px;
    }
    [data-testid="stSidebar"] {
        border-right: 1px solid #1f2937;
    }
</style>
"""


def inject_styles():
    st.markdown(STYLES, unsafe_allow_html=True)


def init_state():
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("pending_question", None)


def reset_chat():
    st.session_state.messages = []
    st.session_state.pending_question = None
    st.rerun()


def build_reply(question):
    answer, score, intent = find_best_match(question)

    if answer is None or score < MATCH_THRESHOLD:
        return {
            "role": "assistant",
            "content": FALLBACK_MESSAGE,
            "intent": None,
            "score": score,
        }

    return {"role": "assistant", "content": answer, "intent": intent, "score": score}


def stream_text(text):
    for chunk in re.findall(r"\S+\s*", text):
        yield chunk
        time.sleep(0.012)


def render_match_meta(message):
    intent = message.get("intent")
    score = message.get("score") or 0.0

    if intent is None:
        st.markdown(
            '<div class="meta miss">No confident match — try rephrasing your question.</div>',
            unsafe_allow_html=True,
        )
        return

    label = INTENT_LABELS.get(intent, intent)
    st.markdown(
        f'<div class="meta">{label} &middot; {round(score * 100)}% match</div>',
        unsafe_allow_html=True,
    )


def render_bubble(message):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            render_match_meta(message)


def render_header():
    st.markdown('<p class="eyebrow">Support assistant</p>', unsafe_allow_html=True)
    st.title("E-commerce FAQ Bot")
    st.caption("Ask about order tracking, delivery times, returns, or cancellations.")
    st.divider()


def render_welcome():
    st.markdown(
        """
        <div class="hero">
            <h2>How can I help today?</h2>
            <p>Pick one of the suggestions below or type your own question.
            Answers are matched against a curated FAQ set, so keep it short and specific.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    columns = st.columns(2)
    for index, suggestion in enumerate(SUGGESTIONS):
        with columns[index % 2]:
            if st.button(
                suggestion,
                key=f"suggestion_{index}",
                use_container_width=True,
            ):
                st.session_state.pending_question = suggestion


def render_sidebar():
    with st.sidebar:
        st.markdown("#### Assistant")
        st.caption(
            "A lightweight FAQ bot for storefront support. Topics covered: "
            "order tracking, delivery, returns, and cancellations."
        )

        st.divider()
        st.markdown("**How matching works**")
        st.caption(
            "Your question and every FAQ question are lowercased, stripped of "
            "stopwords and punctuation, and lemmatized. The best pair is scored "
            "with Jaccard similarity, and answers below 40% match are declined."
        )

        st.divider()
        st.button(
            "Clear chat",
            on_click=reset_chat,
            use_container_width=True,
            type="primary",
        )


def handle_question(question):
    question = question.strip()

    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    reply = build_reply(question)

    with st.chat_message("assistant"):
        st.write_stream(stream_text(reply["content"]))
        render_match_meta(reply)

    st.session_state.messages.append(reply)


def main():
    inject_styles()
    init_state()
    render_sidebar()
    render_header()

    if st.session_state.messages:
        for message in st.session_state.messages:
            render_bubble(message)
    else:
        render_welcome()

    typed = st.chat_input("Ask your question...")
    pending = st.session_state.pending_question
    st.session_state.pending_question = None

    question = typed or pending

    if question:
        handle_question(question)
        st.rerun()


if __name__ == "__main__":
    main()
