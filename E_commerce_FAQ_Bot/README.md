# E-commerce FAQ Bot

A small e-commerce support assistant that answers common questions about order
tracking, delivery, returns, and cancellations. It ships with two front ends: a
terminal chat and a Streamlit web chat.

Answers are matched against `faq.json` with a Jaccard-similarity score over
NLTK-preprocessed tokens (lowercasing, stopword removal, punctuation cleanup,
lemmatization). If the best match scores below the threshold, the bot falls back
to a generic "I don't understand" message.

## Requirements

- Python 3.11+
- Dependencies in `requirements.txt`

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
```

## Run

Terminal chat:

```bash
python main.py
```

Type a question, or `exit` to quit.

Streamlit UI:

```bash
streamlit run app.py
```

The web UI opens at `http://localhost:8501`. It shows a welcome panel with
one-click suggestions on first load, streams answers word by word, tags each
answer with its matched intent and confidence, and resets the conversation via
**Clear chat** in the sidebar.

## Project structure

```
E_commerce_Faq_bot/
├── main.py            # preprocessing, similarity, CLI chat
├── app.py             # Streamlit UI
├── faq.json           # intents, sample questions, and answers
├── requirements.txt
├── README.md
└── .gitignore
```

## How it works

1. `faq.json` is loaded once at startup.
2. The user question and every FAQ question are preprocessed into token sets.
3. Each pair is scored with Jaccard similarity: `|A ∩ B| / |A ∪ B|`.
4. The highest-scoring answer is returned when the score is at least `0.4`;
   otherwise the fallback message is used.

## Adding or editing FAQs

Append an object to the array in `faq.json`:

```json
{
    "intent": "refunds",
    "questions": [
        "When will I get my refund?",
        "How do refunds work?"
    ],
    "answer": "Refunds are issued to the original payment method within 5 business days."
}
```

Restart the app after editing the file. Add several paraphrases per intent to
improve match coverage.

## Notes

- NLTK corpora (`punkt_tab`, `wordnet`, `omw-1.4`, `stopwords`) download on
  first run; the first launch needs a network connection.
- `main.py` is both the importable module and the CLI entry point, so `app.py`
  must be started from the project root.

## License

No license. All rights reserved. This code is proprietary and is not open
source — no permission is granted to use, copy, modify, or distribute it.
