# E-Commerce FAQ Bot

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)](https://streamlit.io/)
[![NLTK](https://img.shields.io/badge/NLTK-Enabled-green.svg)](https://www.nltk.org/)

An intelligent FAQ chatbot designed to handle common e-commerce customer support queries. It provides instant, accurate responses to questions about order tracking, delivery, returns, cancellations, and more through both a CLI and an interactive web interface.

## 🚀 Features

- **Dual Interface**: Choose between a terminal-based chat (`main.py`) or a sleek Streamlit web UI (`app.py`)
- **Smart Matching**: Uses Jaccard similarity with NLTK-powered text preprocessing for accurate intent matching
- **Natural Language Processing**: Handles lowercasing, stopword removal, punctuation cleanup, and lemmatization
- **Confidence Scoring**: Displays matched intent and confidence score for transparency
- **Interactive UI**: Streamlit interface with suggested questions, word-by-word streaming, and conversation reset
- **Extensible Knowledge Base**: Easily add or modify FAQs via `faq.json`
- **Graceful Fallback**: Returns a helpful message when no confident match is found

## 🛠️ Tech Stack

- **Backend**: Python 3.11+
- **NLP**: NLTK (tokenization, lemmatization, stopwords)
- **Web UI**: Streamlit
- **Data**: JSON-based FAQ knowledge base

## 📋 Prerequisites

- Python 3.11 or higher
- pip (Python package manager)
- Internet connection for first-time NLTK corpus downloads

## ⚡ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/KrishnansuMohapatra/ML_projects.git
   cd ML_projects/E_commerce_FAQ_Bot
   ```

2. **Create and activate a virtual environment**
   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate
   
   # macOS/Linux
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 Usage

### Terminal Chat
```bash
python main.py
```
Type your question and press Enter. Type `exit` to quit the conversation.

### Web Application
```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`. Enjoy an interactive chat experience with suggested questions and real-time responses.

## 🏗️ Project Structure

```plaintext
E_commerce_FAQ_Bot/
├── app.py           # Streamlit web interface
├── main.py          # Core logic, preprocessing, similarity matching & CLI
├── faq.json         # FAQ knowledge base (intents, questions & answers)
├── requirements.txt # Project dependencies
├── pyproject.toml   # Project configuration
├── uv.lock          # Dependency lock file
└── README.md        # Project documentation
```

## 🧠 How It Works

1. **Load Knowledge Base**: `faq.json` is loaded at startup containing intents, sample questions, and answers.
2. **Preprocessing**: Both user queries and FAQ questions undergo tokenization, lowercasing, stopword removal, punctuation cleaning, and lemmatization via NLTK.
3. **Similarity Calculation**: Jaccard similarity is computed as `|A ∩ B| / |A ∪ B|` between token sets.
4. **Response Generation**: If the highest similarity score ≥ 0.4, the corresponding answer is returned with its intent and confidence. Otherwise, a fallback message is displayed.

## ✏️ Customizing the Knowledge Base

To add or update FAQs, edit `faq.json`:

```json
{
  "intent": "refunds",
  "questions": [
    "When will I get my refund?",
    "How long does a refund take?",
    "When can I expect my money back?"
  ],
  "answer": "Refunds are processed to the original payment method within 5-7 business days."
}
```

**Tips for better accuracy:**
- Add multiple paraphrased questions per intent
- Use natural, customer-like phrasing
- Keep answers clear and concise
- Restart the application after making changes

## 📝 Notes

- **First Run**: NLTK will automatically download required corpora (`punkt_tab`, `wordnet`, `omw-1.4`, `stopwords`). A stable internet connection is required initially.
- **Working Directory**: Ensure you run commands from the project root directory.
- **Similarity Threshold**: Default threshold is 0.4 - adjust in `main.py` if needed for your use case.

## 📄 License

This project is proprietary. All rights reserved. No part of this codebase may be reproduced, distributed, or used without explicit permission.
