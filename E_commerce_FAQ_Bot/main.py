import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import json

nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("stopwords")

STOPWORDS = set(stopwords.words("english"))
PUNCT = set(string.punctuation)
lemmatizer = WordNetLemmatizer()

MATCH_THRESHOLD = 0.4
FALLBACK_MESSAGE = "Sorry, I don't understand that question yet."

with open("faq.json", "r", encoding="utf-8") as file:
    faq_data = json.load(file)


def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t not in STOPWORDS]
    tokens = [
        t.strip(string.punctuation)
        for t in tokens
        if not all(c in PUNCT for c in t)
    ]
    tokens = [t for t in tokens if t]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return tokens


def similarity(words1, words2):
    words1 = set(words1)
    words2 = set(words2)

    intersection = words1 & words2
    union = words1 | words2

    return len(intersection) / len(union)


def find_best_match(user_question):
    user_words = preprocess(user_question)

    best_score = 0
    best_item = None

    for item in faq_data:
        for question in item["questions"]:
            question_words = preprocess(question)
            score = similarity(user_words, question_words)

            if score > best_score:
                best_score = score
                best_item = item

    if best_item is None:
        return None, 0.0, None

    return best_item["answer"], best_score, best_item["intent"]


def get_response(user_question):
    answer, score, _ = find_best_match(user_question)

    if answer is not None and score >= MATCH_THRESHOLD:
        return answer

    return FALLBACK_MESSAGE


def chat():
    while True:
        user_question = input("You: ")

        if user_question.strip().lower() == "exit":
            print("Bot: Goodbye!")
            break

        if not user_question.strip():
            continue

        response = get_response(user_question)
        print("Bot:", response)


if __name__ == "__main__":
    chat()