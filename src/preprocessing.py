import re
import nltk
from nltk.corpus import stopwords

# Load stopwords once at module initialization
try:
    stop_words = set(stopwords.words("german") + stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    stop_words = set(stopwords.words("german") + stopwords.words("english"))


def clean_text(text: str) -> str:
    """
    Clean support ticket text by lowercasing, removing punctuation,
    and filtering out German and English stop-words.
    """
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove punctuation and special characters
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # Remove stop-words
    words = [word for word in text.split() if word not in stop_words]

    return " ".join(words)
