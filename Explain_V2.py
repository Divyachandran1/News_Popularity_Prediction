import re
from textblob import TextBlob

URGENCY_WORDS = [
    "breaking", "urgent", "alert", "crisis",
    "attack", "war", "election", "death",
    "explosion", "emergency"
]

def explain_article(text):

    explanation = {}

    text_lower = text.lower()

    explanation["text_length"] = len(text)

    sentiment = TextBlob(text).sentiment.polarity
    explanation["sentiment_score"] = round(sentiment, 3)

    urgency_count = sum(word in text_lower for word in URGENCY_WORDS)
    explanation["urgency_terms"] = urgency_count

    words = re.findall(r'\w+', text_lower)
    explanation["unique_words"] = len(set(words))

    # Add reasoning summary
    explanation["reasoning"] = (
        f"This article contains {urgency_count} urgency-related terms, "
        f"sentiment score of {round(sentiment,3)}, "
        f"and {len(set(words))} unique words. "
        "Higher urgency and emotional intensity may increase popularity."
    )

    return explanation

# Test
if __name__ == "__main__":
    sample_text = "Breaking news: Major election results announced today."
    print(explain_article(sample_text))
