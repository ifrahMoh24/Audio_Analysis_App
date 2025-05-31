class SentimentAnalyzer:
    def analyze(self, text):
        if any(word in text.lower() for word in ["happy", "good", "great"]):
            return "Positive"
        elif any(word in text.lower() for word in ["sad", "bad", "terrible"]):
            return "Negative"
        else:
            return "Neutral"
