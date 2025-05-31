import pytest
from sentiment_analyzer import SentimentAnalyzer

@pytest.fixture
def analyzer():
    return SentimentAnalyzer()

def test_positive_sentiment(analyzer):
    result = analyzer.analyze("I feel happy and great!")
    assert result == "Positive", "Positive sentiment test failed."

def test_negative_sentiment(analyzer):
    result = analyzer.analyze("I feel sad and terrible.")
    assert result == "Negative", "Negative sentiment test failed."

def test_neutral_sentiment(analyzer):
    result = analyzer.analyze("It is an average day.")
    assert result == "Neutral", "Neutral sentiment test failed."
