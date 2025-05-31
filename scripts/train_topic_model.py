import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from joblib import dump
import nltk
from langdetect import detect
from deep_translator import GoogleTranslator
from transformers import pipeline

# Ensure NLTK tokenizer is downloaded
nltk.download('punkt')

# Load pre-trained topic detection model
topic_pipeline = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# List of predefined topics
TOPIC_CANDIDATES = [
    "sports", "technology", "education", "entertainment", "health", "finance", "travel", "politics", "science"
]

def translate_to_english(text):
    """Translate text to English using deep-translator."""
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except Exception as e:
        print(f"Translation error: {e}")
        return text 

def detect_topic(text):
    """Detect topic using a pre-trained model after translating non-English text."""
    try:
        # Detect language and translate if necessary
        language = detect(text)
        if language != 'en':
            print(f"Detected language: {language}. Translating to English...")
            text = translate_to_english(text)
        
        # Use the pre-trained topic detection model
        result = topic_pipeline(text, TOPIC_CANDIDATES)
        return result['labels'][0]  # Return the most relevant topic
    except Exception as e:
        print(f"Error during topic detection: {e}")
        return "Unknown"

if __name__ == "__main__":
    while True:
        text = input("Enter a sentence (or 'exit' to quit): ")
        if text.lower() == 'exit':
            break
        topic = detect_topic(text)
        print(f"Detected Topic: {topic}")
