import torch
from transformers import pipeline
from langdetect import detect
from deep_translator import GoogleTranslator 

# Load the pre-trained emotion detection model
emotion_pipeline = pipeline("text-classification", model="./models/emotion_model")

def translate_to_english(text):
    """Translate text to English using deep_translator."""
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except Exception as e:
        print(f"Translation error: {e}")
        return text

def detect_emotion(text):
    """Detect emotion using a pre-trained model after translating non-English text."""
    try:
        # Detect language and translate if necessary
        language = detect(text)
        if language != 'en':
            text = translate_to_english(text)
        
        # Use the pre-trained emotion detection model
        result = emotion_pipeline(text)
        
        # Return both emotion and its probability (confidence score)
        emotion = result[0]['label']
        confidence = result[0]['score']
        
        return emotion, confidence
    except Exception as e:
        print(f"Error during emotion detection: {e}")
        return "Unknown", 0.0

if __name__ == "__main__":
    while True:
        text = input("Enter a sentence (or 'exit' to quit): ")
        if text.lower() == 'exit':
            break
        emotion, confidence = detect_emotion(text)
        print(f"Detected Emotion: {emotion} with Confidence: {confidence:.2f}")
