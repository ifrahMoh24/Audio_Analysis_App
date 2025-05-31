import os
import numpy as np
import librosa
import pickle
import sounddevice as sd
import soundfile as sf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
from transformers import pipeline
import speech_recognition as sr
import re

# Load pre-trained emotion and topic analysis models
emotion_analyzer = pipeline("sentiment-analysis")
topic_analyzer = pipeline("zero-shot-classification")
TOPIC_CANDIDATES = ["technology", "education", "entertainment", "health", "finance"]

# Load trained speaker model, scaler, and selector
with open("models/speaker_classifier.joblib", "rb") as model_file, \
     open("models/scaler.joblib", "rb") as scaler_file, \
     open("models/selector.joblib", "rb") as selector_file:
    speaker_model = pickle.load(model_file)
    scaler = pickle.load(scaler_file)
    selector = pickle.load(selector_file)

# Function to remove numbers from transcription text
def clean_transcription(transcription):
    return re.sub(r'\d+', '', transcription)

# Function to extract features from audio
def extract_features(y, sr):
    try:
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        rms = librosa.feature.rms(y=y)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)

        features = np.hstack([
            np.mean(mfccs, axis=1), np.std(mfccs, axis=1),
            np.mean(chroma, axis=1), np.std(chroma, axis=1),
            np.mean(spectral_contrast), np.mean(rms), np.mean(zero_crossing_rate)
        ])
        return features
    except Exception as e:
        print(f"Error in feature extraction: {e}")
        return None

# Function to create histogram
def create_histogram(y, file_path):
    try:
        plt.figure(figsize=(10, 6))
        plt.hist(y, bins=100, color='blue', alpha=0.7)
        plt.title("Amplitude Histogram")
        plt.xlabel("Amplitude")
        plt.ylabel("Frequency")
        plt.grid()
        histogram_path = file_path.replace('.wav', '_histogram.png')
        plt.savefig(histogram_path)
        plt.close()
        print(f"Histogram saved to {histogram_path}")
    except Exception as e:
        print(f"Error creating histogram: {e}")

# Function to record audio
def record_audio(file_path, duration=5, samplerate=16000):
    while True:
        print("Recording...")
        try:
            audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
            sd.wait()  # Wait for the recording to finish
            sf.write(file_path, audio_data, samplerate)
            print(f"Recording saved to {file_path}")
        except Exception as e:
            print(f"Error during recording: {e}")

        rerecord = input("Do you want to re-record? (y/n): ").strip().lower()
        if rerecord != 'y':
            break

# Function to transcribe audio
def transcribe_audio(file_path):
    recognizer = sr.Recognizer()

    try:
        # Load the audio file into the recognizer
        with sr.AudioFile(file_path) as source:
            audio = recognizer.record(source)

        # Try Google Web Speech API
        transcription = recognizer.recognize_google(audio)
        print(f"Transcription: {transcription}")

        # Count the number of words in the transcription
        word_count = len(transcription.split())
        print(f"Word Count: {word_count}")
        return transcription
    except Exception as e:
        print(f"Error in transcription: {e}")
        return None

# Function to analyze audio file
def analyze_recorded(file_path):
    print("Processing audio file for analysis...")

    try:
        # Load and process audio file
        y, sr = librosa.load(file_path, sr=16000)

        # Create histogram
        create_histogram(y, file_path)

        # Extract features from the audio
        features = extract_features(y, sr)
        if features is not None:
            # Predict speaker
            features = scaler.transform([features])
            features = selector.transform(features)
            speaker_prediction = speaker_model.predict(features)[0]
            print(f"Detected speaker: {speaker_prediction}")

            # Perform speech-to-text
            transcription = transcribe_audio(file_path)

            if transcription:
                # Clean transcription by removing numbersn
                clean_transcription_text = clean_transcription(transcription)

                # Perform emotion analysis
                emotion = emotion_analyzer(clean_transcription_text)[0]
                print(f"Emotion Analysis: {emotion}")

                # Perform topic analysis
                topics = topic_analyzer(clean_transcription_text, candidate_labels=TOPIC_CANDIDATES)
                print(f"Topic Analysis: {topics}")
        else:
            print("Failed to extract features from the audio file.")
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

# Main entry point
if __name__ == "__main__":
    output_file = "recorded_audio.wav"
    record_audio(output_file)  # Record the audio
    analyze_recorded(output_file)  # Analyze the recorded audio
