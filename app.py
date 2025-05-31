import sys
import os
import librosa
import numpy as np
import joblib
import sounddevice as sd
import matplotlib.pyplot as plt
import speech_recognition as sr
import soundfile as sf
from pydub import AudioSegment
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLineEdit, QLabel, QVBoxLayout, QWidget, QFileDialog, QTabWidget
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
from transformers import pipeline

# Global Variables
MODELS = {}
SENTIMENT_PIPELINE = None

# Logging setup
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load models globally
def load_models():
    global MODELS, SENTIMENT_PIPELINE
    try:
        MODELS["scaler"] = joblib.load("models/scaler.joblib")
        MODELS["selector"] = joblib.load("models/selector.joblib")
        MODELS["classifier"] = joblib.load("models/speaker_classifier.joblib")
        SENTIMENT_PIPELINE = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        logger.info("Models loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading models: {e}")

# Feature extraction function
def extract_features(y, sr):
    try:
        if len(y) < 512:
            logger.warning("Audio file is very short; results may be inaccurate.")
        n_fft = min(512, len(y))
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, n_fft=n_fft)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft)
        rms = librosa.feature.rms(y=y, frame_length=n_fft)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)

        features = np.hstack([
            np.mean(mfccs, axis=1), np.std(mfccs, axis=1),
            np.mean(chroma, axis=1), np.std(chroma, axis=1),
            np.mean(spectral_contrast), np.mean(rms), np.mean(zero_crossing_rate)
        ])
        return features
    except Exception as e:
        logger.error(f"Error extracting features: {e}")
        return None

# Preprocess audio for analysis
def preprocess_audio(file_path, output_path="normalized_audio.wav"):
    try:
        y, sr = librosa.load(file_path, sr=16000)
        if len(y) < 512:
            logger.warning("Audio is too short for reliable processing.")
            return None
        y = librosa.util.normalize(y)
        sf.write(output_path, y, sr)
        return output_path
    except Exception as e:
        logger.error(f"Error preprocessing audio: {e}")
        return None

# Convert audio to WAV format
def convert_to_wav(input_file, output_file="converted_audio.wav"):
    try:
        audio = AudioSegment.from_file(input_file)
        audio.export(output_file, format="wav")
        return output_file
    except Exception as e:
        logger.error(f"Error converting audio to WAV: {e}")
        return None

# Analyze emotion from text
def analyze_emotion(transcribed_text):
    try:
        analysis = SENTIMENT_PIPELINE(transcribed_text)
        return analysis
    except Exception as e:
        logger.error(f"Error in emotion analysis: {e}")
        return None

class AudioAnalysisApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Analysis Tool")
        self.setGeometry(200, 200, 800, 600)
        self.font = QFont("Arial", 12)

        # Load models
        load_models()

        # Central Widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Layout
        main_layout = QVBoxLayout()
        self.tabs = QTabWidget()
        self.tabs.setFont(self.font)

        # Tabs
        self.general_tab = QWidget()
        self.analysis_tab = QWidget()
        self.recording_tab = QWidget()

        self.tabs.addTab(self.general_tab, "General")
        self.tabs.addTab(self.analysis_tab, "Analyze")
        self.tabs.addTab(self.recording_tab, "Record and Analyze")
        main_layout.addWidget(self.tabs)
        self.central_widget.setLayout(main_layout)

        # Initialize tabs
        self.init_general_tab()
        self.init_analysis_tab()
        self.init_recording_tab()

    def init_general_tab(self):
        layout = QVBoxLayout()
        self.file_path_input = QLineEdit(self)
        self.file_path_input.setPlaceholderText("Select an audio file...")
        layout.addWidget(self.file_path_input)

        self.upload_button = QPushButton("Upload File", self)
        self.upload_button.clicked.connect(self.upload_file)
        layout.addWidget(self.upload_button)
        self.general_tab.setLayout(layout)

    def init_analysis_tab(self):
        layout = QVBoxLayout()
        self.analyze_button = QPushButton("Analyze Uploaded Audio", self)
        self.analyze_button.clicked.connect(self.analyze_uploaded_audio)
        layout.addWidget(self.analyze_button)
        self.result_label = QLabel("Results will appear here...", self)
        layout.addWidget(self.result_label)
        self.analysis_tab.setLayout(layout)

    def init_recording_tab(self):
        layout = QVBoxLayout()
        self.record_button = QPushButton("Start Recording", self)
        self.record_button.clicked.connect(self.real_time_analysis)
        layout.addWidget(self.record_button)
        self.result_label_record = QLabel("Recording and results will appear here...", self)
        layout.addWidget(self.result_label_record)
        self.recording_tab.setLayout(layout)

    def upload_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Audio File", "", "Audio Files (*.wav *.mp3 *.ogg *.flac)")
        if file_path:
            self.file_path_input.setText(file_path)

    def analyze_uploaded_audio(self):
        try:
            file_path = self.file_path_input.text()
            if not file_path.endswith(".wav"):
                file_path = convert_to_wav(file_path)
            file_path = preprocess_audio(file_path)
            y, sr = librosa.load(file_path, sr=16000)
            features = extract_features(y, sr)
            scaler = MODELS["scaler"]
            selector = MODELS["selector"]
            classifier = MODELS["classifier"]
            features_scaled = scaler.transform([features])
            features_selected = selector.transform(features_scaled)
            speaker_prediction = classifier.predict(features_selected)[0]
            recognizer = sr.Recognizer()
            with sr.AudioFile(file_path) as source:
                audio_data = recognizer.record(source)
            transcription = recognizer.recognize_google(audio_data)
            sentiment = analyze_emotion(transcription)
            self.result_label.setText(
                f"Speaker: {speaker_prediction}\n"
                f"Transcription: {transcription}\n"
                f"Sentiment: {sentiment[0]['label']} ({sentiment[0]['score']:.2f})"
            )
        except Exception as e:
            logger.error(f"Error: {e}")

    def real_time_analysis(self):
        try:
            duration = 10  # Duration of the recording in seconds
            samplerate = 16000  # Sampling rate
            self.result_label_record.setText("Recording in progress...")

            # Record the audio
            recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype="float32")
            sd.wait()  # Wait until recording is finished
            self.result_label_record.setText("Recording completed. Analyzing...")

            # Save recorded audio to a temporary file
            output_file = "real_time_recording.wav"
            sf.write(output_file, recording, samplerate)

            # Silence removal
            y, sr = librosa.load(output_file, sr=samplerate)
            intervals = librosa.effects.split(y, top_db=20)
            non_silent_audio = np.concatenate([y[start:end] for start, end in intervals])

            if len(non_silent_audio) == 0:
                self.result_label_record.setText("Error: Audio contains only silence.")
                return

            # Save non-silent audio
            sf.write(output_file, non_silent_audio, sr)

            # Extract features
            features = extract_features(non_silent_audio, sr)
            if features is None:
                self.result_label_record.setText("Error extracting features.")
                return

            scaler = MODELS["scaler"]
            selector = MODELS["selector"]
            classifier = MODELS["classifier"]

            features_scaled = scaler.transform([features])
            features_selected = selector.transform(features_scaled)
            speaker_prediction = classifier.predict(features_selected)[0]

            recognizer = sr.Recognizer()
            with sr.AudioFile(output_file) as source:
                audio_data = recognizer.record(source)
            transcription = recognizer.recognize_google(audio_data)

            self.result_label_record.setText(
                f"Real-Time Speaker Prediction: {speaker_prediction}\nTranscription: {transcription}"
            )
        except Exception as e:
            logger.error(f"Error in real-time analysis: {e}")
            self.result_label_record.setText(f"Error: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AudioAnalysisApp()
    window.show()
    sys.exit(app.exec_())
