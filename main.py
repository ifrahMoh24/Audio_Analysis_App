import sys
import os
import librosa
import numpy as np
import joblib
import sounddevice as sd
import soundfile as sf
import speech_recognition as sr
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLineEdit, QLabel, QVBoxLayout, QWidget, QFileDialog, QProgressBar, QMessageBox, QSizePolicy
    ,QTabWidget
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from transformers import pipeline
from langdetect import detect
from deep_translator import GoogleTranslator
import logging
import matplotlib
matplotlib.use('Agg')  # Ensure matplotlib does not use any QT backend
from matplotlib.figure import Figure
from io import BytesIO

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load models and pipelines globally
MODELS = {}
EMOTION_MODEL = None
TOPIC_MODEL = None

def load_custom_models():
    global EMOTION_MODEL, TOPIC_MODEL
    try:
        EMOTION_MODEL = pipeline("text-classification", model="./models/emotion_model")
        TOPIC_MODEL = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        logger.info("Custom-trained models loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading custom models: {e}")
        EMOTION_MODEL = None
        TOPIC_MODEL = None

def detect_language_and_translate(text):
    detected_language = detect(text)
    if detected_language != 'en':
        try:
            return GoogleTranslator(source=detected_language, target='en').translate(text)
        except Exception as e:
            logger.error(f"Translation error from {detected_language} to English: {e}")
            return text
    return text

def detect_emotion(text):
    text = detect_language_and_translate(text)
    if not EMOTION_MODEL:
        logger.error("Emotion model is not initialized.")
        return {"Unknown": 0.0}
    try:
        result = EMOTION_MODEL(text)
        total_score = sum(res['score'] for res in result)
        emotions = {res['label']: (res['score'] / total_score) * 100 for res in result}
        return emotions
    except Exception as e:
        logger.error(f"Error during emotion detection: {e}")
        return {"Unknown": 0.0}

def detect_topic(text):
    text = detect_language_and_translate(text)
    if not TOPIC_MODEL:
        logger.error("Topic model is not initialized.")
        return {"Unknown": 0.0}
    try:
        result = TOPIC_MODEL(
            text, ["sports", "technology", "education", "entertainment", "health", "finance", "travel", "politics", "science"]
        )
        topics = {label: score * 100 for label, score in zip(result['labels'], result['scores'])}
        return topics
    except Exception as e:
        logger.error(f"Error during topic detection: {e}")
        return {"Unknown": 0.0}

def load_models():
    try:
        MODELS["scaler"] = joblib.load("models/scaler.joblib")
        MODELS["selector"] = joblib.load("models/selector.joblib")
        MODELS["classifier"] = joblib.load("models/speaker_classifier.joblib")
        logger.info("Other models loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load other models: {e}")

def translate_to_english(text):
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return text

def extract_features(y, sr, plot_histograms=False):
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

        if plot_histograms:
            feature_names = [f"MFCC_{i}" for i in range(1, 41)] + [
                "Chroma Mean", "Chroma Std", "Spectral Contrast", "RMS", "Zero Crossing Rate"
            ]
            plot_histogram(np.vstack([mfccs, chroma, spectral_contrast, rms, zero_crossing_rate]), feature_names)

        return features
    except Exception as e:
        logger.error(f"Error extracting features: {e}")
        return None

def plot_histogram(features, feature_names, title="Audio Feature Histogram"):
    num_features = features.shape[0]
    fig = Figure(figsize=(14, 8))
    for i in range(num_features):
        ax = fig.add_subplot(num_features, 1, i + 1)
        ax.hist(features[i], bins=20, color='blue', alpha=0.7)
        ax.set_title(feature_names[i])
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the layout to not overlap
    fig.suptitle(title)
    return fig  # Return the figure object instead of showing it

def preprocess_audio(file_path, output_path="normalized_audio.wav"):
    try:
        y, sr = librosa.load(file_path, sr=16000)
        y = librosa.util.normalize(y)
        sf.write(output_path, y, sr)
        logger.debug(f"Preprocessed audio saved to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error preprocessing audio: {e}")
        return None

def transcribe_audio(file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio_data = recognizer.record(source)
        try:
            transcription = recognizer.recognize_google(audio_data)
            word_count = len(transcription.split())
            logger.info("Transcription successful.")
            return transcription, word_count
        except sr.UnknownValueError:
            logger.error("Google Speech Recognition could not understand audio")
            return "Transcription not available", 0
        except sr.RequestError as e:
            logger.error(f"Could not request results from Google Speech Recognition service; {e}")
            return "Transcription not available", 0

def load_stylesheet(app):
    with open("styles.css", "r") as f:
        app.setStyleSheet(f.read())

def visualize_topic_analysis_with_histogram(self, topics, histograms):
    fig = Figure(figsize=(15, 6))
    ax1 = fig.add_subplot(121)
    labels = list(topics.keys())
    sizes = list(topics.values())
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    ax1.set_title("Topic Analysis")
    ax1.axis('equal')

    if histograms:
        ax2 = fig.add_subplot(122)
        for i, hist in enumerate(histograms):
            ax2.bar(range(len(hist)), hist, alpha=0.5, label=f"Feature {i+1}")
        ax2.set_title("Selected Histogram")
        ax2.legend()

    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    pixmap = QPixmap()
    pixmap.loadFromData(buf.getvalue())
    self.visualization_label.setPixmap(pixmap)
    buf.close()

from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLineEdit, QLabel, QVBoxLayout, QWidget, QFileDialog, QProgressBar, QTabWidget)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
import sys
import os
import librosa
import numpy as np
import joblib
import sounddevice as sd
import soundfile as sf
import speech_recognition as sr
from matplotlib.figure import Figure
from io import BytesIO

class AudioAnalysisApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Analysis Tool")
       # self.setGeometry(200, 200, 1900, 1000)  # Adjusted width to avoid overlap
        self.setMinimumSize(1438, 790)
        self.setMaximumSize(1438, 790)       
        self.initUI()
        load_custom_models()
        load_models()
        self.selected_histograms = []  # To store selected histograms for visualization

    def initUI(self):
        layout = QVBoxLayout()
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.central_widget.setLayout(layout)

        self.tabs = QTabWidget(self)
        layout.addWidget(self.tabs)

        self.input_tab = QWidget()
        self.analysis_tab = QWidget()
        self.visualization_tab = QWidget()

        self.tabs.addTab(self.input_tab, "General")
        self.tabs.addTab(self.analysis_tab, "Analysis")
        self.tabs.addTab(self.visualization_tab, "Visualization")

        self.setup_input_tab()
        self.setup_analysis_tab()
        self.setup_visualization_tab()

    def setup_input_tab(self):
        input_layout = QVBoxLayout()
        self.file_path_input = QLineEdit(self)
        self.file_path_input.setPlaceholderText("Select an audio file...")
        input_layout.addWidget(self.file_path_input)

        self.upload_button = QPushButton("Upload File", self)
        self.upload_button.clicked.connect(self.upload_file)
        input_layout.addWidget(self.upload_button)

        self.record_button = QPushButton("Start Recording", self)
        self.record_button.clicked.connect(self.real_time_analysis)
        input_layout.addWidget(self.record_button)

        self.reset_button = QPushButton("Reset", self)
        self.reset_button.clicked.connect(self.reset_app)
        input_layout.addWidget(self.reset_button)

        self.input_tab.setLayout(input_layout)

    def setup_analysis_tab(self):
        analysis_layout = QVBoxLayout()
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        analysis_layout.addWidget(self.progress_bar)

        self.results = QLabel("Results will appear here...", self)
        analysis_layout.addWidget(self.results)

        self.analysis_tab.setLayout(analysis_layout)

    def setup_visualization_tab(self):
        visualization_layout = QVBoxLayout()
        self.visualization_label = QLabel("Topic Analysis and Audio Waveform will appear here...", self)
        visualization_layout.addWidget(self.visualization_label)
        self.visualization_tab.setLayout(visualization_layout)

    def reset_app(self):
        self.file_path_input.clear()
        self.results.setText("Results will appear here...")
        self.progress_bar.setValue(0)

    def upload_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Audio File", "", "Audio Files (*.wav *.mp3 *.ogg *.flac)")
        if file_path:
            self.file_path_input.setText(file_path)
            self.analyze_uploaded_audio(file_path)

    def analyze_uploaded_audio(self, file_path):
        self.progress_bar.setValue(10)
        processed_file_path = preprocess_audio(file_path)
        self.progress_bar.setValue(30)

        if not processed_file_path:
            self.results.setText("Failed to preprocess audio.")
            return

        y, sr = librosa.load(processed_file_path, sr=None)
        self.progress_bar.setValue(50)

        features = extract_features(y, sr)
        feature_names = ['MFCCs Mean', 'MFCCs Std', 'Chroma Mean', 'Chroma Std', 'Spectral Contrast Mean', 'RMS Mean', 'Zero Crossing Rate Mean']
        if features is None:
            self.results.setText("Failed to process audio.")
            return

        self.progress_bar.setValue(70)

        try:
            scaled_features = MODELS['scaler'].transform([features])
            selected_features = MODELS['selector'].transform(scaled_features)
            speaker = MODELS['classifier'].predict(selected_features)[0]
        except Exception as e:
            logger.error(f"Error in speaker classification: {e}")
            speaker = "Unknown"

        transcription, word_count = transcribe_audio(processed_file_path)
        if detect(transcription) != 'en':
            transcription = translate_to_english(transcription)

        emotion = detect_emotion(transcription)
        topic = detect_topic(transcription)
        self.progress_bar.setValue(100)

        emotion_text = " | ".join([f"{emo}: {score:.2f}%" for emo, score in emotion.items()])
        topic_text = " | ".join([f"{topic}: {score:.2f}%" for topic, score in topic.items()])

        self.results.setText(
            f"Speaker: {speaker}\n"
            f"Emotion: {emotion_text}\n"
            f"Topic: {topic_text}\n"
            f"Word Count: {word_count}\n"
            f"Transcription: {transcription}"
        )

        # Visualize the topic analysis and audio waveform
        self.visualize_topic_analysis_with_audio_waveform(topic, y, sr)

    def visualize_topic_analysis_with_audio_waveform(self, topics, audio_data, sample_rate):
        fig = Figure(figsize=(15, 6))  # Increased width for better spacing

        # Topic Analysis Pie Chart
        ax1 = fig.add_subplot(121)
        labels = list(topics.keys())
        sizes = list(topics.values())
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
        ax1.set_title("Topic Analysis")
        ax1.axis('equal')

        # Audio Waveform Plot
        ax2 = fig.add_subplot(122)
        time = np.linspace(0, len(audio_data) / sample_rate, num=len(audio_data))
        ax2.plot(time, audio_data, color='blue', alpha=0.7)
        ax2.set_title("Audio Waveform")
        ax2.set_xlabel("Time (seconds)")
        ax2.set_ylabel("Amplitude")

        buf = BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        pixmap = QPixmap()
        pixmap.loadFromData(buf.getvalue())
        self.visualization_label.setPixmap(pixmap)
        buf.close()

    def real_time_analysis(self):
        try:
            duration = 10
            samplerate = 16000
            self.results.setText("Recording in progress...")
            recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
            sd.wait()
            output_file = "real_time_recording.wav"
            sf.write(output_file, recording, samplerate)
            self.analyze_uploaded_audio(output_file)
        except Exception as e:
            logger.error(f"Error in real-time analysis: {e}")
            self.results.setText(f"Error: {str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    load_stylesheet(app)
    window = AudioAnalysisApp()
    window.show()
    sys.exit(app.exec_())
