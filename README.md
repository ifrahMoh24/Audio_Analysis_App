# AudioAnalysisApp — Speech & Sentiment Analyzer

## 📌 Overview
AudioAnalysisApp is a desktop application that records audio, transcribes it, and performs basic sentiment analysis. It’s designed for quick audio diagnostics, educational demos, and prototypes that need speech-to-text and tone detection.

## 🚀 Features
- Record audio from the system microphone
- Save recordings as WAV files
- Transcribe audio using a speech‑recognition library
- Perform sentiment analysis on transcribed text
- Simple GUI for recording and viewing results

## 🛠 Tech Stack
- **Language:** Python 3.9
- **GUI:** PyQt5 (via `ui/` folder)
- **Speech Recognition:** `speech_recognition` library
- **Sentiment Analysis:** `textblob` or `vaderSentiment` (choose one)
- **Other dependencies:** `pyaudio`, `numpy`

## 📱 Screenshots / Demo


## ⚙️ Installation & Setup
1. Clone the repo:


git clone https://github.com/ifrahMoh24/AudioAnalysisApp-Clean.git

cd AudioAnalysisApp-Clean

2. Create a virtual environment and install dependencies:


python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

3. Run the application:

python main.py

4. Ensure you have the necessary system libraries for audio (e.g., `portaudio` on macOS/Linux).

## 📂 Folder Structure
- `scripts/` — helper scripts for audio preprocessing
- `ui/` — PyQt UI files
- `app.py` — main application module
- `main.py` — entry point
- `speech_recognizer.py` — speech recognition utilities
- `sentiment_analyzer.py` — sentiment analysis utilities
- `requirements.txt` — Python dependencies

## 📌 Status
Under development — prototype works; improvements planned for accuracy and UI polish.

## 👤 Author
Ifrah Mohamed  
[LinkedIn](https://linkedin.com/in/ifrah-mohamed-4233b7225) | [GitHub](https://github.com/ifrahMoh24)
