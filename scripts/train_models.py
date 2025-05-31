import os
import numpy as np
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import pickle

# Paths to audio files
audio_files = {
    "Ifrah": "data/raw/Ifrah/Ifra5_converted.wav",
    "Sharonne": "data/raw/Sharonne/Sharonne1_converted.wav",
    "Talha": "data/raw/Talha/Talha5_converted.wav"
}

# Function to extract features
def extract_features(y, sr):
    try:
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        rms = librosa.feature.rms(y=y)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)

        # Combine features into a consistent vector
        features = np.hstack([
            np.mean(mfccs, axis=1), np.std(mfccs, axis=1),
            np.mean(chroma, axis=1), np.std(chroma, axis=1),
            np.mean(spectral_contrast), np.mean(rms), np.mean(zero_crossing_rate)
        ])

        if features.shape[0] < 20:
            raise ValueError("Insufficient features extracted.")

        print(f"Extracted features shape: {features.shape}")  # Log feature shape
        return features
    except Exception as e:
        print(f"Error in feature extraction: {e}")
        return None

# Function to augment audio
def augment_audio(y, sr):
    try:
        noise = y + 0.005 * np.random.normal(0, 1, len(y))
        pitch_up = librosa.effects.pitch_shift(y, sr=sr, n_steps=4)
        pitch_down = librosa.effects.pitch_shift(y, sr=sr, n_steps=-4)
        speed_up = librosa.effects.time_stretch(y, rate=1.25)
        speed_down = librosa.effects.time_stretch(y, rate=0.75)
        return [noise, pitch_up, pitch_down, speed_up, speed_down]
    except Exception as e:
        print(f"Error in audio augmentation: {e}")
        return []

# Train Speaker Model
def train_speaker_model():
    print("Loading data...")
    data, labels = [], []
    for label, audio_path in audio_files.items():
        try:
            y, sr = librosa.load(audio_path, sr=16000)
            features = extract_features(y, sr)
            if features is not None:
                data.append(features)
                labels.append(label)

            # Add augmentations
            for i, aug in enumerate(augment_audio(y, sr)):
                aug_features = extract_features(aug, sr)
                if aug_features is not None:
                    data.append(aug_features)
                    labels.append(label)
                    print(f"Augmentation {i + 1} for {label} processed.")
        except Exception as e:
            print(f"Error loading file {audio_path}: {e}")

    if not data:
        print("No valid data available for training.")
        return

    # Train-test split
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Feature Scaling
    print("Scaling features...")
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Feature Selection
    print("Selecting features...")
    selector = SelectKBest(f_classif, k=20)  # Match feature count to model expectation
    X_train = selector.fit_transform(X_train, y_train)
    X_test = selector.transform(X_test)

    # Train Model
    print("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(classification_report(y_test, y_pred))

    # Save model, scaler, and selector
    with open("models/speaker_classifier.joblib", "wb") as model_file, \
         open("models/scaler.joblib", "wb") as scaler_file, \
         open("models/selector.joblib", "wb") as selector_file:
        pickle.dump(model, model_file)
        pickle.dump(scaler, scaler_file)
        pickle.dump(selector, selector_file)

    print("Speaker model saved successfully.")

# Main entry point
if __name__ == "__main__":
    train_speaker_model()
