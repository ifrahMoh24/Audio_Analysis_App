import os
import librosa
import numpy as np
import logging

# Configuration
AUGMENTED_DATA_DIR = "data/augmented"
PROCESSED_DATA_DIR = "data/processed"
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def extract_features(file_path):
    """Extract audio features from a file."""
    y, sr = librosa.load(file_path, sr=16000)
    
    # Feature extraction: MFCCs, delta, delta-delta, and additional features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta_mfcc = librosa.feature.delta(mfccs)
    delta2_mfcc = librosa.feature.delta(mfccs, order=2)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)

    # Combine all features into a single array
    features = np.hstack((
        np.mean(mfccs, axis=1),  # MFCC mean (13 features)
        np.mean(delta_mfcc, axis=1),  # Delta MFCC mean (13 features)
        np.mean(delta2_mfcc, axis=1),  # Delta-Delta MFCC mean (13 features)
        np.mean(spectral_centroid),  # Spectral centroid (1 feature)
        np.mean(spectral_bandwidth),  # Spectral bandwidth (1 feature)
        np.mean(zero_crossing_rate)  # Zero-crossing rate (1 feature)
    ))

    # Ensure the feature array matches the required feature count (e.g., 39)
    required_features = 39
    if features.shape[0] < required_features:
        features = np.pad(features, (0, required_features - features.shape[0]), mode='constant')
    elif features.shape[0] > required_features:
        features = features[:required_features]

    logging.info(f"Extracted features shape: {features.shape}")  # Debugging
    return features


def process_audio_files(input_dir, output_dir):
    """Process all audio files and save features."""
    if not os.path.exists(input_dir) or not os.listdir(input_dir):
        logging.error(f"Input directory {input_dir} does not exist or is empty.")
        return

    for file_name in os.listdir(input_dir):
        if file_name.endswith(".wav"):
            file_path = os.path.join(input_dir, file_name)
            logging.info(f"Processing file: {file_path}")
            try:
                features = extract_features(file_path)
                output_file = os.path.join(output_dir, f"{file_name.split('.')[0]}_features.npy")
                np.save(output_file, features)
                logging.info(f"Saved features to: {output_file}")
            except Exception as e:
                logging.error(f"Failed to process {file_path}: {e}")


def main():
    """Main function to process augmented audio data."""
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    process_audio_files(AUGMENTED_DATA_DIR, PROCESSED_DATA_DIR)


if __name__ == "__main__":
    main()
