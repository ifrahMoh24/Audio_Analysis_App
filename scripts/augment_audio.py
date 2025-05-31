import os
import librosa
import numpy as np
import soundfile as sf
import logging
from datetime import datetime

# Configuration
RAW_DATA_DIR = "data/raw"
AUGMENTED_DATA_DIR = "data/augmented"
os.makedirs(AUGMENTED_DATA_DIR, exist_ok=True)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def augment_audio(file_path, output_dir):
    """Apply augmentation techniques to an audio file."""
    y, sr = librosa.load(file_path, sr=None)

    # Augmentation techniques
    augmentations = {
        "noise": y + 0.005 * np.random.normal(0, 1, len(y)),
        "pitch_up": librosa.effects.pitch_shift(y=y, sr=sr, n_steps=2),
        "pitch_down": librosa.effects.pitch_shift(y=y, sr=sr, n_steps=-2),
        "speed_up": librosa.effects.time_stretch(y, rate=1.2),
        "slow_down": librosa.effects.time_stretch(y, rate=0.8),
    }

    # Save augmented files with unique names
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    for aug_type, aug_y in augmentations.items():
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        output_file = os.path.join(output_dir, f"{base_name}_{aug_type}_{timestamp}.wav")
        sf.write(output_file, aug_y, sr)
        logging.info(f"Saved {aug_type} version: {output_file}")

def main():
    """Augment all audio files in RAW_DATA_DIR without skipping."""
    logging.info("Starting augmentation process...")

    # Iterate through raw audio files
    for root, _, files in os.walk(RAW_DATA_DIR):
        for file_name in files:
            if file_name.endswith(".wav"):
                file_path = os.path.join(root, file_name)
                logging.info(f"Augmenting file: {file_path}")
                augment_audio(file_path, AUGMENTED_DATA_DIR)

    logging.info("Audio augmentation completed successfully.")

if __name__ == "__main__":
    main()
