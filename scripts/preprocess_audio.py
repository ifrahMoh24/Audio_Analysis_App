import numpy as np
import matplotlib.pyplot as plt
import wave
import os

# Configuration
RAW_DATA_DIR = "data/raw"
HISTOGRAM_DIR = "data/histograms"
os.makedirs(HISTOGRAM_DIR, exist_ok=True)

def load_audio(file_path):
    """Load audio data from a .wav file."""
    with wave.open(file_path, 'rb') as audio_file:
        sample_rate = audio_file.getframerate()
        num_frames = audio_file.getnframes()
        audio_data = audio_file.readframes(num_frames)
        audio_signal = np.frombuffer(audio_data, dtype=np.int16)
        return sample_rate, audio_signal

def compute_fft(audio_signal, sample_rate):
    """Compute the FFT of the audio signal."""
    n = len(audio_signal)
    fft_signal = np.fft.fft(audio_signal)
    frequencies = np.fft.fftfreq(n, d=1/sample_rate)
    return frequencies[:n//2], np.abs(fft_signal[:n//2])

def plot_histogram(audio_signal, file_name, output_dir):
    """Plot and save histogram of audio signal amplitudes."""
    plt.figure(figsize=(10, 6))
    plt.hist(audio_signal, bins=100, color='blue', alpha=0.7)
    plt.title("Amplitude Histogram")
    plt.xlabel("Amplitude")
    plt.ylabel("Frequency")
    plt.grid()
    output_file = os.path.join(output_dir, f"{file_name}_histogram.png")
    plt.savefig(output_file)
    plt.close()
    print(f"Histogram saved at: {output_file}")

def plot_fft(frequencies, fft_magnitudes, file_name, output_dir):
    """Plot and save FFT of the audio signal."""
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies, fft_magnitudes, color='red')
    plt.title("FFT Analysis")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid()
    output_file = os.path.join(output_dir, f"{file_name}_fft.png")
    plt.savefig(output_file)
    plt.close()
    print(f"FFT plot saved at: {output_file}")

def process_audio(file_path):
    """Process audio file for histogram and FFT analysis."""
    try:
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        sample_rate, audio_signal = load_audio(file_path)

        # Create and save histogram
        plot_histogram(audio_signal, file_name, HISTOGRAM_DIR)

        # Compute FFT and save plot
        frequencies, fft_magnitudes = compute_fft(audio_signal, sample_rate)
        plot_fft(frequencies, fft_magnitudes, file_name, HISTOGRAM_DIR)

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def main():
    """Main function to process all raw audio files."""
    for file_name in os.listdir(RAW_DATA_DIR):
        if file_name.endswith(".wav"):
            file_path = os.path.join(RAW_DATA_DIR, file_name)
            print(f"Processing file: {file_path}")
            process_audio(file_path)

if __name__ == "__main__":
    main()
