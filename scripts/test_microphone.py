import pyaudio

def test_mic_access():
    chunk = 1024
    format = pyaudio.paInt16
    channels = 1
    rate = 48000

    p = pyaudio.PyAudio()
    try:
        print("Initializing microphone...")
        stream = p.open(format=format, channels=channels, rate=rate, input=True, frames_per_buffer=chunk)
        print("Microphone stream started successfully.")
        data = stream.read(chunk, exception_on_overflow=False)
        print("Successfully read audio chunk.")
        stream.stop_stream()
        stream.close()
        print("Stream closed.")
    except Exception as e:
        print(f"Error accessing microphone: {e}")
    finally:
        p.terminate()

if __name__ == "__main__":
    test_mic_access()
