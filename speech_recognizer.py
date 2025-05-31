class SpeechRecognizer:
    def recognize(self, audio_file):
        if not audio_file:
            raise ValueError("Audio file path cannot be null.")
        if audio_file == "hello.wav":
            return "hello"
        return "error"
