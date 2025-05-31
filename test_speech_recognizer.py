import pytest
from speech_recognizer import SpeechRecognizer

@pytest.fixture
def recognizer():
    return SpeechRecognizer()

def test_recognize_valid_audio(recognizer):
    result = recognizer.recognize("hello.wav")
    assert result == "hello", "Speech recognition failed for valid audio."

def test_recognize_invalid_audio(recognizer):
    result = recognizer.recognize("invalid.wav")
    assert result == "error", "Speech recognition failed for invalid audio."

def test_recognize_null_audio(recognizer):
    with pytest.raises(ValueError, match="Audio file path cannot be null."):
        recognizer.recognize(None)
