import pytest
from word_counter import WordCounter

@pytest.fixture
def counter():
    return WordCounter()

def test_word_count_empty(counter):
    result = counter.count_words("")
    assert result == 0, "Word count failed for empty text."

