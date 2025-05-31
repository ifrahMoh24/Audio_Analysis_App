class WordCounter:
    def count_words(self, text):
        return len([word for word in text.split() if word.isalnum()])
