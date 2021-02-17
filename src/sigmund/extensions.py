from src.pipelinelib.extension import Extension

# Preprocessing Levels
TOKENS_SENTENCE = Extension("tokens_sentence")
TOKENS_PARAGRAPHS = Extension("tokens_paragraphs")
TOKENS_DOCUMENT = Extension("tokens_document")
STEMMED = Extension("stemmed")
LEMMATIZED = Extension("lemmatized")
SYLLABLES = Extension("syllabels")

# Features
TFIDF = Extension("tfidf")
