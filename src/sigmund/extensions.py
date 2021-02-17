from src.pipelinelib.extension import Extension

# Preprocessing Levels
TOKENS_SENTENCE = Extension("tokens_sentence")
TOKENS_PARAGRAPHS = Extension("tokens_paragraphs")
TOKENS_DOCUMENT = Extension("tokens_document")

WO_ANNOTATIONS = Extension("wo_annotations")
WO_STOPWORDS = Extension("wo_stopwords")
STEMMED = Extension("stemmed")
LEMMATIZED = Extension("lemmatized")
SYLLABLES = Extension("syllabels")

# Features
TFIDF = Extension("tfidf")
