from src.pipelinelib.extension import Extension

# Raw text representations
RAW_TEXT = Extension("raw_text")
RAW_TEXT_WO_ANNOTATIONS = Extension("raw_text_wo_annotations")
RAW_TEXT_WO_STOPWORDS = Extension("raw_text_wo_stopwords")

# Preprocessing Levels
TOKENS_SENTENCE = Extension("tokens_sentence")
TOKENS_PAdRAGRAPH = Extension("tokens_paragraphs")
TOKENS_DOCUMENT = Extension("tokens_document")

STEMMED_SENTENCE = Extension("stemmed_sentence")
STEMMED_PARAGRAPH = Extension("stemmed_paragraph")
STEMMED_DOCUMENT = Extension("stemmed_document")

LEMMATIZED = Extension("lemmatized")


SYLLABLES = Extension("syllabels")

# Features
TFIDF = Extension("tfidf")
