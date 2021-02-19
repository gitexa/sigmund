from src.pipelinelib.extension import Extension

# Raw text representations
RAW_TEXT = Extension("raw_text")
RAW_TEXT_WO_ANNOTATIONS = Extension("raw_text_wo_annotations")
RAW_TEXT_WO_STOPWORDS = Extension("raw_text_wo_stopwords")

# Preprocessing Levels
TOKENS_SENTENCE = Extension("tokens_sentence")
TOKENS_PARAGRAPH = Extension("tokens_paragraphs")
TOKENS_DOCUMENT = Extension("tokens_document")

STEMMED_SENTENCE = Extension("stemmed_sentence")
STEMMED_PARAGRAPH = Extension("stemmed_paragraph")
STEMMED_DOCUMENT = Extension("stemmed_document")

LEMMATIZED_SENTENCE = Extension("lemmatized_sentence")
LEMMATIZED_PARAGRAPH = Extension("lemmatized_paragraph")
LEMMATIZED_DOCUMENT = Extension("lemmatized_document")

FRE_SENTENCE = Extension("fre_sentence")
FRE_PARAGRAPH = Extension("fre_paragraph")
FRE_DOCUMENT = Extension("fre_document")

LOH_SENTENCE = Extension("LOH_sentence")
LOH_PARAGRAPH = Extension("LOH_paragraph")
LOH_DOCUMENT = Extension("LOH_document")

SYLLABLES = Extension("syllabels")

# Features
TFIDF_DOCUMENT = Extension("tfidf")
VOCABULARY_SIZE_DOCUMENT = Extension("vocabulary_size")
TALKTURN = Extension("talkturn")
AGREEMENTSCORE = Extension("agreementscore")

FEATURE_VECTOR = Extension("feature_vector")