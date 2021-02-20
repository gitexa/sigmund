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

FRE_SENTENCE_M = Extension("fre_sentence_m")
FRE_SENTENCE_F = Extension("fre_sentence_f")
FRE_PARAGRAPH_M = Extension("fre_paragraph_m")
FRE_PARAGRAPH_F = Extension("fre_paragraph_f")
FRE_DOCUMENT_M = Extension("fre_document_m")
FRE_DOCUMENT_F = Extension("fre_document_f")
FRE_DOCUMENT_MF = Extension("fre_document_mf")


LIWC_SENTENCE_M = Extension("liwc_sentence_m")
LIWC_SENTENCE_F = Extension("liwc_sentence_f")
LIWC_PARAGRAPH_M = Extension("liwc_paragraph_m")
LIWC_PARAGRAPH_F = Extension("liwc_paragraph_f")
LIWC_DOCUMENT_M = Extension("liwc_document_m", is_feature=True)
LIWC_DOCUMENT_F = Extension("liwc_document_f")
LIWC_DOCUMENT_MF = Extension("liwc_document_mf")

SYLLABLES = Extension("syllabels")

# Features
TFIDF_DOCUMENT = Extension("tfidf_document", is_feature=True)
VOCABULARY_SIZE_DOCUMENT = Extension("vocabulary_size")
TALKTURN = Extension("talkturn")
AGREEMENTSCORE = Extension("agreementscore")


# Aggregated features
FEATURE_VECTOR = Extension("feature_vector")

# Classification results
CLASSIFICATION_NAIVE_BAYES = Extension("classification_naive_bayes")
