from src.pipelinelib.extension import Extension, ExtensionKind

# Raw text representations
RAW_TEXT = Extension(name="raw_text", kind=ExtensionKind.PREPROCESSING)
RAW_TEXT_WO_ANNOTATIONS = Extension(
    name="raw_text_wo_annotations", kind=ExtensionKind.PREPROCESSING)
RAW_TEXT_WO_STOPWORDS = Extension(
    name="raw_text_wo_stopwords", kind=ExtensionKind.PREPROCESSING)

# Preprocessing Levels
TOKENS_SENTENCE = Extension(name="tokens_sentence", kind=ExtensionKind.PREPROCESSING)
TOKENS_PARAGRAPH = Extension(name="tokens_paragraphs", kind=ExtensionKind.PREPROCESSING)
TOKENS_DOCUMENT = Extension(name="tokens_document", kind=ExtensionKind.PREPROCESSING)

STEMMED_SENTENCE = Extension(name="stemmed_sentence", kind=ExtensionKind.PREPROCESSING)
STEMMED_PARAGRAPH = Extension(name="stemmed_paragraph",
                              kind=ExtensionKind.PREPROCESSING)
STEMMED_DOCUMENT = Extension(name="stemmed_document", kind=ExtensionKind.PREPROCESSING)

LEMMATIZED_SENTENCE = Extension(
    name="lemmatized_sentence", kind=ExtensionKind.PREPROCESSING)
LEMMATIZED_PARAGRAPH = Extension(
    name="lemmatized_paragraph", kind=ExtensionKind.PREPROCESSING)
LEMMATIZED_DOCUMENT = Extension(
    name="lemmatized_document", kind=ExtensionKind.PREPROCESSING)

SYLLABLES = Extension(name="syllables", kind=ExtensionKind.PREPROCESSING)

# Features
BASIC_STATISTICS_DOCUMENT_MF = Extension(
    name="basic_statistics_document_mf", kind=ExtensionKind.FEATURE)
BASIC_STATISTICS_DOCUMENT_M = Extension(
    name="basic_statistics_document_m", kind=ExtensionKind.FEATURE)
BASIC_STATISTICS_DOCUMENT_F = Extension(
    name="basic_statistics_document_f", kind=ExtensionKind.FEATURE)

FRE_SENTENCE_M = Extension(name="fre_sentence_m", kind=ExtensionKind.FEATURE)
FRE_SENTENCE_F = Extension(name="fre_sentence_f", kind=ExtensionKind.FEATURE)
FRE_PARAGRAPH_M = Extension("name=fre_paragraph_m", kind=ExtensionKind.FEATURE)
FRE_PARAGRAPH_F = Extension(name="fre_paragraph_f", kind=ExtensionKind.FEATURE)
FRE_DOCUMENT_M = Extension(name="fre_document_m", kind=ExtensionKind.FEATURE)
FRE_DOCUMENT_F = Extension(name="fre_document_f", kind=ExtensionKind.FEATURE)
FRE_DOCUMENT_MF = Extension(name="fre_document_mf", kind=ExtensionKind.FEATURE)

LIWC_SENTENCE_M = Extension(name="liwc_sentence_m", kind=ExtensionKind.FEATURE)
LIWC_SENTENCE_F = Extension(name="liwc_sentence_f", kind=ExtensionKind.FEATURE)
LIWC_PARAGRAPH_M = Extension(name="liwc_paragraph_m", kind=ExtensionKind.FEATURE)
LIWC_PARAGRAPH_F = Extension(name="liwc_paragraph_f", kind=ExtensionKind.FEATURE)
LIWC_DOCUMENT_M = Extension(name="liwc_document_m", kind=ExtensionKind.FEATURE)
LIWC_DOCUMENT_F = Extension(name="liwc_document_f", kind=ExtensionKind.FEATURE)
LIWC_DOCUMENT_MF = Extension(name="liwc_document_mf", kind=ExtensionKind.FEATURE)

TFIDF_DOCUMENT_F = Extension(name="tfidf_document_f", kind=ExtensionKind.FEATURE)
TFIDF_DOCUMENT_M = Extension(name="tfidf_document_m", kind=ExtensionKind.FEATURE)
TFIDF_DOCUMENT_MF = Extension(name="tfidf_document_mf", kind=ExtensionKind.FEATURE)

VOCABULARY_SIZE_DOCUMENT_MF = Extension(
    name="vocabulary_size_mf", kind=ExtensionKind.FEATURE)
VOCABULARY_SIZE_DOCUMENT_F = Extension(
    name="vocabulary_size_f", kind=ExtensionKind.FEATURE)
VOCABULARY_SIZE_DOCUMENT_M = Extension(
    name="vocabulary_size_m", kind=ExtensionKind.FEATURE)

TALKTURN = Extension(name="talkturn", kind=ExtensionKind.FEATURE)

AGREEMENTSCORE = Extension(name="agreementscore", kind=ExtensionKind.FEATURE)

POS_SENTENCE_M = Extension(name='pos_sentence_m', kind=ExtensionKind.FEATURE)
POS_SENTENCE_F = Extension(name='pos_sentence_f', kind=ExtensionKind.FEATURE)
POS_PARAGRAPH_M = Extension(name='pos_paragraph_m', kind=ExtensionKind.FEATURE)
POS_PARAGRAPH_F = Extension(name='pos_paragraph_f', kind=ExtensionKind.FEATURE)
POS_DOCUMENT_M = Extension(name='pos_document_m', kind=ExtensionKind.FEATURE)
POS_DOCUMENT_F = Extension(name='pos_document_f', kind=ExtensionKind.FEATURE)
POS_DOCUMENT_MF = Extension(name='pos_document_mf', kind=ExtensionKind.FEATURE)

# Inverse LIWC Search
LIWC_INVERSE = Extension(name='liwc_inverse', kind=ExtensionKind.FEATURE)

# Aggregated features
FEATURE_VECTOR = Extension(name="feature_vector", kind=ExtensionKind.FEATURE)

# Classification results
CLASSIFICATION_NAIVE_BAYES = Extension(
    name="nb_class", kind=ExtensionKind.CLASSIFIER)
CLASSIFICATION_LINEAR_DISCRIMINANT_ANALYSIS = Extension(
    name="classification_linear_discriminant_analysis", kind=ExtensionKind.CLASSIFIER)
PCA_REDUCTION = Extension(name="pca_reduction", kind=ExtensionKind.CLASSIFIER)


CLASSIFICATION_NAIVE_BAYES_TFIDF = Extension(
    name="nb_class_tfidf", kind=ExtensionKind.CLASSIFIER
)

CLASSIFICATION_NAIVE_BAYES_LIWC = Extension(
    name="nb_class_liwc", kind=ExtensionKind.CLASSIFIER
)


CLASSIFICATION_NAIVE_BAYES_VOTING = Extension(
    name="nb_class_voting", kind=ExtensionKind.CLASSIFIER
)

CLASSIFICATION_NAIVE_BAYES_POS = Extension(
    name="nb_class_pos", kind=ExtensionKind.CLASSIFIER
)
