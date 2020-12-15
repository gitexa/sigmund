import string

import pyphen
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

dic = pyphen.Pyphen(lang='de')
import spacy

nlp = spacy.load("de_core_news_md")


def add_features(data_frame):
    data_frame["flesh_score"] = data_frame["Text"].apply(reading_ease_german)
    return data_frame

def syllable_counter(text):
    doc = nlp(text)
    syllable_count = 0
    word_count = 0
    sent_count = len(list(doc.sents))

    for token in doc:
        if not str(token) in string.punctuation:
            word_count += 1
            syllable_count += len(dic.inserted(str(token)).split("-"))

    return word_count, syllable_count, sent_count

## Adopted formula to calculate Flesh reading ease for german
def reading_ease_german(text):
    word_count, syllable_count, sent_count = syllable_counter(text)
    if syllable_count == 0 or word_count == 0 or sent_count == 0:
        return 0
    score = 180 - (word_count / sent_count) - (58.5 * (syllable_count / word_count))
    return score


def tf_idf_svd(corpus, components):
    svd = TruncatedSVD(n_components=components, algorithm="arpack")
    vectorizer = TfidfVectorizer()
    vectorizer.fit(corpus)
    x = vectorizer.transform(corpus).toarray()
    svd.fit(x)
    print(svd.singular_values_)
