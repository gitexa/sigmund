import operator
import re
import string
from collections import Counter
from itertools import filterfalse
from typing import Dict

import liwc
import pandas as pd
from IPython.core.display import display
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from spacy.tokens import Doc

from src.pipelinelib.component import Component
from src.pipelinelib.extension import Extension
from src.pipelinelib.querying import Queryable
from src.pipelinelib.text_body import TextBody
from src.sigmund.extensions import (LEMMATIZED_SENTENCE, STEMMED_DOCUMENT, STEMMED_SENTENCE,
                                    TFIDF, TOKENS_SENTENCE)
from src.sigmund.preprocessing.syllables import SyllableExtractor
from src.sigmund.preprocessing.words import WordExtractor


class FeatureTFIDF(Component):
    """
    This component provides features for classification by using sklearn tfidf
    """

    def __init__(self):
        super().__init__(FeatureTFIDF.__name__, required_extensions=[
            STEMMED_DOCUMENT], creates_extensions=[TFIDF])

    def apply(self, storage: Dict[Extension, pd.DataFrame],
              queryable: Queryable) -> Dict[Extension, pd.DataFrame]:
        # Get transcipts on document level stemmed
        STEMMED_DOCUMENT.load_from(storage)
        display(STEMMED_DOCUMENT)

        tokens = queryable.execute(level=TextBody.DOCUMENT)
        tokens = tokens[['document_id', 'paragraph_id',
                         'sentence_id', 'speaker', 'text']]
        tokens['text'] = tokens['text'].apply(
            tokenize_df, nlp=queryable.nlp())

        return {TOKENS_SENTENCE: tokens}

    POS = Extension("tfidf_scores", dict())


def __tokenize_and_lower(self, doc: Doc):
    getter = operator.attrgetter(WordExtractor.WORDS.name)
    return [word.lower() for word in getter(doc._)]


def __get_tfidf(tokens):
    # get vectors with frequencies for the words in the lines; each line is considered a document; remove stop words with stop-word-list from scikit-learn; exclude words with frequency smaller 5
    count_vectorizer = CountVectorizer(min_df=5)
    count_vectorized = count_vectorizer.fit_transform(tokens)

    # transform the vector-frequency matrix in tfidf
    tfidf_transformer = TfidfTransformer(use_idf=True)
    tfidf = tfidf_transformer.fit_transform(count_vectorized)

    # create new dataframe with tfidfs and the feature names; calculate the mean tfidf over all lines/documents
    df_tfidf = pd.DataFrame(
        tfidf.T.todense(),
        index=count_vectorizer.get_feature_names())
    df_tfidf['mean'] = df_tfidf.mean(axis=1)
    df_tfidf = df_tfidf.sort_values('mean', ascending=False)

    return df_tfidf['mean']
