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
from src.sigmund.extensions import (LEMMATIZED_DOCUMENT, LEMMATIZED_SENTENCE,
                                    STEMMED_DOCUMENT, STEMMED_SENTENCE, TFIDF_DOCUMENT,
                                    TOKENS_SENTENCE)


class FeatureTFIDF(Component):
    """
    This component provides features for classification by using sklearn tfidf
    """

    def __init__(self):
        super().__init__(
            FeatureTFIDF.__name__,
            required_extensions=[LEMMATIZED_DOCUMENT],
            creates_extensions=[TFIDF_DOCUMENT])

    def apply(self, storage: Dict[Extension, pd.DataFrame],
              queryable: Queryable) -> Dict[Extension, pd.DataFrame]:

        # Get transcipts on document level stemmed
        df_lemmatized_document = LEMMATIZED_DOCUMENT.load_from(storage)
        display(df_lemmatized_document)
        df_tfidf_document = self.get_tfidf(df_lemmatized_document)

        return {TFIDF_DOCUMENT: df_tfidf_document}

    def get_tfidf(self, lines):

        # get vectors with frequencies for the words in the lines; each line is considered a document; remove stop words with stop-word-list from scikit-learn; exclude words with frequency smaller 5
        count_vectorizer = CountVectorizer(min_df=1)
        count_vectorized = count_vectorizer.fit_transform(lines)

        # transform the vector-frequency matrix in tfidf
        tfidf_transformer = TfidfTransformer(use_idf=True)
        tfidf = tfidf_transformer.fit_transform(count_vectorized)

        # create new dataframe with tfidfs and the feature names; calculate the mean tfidf over all lines/documents
        df_tfidf = pd.DataFrame(
            tfidf.T.todense(),
            index=count_vectorizer.get_feature_names())
        df_tfidf['mean'] = df_tfidf.mean(axis=1)
        df_tfidf = df_tfidf.sort_values('mean', ascending=False)

        display(df_tfidf)

        return df_tfidf['mean']
