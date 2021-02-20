import operator
import re
import string
from typing import Dict

import pandas as pd
from IPython.core.display import display
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from src.pipelinelib.component import Component
from src.pipelinelib.extension import Extension
from src.pipelinelib.querying import Queryable
from src.pipelinelib.text_body import TextBody
from src.sigmund.extensions import *


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
        df_tfidf_document = self.get_tfidf(df_lemmatized_document)

        return {TFIDF_DOCUMENT: df_tfidf_document}

    def get_tfidf(self, lines):

        # join text to string and retokenize with min_frequency
        lines['tokens_document'] = lines['tokens_document'].apply(
            lambda row: ' '.join(token for token in row))

        # get vectors with frequencies for the words in the lines; each line is considered a document; remove stop words with stop-word-list from scikit-learn; exclude words with frequency smaller 5
        count_vectorizer = CountVectorizer(min_df=5)
        count_vectorized = count_vectorizer.fit_transform(lines['tokens_document'])

        # transform the vector-frequency matrix in tfidf
        tfidf_transformer = TfidfTransformer(use_idf=True)
        tfidf = tfidf_transformer.fit_transform(count_vectorized)

        # create new dataframe with tfidfs and the feature names; calculate the mean tfidf over all lines/documents
        df_tfidf = pd.DataFrame(
            tfidf.T.todense(),
            index=count_vectorizer.get_feature_names()).T

        # construct results with couple_id
        df_tfidf = df_tfidf.insert(loc=0, column='couple_id', value=lines['couple_id'])

        # select only columns which differ between depressed and non-depressed couples
        # TODO if desired

        return df_tfidf
