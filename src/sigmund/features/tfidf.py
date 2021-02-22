from itertools import chain
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
            creates_extensions=[TFIDF_DOCUMENT_MF, TFIDF_DOCUMENT_M, TFIDF_DOCUMENT_F])

    def apply(self, storage: Dict[Extension, pd.DataFrame],
              queryable: Queryable) -> Dict[Extension, pd.DataFrame]:

        # Get transcipts on document level stemmed
        df_lemmatized_document = LEMMATIZED_DOCUMENT.load_from(storage)

        # couple
        df_tfidf_document = self.get_tfidf(df_lemmatized_document)

        # per person
        df_lemmatized_paragraph = LEMMATIZED_PARAGRAPH.load_from(storage)
        df_lemmatized_document_pp = df_lemmatized_paragraph.groupby(
            ['couple_id', 'gender'])['tokens_paragraph'].apply(
            lambda x: list(chain.from_iterable(x))).reset_index()
        df_lemmatized_document_pp = df_lemmatized_document_pp.rename(columns={'tokens_paragraph': 'tokens_document'})

        # masculin
        df_lemmatized_document_pp_m = df_lemmatized_document_pp.loc[
            df_lemmatized_document_pp['gender'] == 'M']
        df_tfidf_document_m = self.get_tfidf(df_lemmatized_document_pp_m)

        # feminin person
        df_lemmatized_document_pp_f = df_lemmatized_document_pp.loc[
            df_lemmatized_document_pp['gender'] == 'W']
        df_tfidf_document_f = self.get_tfidf(
            df_lemmatized_document_pp_f)

        return {TFIDF_DOCUMENT_MF: df_tfidf_document, TFIDF_DOCUMENT_M: df_tfidf_document_m, TFIDF_DOCUMENT_F: df_tfidf_document_f}

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
        df_tfidf.insert(loc=0, column='couple_id', value=lines['couple_id'])

        # select only columns which differ between depressed and non-depressed couples
        # TODO if desired

        return df_tfidf
