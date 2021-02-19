import operator
import re
import string
from functools import reduce
from itertools import chain
from typing import Dict

import pandas as pd
from IPython.core.display import display
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from spacy.attrs import ORTH

from src.pipelinelib.component import Component
from src.pipelinelib.extension import Extension
from src.pipelinelib.querying import Queryable
from src.pipelinelib.text_body import TextBody
from src.sigmund.extensions import *


class VocabularySize(Component):

    """
    This component calculates the size of the used vocabulary
    """

    def __init__(self):
        super().__init__(
            VocabularySize.__name__,
            required_extensions=[LEMMATIZED_DOCUMENT],
            creates_extensions=[VOCABULARY_SIZE_DOCUMENT])

    def apply(self, storage: Dict[Extension, pd.DataFrame],
              queryable: Queryable) -> Dict[Extension, pd.DataFrame]:

        # couple
        df_lemmatized_document = LEMMATIZED_DOCUMENT.load_from(storage)
        df_vocab_size_couple = self.get_unique_words(
            df_lemmatized_document, queryable.nlp())

        # per person
        df_lemmatized_paragraph = LEMMATIZED_PARAGRAPH.load_from(storage)
        df_lemmatized_document_pp = df_lemmatized_paragraph.groupby(
            ['couple_id', 'gender'])['text'].apply(
            lambda x: list(chain.from_iterable(x))).reset_index()

        # masculin
        df_lemmatized_document_pp_m = df_lemmatized_document_pp.loc[
            df_lemmatized_document_pp['gender'] == 'M']
        df_vocab_size_m = self.get_unique_words(
            df_lemmatized_document_pp_m, queryable.nlp())

        # feminin person
        df_lemmatized_document_pp_f = df_lemmatized_document_pp.loc[
            df_lemmatized_document_pp['gender'] == 'W']
        df_vocab_size_f = self.get_unique_words(
            df_lemmatized_document_pp_f, queryable.nlp())

        # merge dataframes for couple, masculin and feminin person
        df_vocab_size = df_vocab_size_couple.merge(
            df_vocab_size_m, on='couple_id').merge(
            df_vocab_size_f, on='couple_id')
        df_vocab_size.columns = ['couple_id', 'vocab_size_couple', 'vocab_size_m', 'vocab_size_f']
        print(df_vocab_size)

        return {VOCABULARY_SIZE_DOCUMENT: df_vocab_size}

    def get_unique_words(self, lines, nlp):

        # join text to string and retokenize with min_frequency
        lines['text'] = lines['text'].apply(
            lambda row: ' '.join(token for token in row))
        display(lines)

        # calulcate vocabulary size
        lines['vocab_size'] = lines['text'].apply(
            lambda row: len(nlp(row).count_by(ORTH)))
        display(lines)

        # select relevant columns
        vocab_size = lines[['couple_id', 'vocab_size']]
        display(vocab_size)

        return vocab_size
