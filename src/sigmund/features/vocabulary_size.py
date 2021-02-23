from itertools import chain
from typing import Dict

import pandas as pd
from IPython.core.display import display
from spacy.attrs import ORTH

from src.pipelinelib.component import Component
from src.pipelinelib.extension import Extension
from src.pipelinelib.querying import Queryable
from src.sigmund.extensions import LEMMATIZED_DOCUMENT, VOCABULARY_SIZE_DOCUMENT_F, VOCABULARY_SIZE_DOCUMENT_M, \
    VOCABULARY_SIZE_DOCUMENT_MF, LEMMATIZED_PARAGRAPH


class VocabularySize(Component):
    """
    This component calculates the size of the used vocabulary
    """

    def __init__(self):
        super().__init__(VocabularySize.__name__,
                         required_extensions=[LEMMATIZED_DOCUMENT],
                         creates_extensions=[VOCABULARY_SIZE_DOCUMENT_MF,
                                             VOCABULARY_SIZE_DOCUMENT_M,
                                             VOCABULARY_SIZE_DOCUMENT_F])

    def apply(self, storage: Dict[Extension, pd.DataFrame],
              queryable: Queryable) -> Dict[Extension, pd.DataFrame]:
        # couple
        df_lemmatized_document = LEMMATIZED_DOCUMENT.load_from(storage)
        df_vocab_size_couple = self.get_unique_words(
            df_lemmatized_document, queryable.nlp())

        # per person
        df_lemmatized_paragraph = LEMMATIZED_PARAGRAPH.load_from(storage)
        df_lemmatized_document_pp = df_lemmatized_paragraph.groupby(
            ['couple_id', 'gender'])['tokens_paragraph'].apply(
            lambda x: list(chain.from_iterable(x))).reset_index()
        df_lemmatized_document_pp = df_lemmatized_document_pp.rename(
            columns={'tokens_paragraph': 'tokens_document'})

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
        # df_vocab_size = df_vocab_size_couple.merge(
        #    df_vocab_size_m, on='couple_id').merge(
        #    df_vocab_size_f, on='couple_id')
        # df_vocab_size.columns = ['couple_id', 'vocab_size_couple', 'vocab_size_m', 'vocab_size_f']
        # print(df_vocab_size)

        display(df_vocab_size_couple)
        display(df_vocab_size_f)
        display(df_vocab_size_m)

        return {VOCABULARY_SIZE_DOCUMENT_MF: df_vocab_size_couple, VOCABULARY_SIZE_DOCUMENT_M: df_vocab_size_m,
                VOCABULARY_SIZE_DOCUMENT_F: df_vocab_size_f}

    def get_unique_words(self, lines, nlp):
        # join text to string and retokenize with min_frequency
        lines['tokens_document'] = lines['tokens_document'].apply(
            lambda row: ' '.join(token for token in row))
        # display(lines)

        # calulcate vocabulary size
        lines['vocab_size'] = lines['tokens_document'].apply(
            lambda row: len(nlp(row).count_by(ORTH)))
        # display(lines)

        # select relevant columns
        vocab_size = lines[['couple_id', 'vocab_size']]
        # display(vocab_size)

        return vocab_size
