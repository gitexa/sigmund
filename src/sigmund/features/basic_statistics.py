from typing import Dict

import pandas as pd

from src.pipelinelib.component import Component
from src.pipelinelib.extension import Extension
from src.pipelinelib.querying import Queryable
from src.sigmund.extensions import LEMMATIZED_PARAGRAPH, BASIC_STATISTICS_DOCUMENT_F, BASIC_STATISTICS_DOCUMENT_MF, \
    BASIC_STATISTICS_DOCUMENT_M


class BasicStatistics(Component):
    """
    This component computes basic statistics as features
    """

    def __init__(self):
        super().__init__(BasicStatistics.__name__,
                         required_extensions=[LEMMATIZED_PARAGRAPH],
                         creates_extensions=[BASIC_STATISTICS_DOCUMENT_MF,
                                             BASIC_STATISTICS_DOCUMENT_M, BASIC_STATISTICS_DOCUMENT_F])

    def apply(self, storage: Dict[Extension, pd.DataFrame],
              queryable: Queryable) -> Dict[Extension, pd.DataFrame]:
        # Get transcipts on paragraph level, stemmed
        df_lemmatized_paragraph = LEMMATIZED_PARAGRAPH.load_from(storage)

        # Scores per couple
        df_wpp_mf = self.get_words_per_paragraph(df_lemmatized_paragraph)

        # Scores for male person
        df_wpp_m = self.get_words_per_paragraph(
            df_lemmatized_paragraph.loc[df_lemmatized_paragraph['gender'] == 'M'])

        # Scores for female person
        df_wpp_f = self.get_words_per_paragraph(
            df_lemmatized_paragraph.loc[df_lemmatized_paragraph['gender'] == 'W'])

        return {BASIC_STATISTICS_DOCUMENT_MF: df_wpp_mf, BASIC_STATISTICS_DOCUMENT_M: df_wpp_m,
                BASIC_STATISTICS_DOCUMENT_F: df_wpp_f}

    def get_number_of_words(self, df):
        df['word_count'] = df['tokens_paragraph'].apply(lambda row: len(row))
        df_w_count = df.groupby('couple_id').agg('sum').word_count

        return df_w_count

    def get_number_of_paragraphs(self, df):
        df_p_count = df.groupby('couple_id').agg('count').tokens_paragraph

        return df_p_count

    def get_words_per_paragraph(self, df):
        df_w_count = self.get_number_of_words(df)
        df_p_count = self.get_number_of_paragraphs(df)
        df_wpp = pd.concat(
            (df_w_count, df_p_count),
            keys=('word_count', 'paragraph_count'),
            axis=1)

        df_wpp['words_per_paragraph'] = df_wpp.word_count / df_wpp.paragraph_count
        df_wpp.reset_index(inplace=True)

        return df_wpp
