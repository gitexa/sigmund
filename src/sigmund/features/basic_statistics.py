from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd

from src.pipelinelib.component import Component
from src.pipelinelib.extension import Extension
from src.pipelinelib.querying import Queryable
from src.pipelinelib.text_body import TextBody
from src.sigmund.extensions import (BASIC_STATISTICS_DOCUMENT_F,
                                    BASIC_STATISTICS_DOCUMENT_M,
                                    BASIC_STATISTICS_DOCUMENT_MF, LEMMATIZED_PARAGRAPH)


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

    def visualise(self, created: Dict[Extension, pd.DataFrame], queryable: Queryable):

        bs_document_mf = BASIC_STATISTICS_DOCUMENT_MF.load_from(storage=created)
        bs_document_f = BASIC_STATISTICS_DOCUMENT_F.load_from(storage=created)
        bs_document_m = BASIC_STATISTICS_DOCUMENT_M.load_from(storage=created)

        is_depressed_group_labels = queryable.execute(level=TextBody.DOCUMENT).is_depressed_group

        bs_document_mf['is_depressed_group'] = is_depressed_group_labels
        bs_document_f['is_depressed_group'] = is_depressed_group_labels
        bs_document_m['is_depressed_group'] = is_depressed_group_labels

        ###################### FOR COUPLE
        fig, ax = plt.subplots(2, 3, figsize=(30, 15))

        # First barplot: depr/non_depr couples - Number of paragraphs - mean
        df = pd.DataFrame({'Depressed couple': bs_document_mf[bs_document_mf['is_depressed_group'] == True]['paragraph_count'].mean(),
                           'Non-depressed couple': bs_document_mf[bs_document_mf['is_depressed_group'] == False]['paragraph_count'].mean()}, index=['paragraph_count'])
        df.T.plot.bar(rot=0, ax=ax[0, 0], legend=False)
        ax[0, 0].set_title('Number of paragraphs - mean')

        # First boxplot: depr/non_depr couples - Number of paragraphs
        df = pd.DataFrame({'Depressed couple': bs_document_mf[bs_document_mf['is_depressed_group'] == True]['paragraph_count'].to_numpy(),
                           'Non-depressed couple': bs_document_mf[bs_document_mf['is_depressed_group'] == False]['paragraph_count'].to_numpy()})
        df.boxplot(rot=0, ax=ax[1, 0])
        ax[1, 0].set_title('Number of paragraphs')

        # Second barplot: depr/non_depr couples - Number of words - mean
        df = pd.DataFrame({'Depressed couple': bs_document_mf[bs_document_mf['is_depressed_group'] == True]['word_count'].mean(),
                           'Non-depressed couple': bs_document_mf[bs_document_mf['is_depressed_group'] == False]['word_count'].mean()}, index=['word_count'])
        df.T.plot.bar(rot=0, ax=ax[0, 1], legend=False)
        ax[0, 1].set_title('Number of words - mean')

        # Second boxplot: depr/non_depr couples - Number of words
        df = pd.DataFrame({'Depressed couple': bs_document_mf[bs_document_mf['is_depressed_group'] == True]['word_count'].to_numpy(),
                           'Non-depressed couple': bs_document_mf[bs_document_mf['is_depressed_group'] == False]['word_count'].to_numpy()})
        df.boxplot(rot=0, ax=ax[1, 1])
        ax[1, 1].set_title('Number of words')

        # Third barplot: depr/non_depr couples - Number of words per paragraph- mean
        df = pd.DataFrame({'Depressed couple': bs_document_mf[bs_document_mf['is_depressed_group'] == True]['words_per_paragraph'].mean(),
                           'Non-depressed couple': bs_document_mf[bs_document_mf['is_depressed_group'] == False]['words_per_paragraph'].mean()}, index=['words_per_paragraph'])
        df.T.plot.bar(rot=0, ax=ax[0, 2], legend=False)
        ax[0, 2].set_title('Words per paragraph - mean')

        # Thrid boxplot: depr/non_depr couples - Number of words per paragraph
        df = pd.DataFrame({'Depressed couple': bs_document_mf[bs_document_mf['is_depressed_group'] == True]['words_per_paragraph'].to_numpy(),
                           'Non-depressed couple': bs_document_mf[bs_document_mf['is_depressed_group'] == False]['words_per_paragraph'].to_numpy()})
        df.boxplot(rot=0, ax=ax[1, 2])
        ax[1, 2].set_title('Words per paragraph')

###################### FOR PERSON PER COUPLE
        fig, ax = plt.subplots(2, 3, figsize=(30, 15))
        # First barplot: depr/non_depr per person - Number of paragraphs - mean
        df = pd.DataFrame({'Female': [bs_document_f[bs_document_f['is_depressed_group'] == True]['paragraph_count'].mean(),
                                      bs_document_f[bs_document_f['is_depressed_group'] == False]['paragraph_count'].mean()],
                            'Male': [bs_document_m[bs_document_m['is_depressed_group'] == True]['paragraph_count'].mean(),
                                    bs_document_m[bs_document_m['is_depressed_group'] == False]['paragraph_count'].mean()]},
                            index=['Depressed couple', 'Non-depressed couple'])
        df.plot.bar(rot=0, ax=ax[0, 0])
        ax[0, 0].set_title('Number of paragraphs per person - mean')

        # First boxplot: depr/non_depr per person - Number of paragraphs
        df = pd.DataFrame({'depr. couple - Female': bs_document_f[bs_document_f['is_depressed_group'] == True]['paragraph_count'].to_numpy(),
                            'depr. couple - Male': bs_document_m[bs_document_m['is_depressed_group'] == True]['paragraph_count'].to_numpy(),
                            'non-depr. couple - Female ': bs_document_f[bs_document_f['is_depressed_group'] == False]['paragraph_count'].to_numpy(),
                            'non-depr. couple - Male ': bs_document_m[bs_document_m['is_depressed_group'] == False]['paragraph_count'].to_numpy()})
        df.boxplot(ax=ax[1, 0])
        ax[1, 0].set_title('Number of paragraphs per person')
        ax[1, 0].tick_params(labelrotation=45)
        
        # Second barplot: depr/non_depr per person - Number of words - mean
        df = pd.DataFrame({'Female': [bs_document_f[bs_document_f['is_depressed_group'] == True]['word_count'].mean(),
                                      bs_document_f[bs_document_f['is_depressed_group'] == False]['word_count'].mean()],
                            'Male': [bs_document_m[bs_document_m['is_depressed_group'] == True]['word_count'].mean(),
                                    bs_document_m[bs_document_m['is_depressed_group'] == False]['word_count'].mean()]},
                            index=['Depressed couple', 'Non-depressed couple'])
        df.plot.bar(rot=0, ax=ax[0, 1])
        ax[0, 1].set_title('Number of words - mean')

        # Second boxplot: depr/non_depr per person - Number of words
        df = pd.DataFrame({'depr. couple - Female': bs_document_f[bs_document_f['is_depressed_group'] == True]['word_count'].to_numpy(),
                            'depr. couple - Male': bs_document_m[bs_document_m['is_depressed_group'] == True]['word_count'].to_numpy(),
                            'non-depr. couple - Female ': bs_document_f[bs_document_f['is_depressed_group'] == False]['word_count'].to_numpy(),
                            'non-depr. couple - Male ': bs_document_m[bs_document_m['is_depressed_group'] == False]['word_count'].to_numpy()})
        df.boxplot(ax=ax[1, 1])
        ax[1, 1].set_title('Number of words')
        ax[1, 1].tick_params(labelrotation=45)

        # Third barplot: depr/non_depr per person - Number of words per paragraph- mean
        df = pd.DataFrame({'Female': [bs_document_f[bs_document_f['is_depressed_group'] == True]['words_per_paragraph'].mean(),
                                      bs_document_f[bs_document_f['is_depressed_group'] == False]['words_per_paragraph'].mean()],
                            'Male': [bs_document_m[bs_document_m['is_depressed_group'] == True]['words_per_paragraph'].mean(),
                                    bs_document_m[bs_document_m['is_depressed_group'] == False]['words_per_paragraph'].mean()]},
                            index=['Depressed couple', 'Non-depressed couple'])
        df.plot.bar(rot=0, ax=ax[0, 2])
        ax[0, 2].set_title('Words per paragraph - mean')

        # Thrid boxplot: depr/non_depr per person - Number of words per paragraph
        df = pd.DataFrame({'depr. couple - Female': bs_document_f[bs_document_f['is_depressed_group'] == True]['words_per_paragraph'].to_numpy(),
                            'depr. couple - Male': bs_document_m[bs_document_m['is_depressed_group'] == True]['words_per_paragraph'].to_numpy(),
                            'non-depr. couple - Female ': bs_document_f[bs_document_f['is_depressed_group'] == False]['words_per_paragraph'].to_numpy(),
                            'non-depr. couple - Male ': bs_document_m[bs_document_m['is_depressed_group'] == False]['words_per_paragraph'].to_numpy()})
        df.boxplot(ax=ax[1, 2])
        ax[1, 2].set_title('Words per paragraph')
        ax[1, 2].tick_params(labelrotation=45)
