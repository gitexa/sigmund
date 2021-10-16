from itertools import chain
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from src.pipelinelib.component import Component
from src.pipelinelib.extension import Extension
from src.pipelinelib.querying import Queryable
from src.pipelinelib.text_body import TextBody
from src.sigmund.extensions import (LEMMATIZED_DOCUMENT, LEMMATIZED_PARAGRAPH,
                                    TFIDF_DOCUMENT_F, TFIDF_DOCUMENT_M,
                                    TFIDF_DOCUMENT_MF)


class FeatureTFIDF(Component):
    """
    This component provides features for classification by using sklearn's TFIDF classes.

    The TFIDF value corresponds to how much weight a word has within the given corpus.

    Words that appear rarely or exceedingly commonly are deemed to have low relevance for
    the dataset, and are given low scores.
    Therefore words with higher scores are deemed to be more important, because they appear
    a "meaningful" amount of times within the corpus.
    """

    def __init__(self, white_list=[], black_list=[]):
        super().__init__(
            FeatureTFIDF.__name__,
            required_extensions=[LEMMATIZED_DOCUMENT],
            creates_extensions=[TFIDF_DOCUMENT_MF, TFIDF_DOCUMENT_M, TFIDF_DOCUMENT_F])
        self.white_list = white_list
        self.black_list = black_list

    def apply(self, storage: Dict[Extension, pd.DataFrame],
              queryable: Queryable) -> Dict[Extension, pd.DataFrame]:

        # Get transcipts on document level stemmed
        df_lemmatized_document = LEMMATIZED_DOCUMENT.load_from(storage)

        # Calculate TFIDF per couple
        df_tfidf_document_mf = self.get_tfidf(df_lemmatized_document)

        # Aggregate paragraph information to docment per person
        df_lemmatized_paragraph = LEMMATIZED_PARAGRAPH.load_from(storage)
        df_lemmatized_document_pp = df_lemmatized_paragraph.groupby(
            ['couple_id', 'gender'])['tokens_paragraph'].apply(
            lambda x: list(chain.from_iterable(x))).reset_index()
        df_lemmatized_document_pp = df_lemmatized_document_pp.rename(
            columns={'tokens_paragraph': 'tokens_document'})

        # Calculate TFIDF for masculin person
        df_lemmatized_document_pp_m = df_lemmatized_document_pp.loc[
            df_lemmatized_document_pp['gender'] == 'M'].reset_index(drop=True)
        df_tfidf_document_m = self.get_tfidf(df_lemmatized_document_pp_m)

        # Calculate TFIDF for feminin person
        df_lemmatized_document_pp_f = df_lemmatized_document_pp.loc[
            df_lemmatized_document_pp['gender'] == 'W'].reset_index(drop=True)
        df_tfidf_document_f = self.get_tfidf(df_lemmatized_document_pp_f)

        # Check black and white list
        if self.white_list != [] and self.black_list != []:
            raise Exception(
                'Both: black and white list where given. Please just enter one.')
        elif self.black_list != [] and self.white_list == []:
            df_tfidf_document_mf = df_tfidf_document_mf.drop(columns=self.black_list)
            df_tfidf_document_m = df_tfidf_document_m.drop(columns=self.black_list)
            df_tfidf_document_f = df_tfidf_document_f.drop(columns=self.black_list)
        elif self.white_list != [] and self.black_list == []:
            df_tfidf_document_mf = df_tfidf_document_mf[['couple_id'] + self.white_list]
            df_tfidf_document_m = df_tfidf_document_m[['couple_id'] + self.white_list]
            df_tfidf_document_f = df_tfidf_document_f[['couple_id'] + self.white_list]

        return {TFIDF_DOCUMENT_MF: df_tfidf_document_mf, TFIDF_DOCUMENT_M: df_tfidf_document_m,
                TFIDF_DOCUMENT_F: df_tfidf_document_f}

    def get_tfidf(self, lines):

        # join text to string and retokenize with min_frequency
        lines['tokens_document'] = lines['tokens_document'].apply(
            lambda row: ' '.join(token for token in row))

        # get vectors with frequencies for the words in the lines; each line is considered a document; exclude words with frequency smaller 5
        count_vectorizer = CountVectorizer(min_df=2)
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

        return df_tfidf

    def visualise(self, created: Dict[Extension, pd.DataFrame], queryable: Queryable):

        tfidf_document_mf = TFIDF_DOCUMENT_MF.load_from(storage=created)
        tfidf_document_f = TFIDF_DOCUMENT_F.load_from(storage=created)
        tfidf_document_m = TFIDF_DOCUMENT_M.load_from(storage=created)

        is_depressed_group_labels = queryable.execute(
            level=TextBody.DOCUMENT).is_depressed_group

        tfidf_document_mf['is_depressed_group'] = is_depressed_group_labels
        tfidf_document_f['is_depressed_group'] = is_depressed_group_labels
        tfidf_document_m['is_depressed_group'] = is_depressed_group_labels

        for cat in tfidf_document_mf.drop(
                columns=['couple_id', 'is_depressed_group']).columns.values:

            if (cat not in tfidf_document_f.columns or cat not in tfidf_document_m.columns):
                continue

            cat_document_mf = tfidf_document_mf[[
                'couple_id', 'is_depressed_group', cat]]
            cat_document_f = tfidf_document_f[['couple_id', 'is_depressed_group', cat]]
            cat_document_m = tfidf_document_m[['couple_id', 'is_depressed_group', cat]]
            fig, ax = plt.subplots(2, 2, figsize=(30, 15))

            # First barplot: depr/non_depr couples
            df = pd.DataFrame({'depressed couple': cat_document_mf[cat_document_mf['is_depressed_group'] == True][cat].mean(
            ), 'non-depressed couple': cat_document_mf[cat_document_mf['is_depressed_group'] == False][cat].mean()}, index=[cat])
            df.plot.bar(rot=0, ax=ax[0, 0])
            ax[0, 0].set_title('TFIDF - ' + cat + ' - mean')

            # Second barplot: Female/Male in depr/non_depr couples
            #df = pd.DataFrame({'depressed couple': cat_document_mf[cat_document_mf['is_depressed_group'] == True][cat].to_numpy(
            #), 'non-depressed couple': cat_document_mf[cat_document_mf['is_depressed_group'] == False][cat].to_numpy()})
            #df.boxplot(ax=ax[1, 0])
            #ax[1, 0].set_title('TFIDF - ' + cat + ' - all')

            df = pd.DataFrame([cat_document_mf[cat_document_mf['is_depressed_group'] == True][cat].reset_index(
                drop=True), cat_document_mf[cat_document_mf['is_depressed_group'] == False][cat].reset_index(drop=True)]).T
            df.columns = ['depressed couple', 'non-depressed couple']
            df.boxplot(ax=ax[1, 0])
            ax[1, 0].set_title('TFIDF - ' + cat + ' - all')

            # First boxplot: depr/non_depr couples
            df = pd.DataFrame({'Female': [cat_document_f[cat_document_f['is_depressed_group'] == True][cat].mean(),
                                          cat_document_f[cat_document_f['is_depressed_group'] == False][cat].mean()],
                               'Male': [cat_document_m[cat_document_m['is_depressed_group'] == True][cat].mean(),
                                        cat_document_m[cat_document_m['is_depressed_group'] == False][cat].mean()]},
                              index=['depressed couple', 'non-depressed couple'])
            df.plot.bar(rot=0, ax=ax[0, 1])
            ax[0, 1].set_title('TFIDF - ' + cat + ' - mean')

            # Second boxplot: Female/Male in depr/non_depr couples
            #df = pd.DataFrame({'depressed couple - Female': cat_document_f[cat_document_f['is_depressed_group'] == True][cat].to_numpy(),
            #                   'depressed couple - Male': cat_document_m[cat_document_m['is_depressed_group'] == True][cat].to_numpy(),
            #                   'non-depressed couple - Female ': cat_document_f[cat_document_f['is_depressed_group'] == False][cat].to_numpy(),
            #                   'non-depressed couple - Male ': cat_document_m[cat_document_m['is_depressed_group'] == False][cat].to_numpy()})
            df = pd.DataFrame([cat_document_f[cat_document_f['is_depressed_group'] == True][cat].reset_index(drop=True),
                               cat_document_m[cat_document_m['is_depressed_group'] == True][cat].reset_index(drop=True),
                               cat_document_f[cat_document_f['is_depressed_group'] == False][cat].reset_index(drop=True),
                               cat_document_m[cat_document_m['is_depressed_group'] == False][cat].reset_index(drop=True)]).T
            df.columns = ['depressed couple - Female', 
                            'depressed couple - Male',
                            'non-depressed couple - Female', 
                            'non-depressed couple - Male']
            df.boxplot(ax=ax[1, 1])
            ax[1, 1].set_title('TFIDF - ' + cat + ' - all')
