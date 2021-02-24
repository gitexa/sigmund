from collections import Counter, OrderedDict
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.pipelinelib.component import Component
from src.pipelinelib.extension import Extension
from src.pipelinelib.querying import Parser, Queryable
from src.pipelinelib.text_body import TextBody
from src.sigmund.extensions import (POS_DOCUMENT_F, POS_DOCUMENT_M, POS_DOCUMENT_MF,
                                    POS_PARAGRAPH_F, POS_PARAGRAPH_M, POS_SENTENCE_F,
                                    POS_SENTENCE_M, TOKENS_PARAGRAPH, TOKENS_SENTENCE)


class PartOfSpeech(Component):
    """
    This component calculates part of speech features by using Spacy's POS component to
    extract the POS categories for each token in each layer (document,paragraph,sentence) and stores the number of
    occurrences of the POS categories in different columns.
    For the layers paragraph/sentence the mean of the occurrences per category over each document is stored.
    """

    def __init__(self, white_list=None, black_list=None):
        super(PartOfSpeech, self).__init__(
            PartOfSpeech.__name__,
            required_extensions=[TOKENS_PARAGRAPH, TOKENS_SENTENCE],
            creates_extensions=[POS_DOCUMENT_F, POS_DOCUMENT_M, POS_DOCUMENT_MF,
                                POS_PARAGRAPH_F, POS_PARAGRAPH_M,
                                POS_SENTENCE_F, POS_SENTENCE_M]
        )

        self.white_list = white_list or []
        self.black_list = black_list or []

    def apply(self, storage: Dict[Extension, pd.DataFrame],
              queryable: Queryable) -> Dict[Extension, pd.DataFrame]:

        tokens_sent = TOKENS_SENTENCE.load_from(storage=storage)
        tokens_par = TOKENS_PARAGRAPH.load_from(storage=storage)
        # To be able to split gender the paragraph layer is needed for document
        tokens_doc = TOKENS_PARAGRAPH.load_from(storage=storage)

        doc_count = tokens_doc['document_id'].max()
        couple_ids = tokens_doc['couple_id'].unique()

        pos_sentence = tokens_sent
        pos_paragraph = tokens_par
        pos_document = tokens_doc

        # Get POS for sentence, paragraph
        pos_sentence['tokens_sentence'] = pos_sentence['tokens_sentence'].apply(
            self._get_pos, nlp=queryable.nlp())
        pos_sentence = pos_sentence.rename(columns={'tokens_sentence': 'pos_sentence'})

        pos_paragraph['tokens_paragraph'] = pos_paragraph['tokens_paragraph'].apply(
            self._get_pos, nlp=queryable.nlp())
        pos_paragraph = pos_paragraph.rename(
            columns={'tokens_paragraph': 'pos_paragraph'})

        # Get POS for document with gender split and joined
        pos_document_A_B = pos_document.groupby(
            ['document_id', 'gender'])['tokens_paragraph'].apply(list).apply(
            self._get_pos_doc, nlp=queryable.nlp())
        pos_document_AB = pos_document.groupby(
            ['document_id'])['tokens_paragraph'].apply(list).apply(
            self._get_pos_doc, nlp=queryable.nlp())

        pos_document_A_B = pos_document_A_B.to_numpy()
        pos_document_AB = pos_document_AB.to_numpy()

        # Build Dataframe with columns: document_id, couple_id, is_depressed_group, pos_female, pos_male, pos_both
        values = np.concatenate(
            (np.arange(0, doc_count + 1, dtype=int),
             np.array(couple_ids),
             pos_document_A_B[0:: 2],
             pos_document_A_B[1:: 2],
             pos_document_AB),
            axis=0).reshape(
            (5, 10)).transpose()

        pos_document = pd.DataFrame(values,
                                    columns=['document_id', 'couple_id',
                                             'pos_document_m', 'pos_document_f',
                                             'pos_document_mf'])
        # Convert type to int
        pos_document['document_id'] = pos_document['document_id'].astype(np.int64)
        pos_document['couple_id'] = pos_document['couple_id'].astype(np.int64)

        # Split Dataframe by gender
        pos_sentence_m = pos_sentence[pos_sentence['gender'] == 'M']
        pos_sentence_f = pos_sentence[pos_sentence['gender'] == 'W']

        pos_paragraph_m = pos_paragraph[pos_paragraph['gender'] == 'M']
        pos_paragraph_f = pos_paragraph[pos_paragraph['gender'] == 'W']

        # Drop redundant columns
        pos_sentence_m = pos_sentence_m.drop(columns=['is_depressed_group'])
        pos_sentence_f = pos_sentence_f.drop(columns=['is_depressed_group'])
        pos_paragraph_m = pos_paragraph_m.drop(columns=['is_depressed_group'])
        pos_paragraph_f = pos_paragraph_f.drop(columns=['is_depressed_group'])

        pos_document_m = pos_document.drop(
            columns=['pos_document_f', 'pos_document_mf'])
        pos_document_f = pos_document.drop(
            columns=['pos_document_m', 'pos_document_mf'])
        pos_document_mf = pos_document.drop(
            columns=['pos_document_m', 'pos_document_f'])

        # Rename colums with respect to gender
        pos_sentence_m = pos_sentence_m.rename(
            columns={'pos_sentence': 'pos_sentence_m'})
        pos_sentence_f = pos_sentence_f.rename(
            columns={'pos_sentence': 'pos_sentence_f'})

        pos_paragraph_m = pos_paragraph_m.rename(
            columns={'pos_paragraph': 'pos_paragraph_m'})
        pos_paragraph_f = pos_paragraph_f.rename(
            columns={'pos_paragraph': 'pos_paragraph_f'})

        # Convert dictionary of POS (with counts) in a dataframe to separate columns and concatenate with the labels + NaN -> 0
        pos_sentence_m = pd.concat(
            [pos_sentence_m.drop(['pos_sentence_m'], axis=1),
             pos_sentence_m['pos_sentence_m'].apply(pd.Series).fillna(0).sort_index(axis=1)], axis=1)

        pos_sentence_f = pd.concat(
            [pos_sentence_f.drop(['pos_sentence_f'], axis=1),
             pos_sentence_f['pos_sentence_f'].apply(pd.Series).fillna(0).sort_index(axis=1)], axis=1)

        pos_paragraph_m = pd.concat(
            [pos_paragraph_m.drop(['pos_paragraph_m'], axis=1),
             pos_paragraph_m['pos_paragraph_m'].apply(pd.Series).fillna(0).sort_index(axis=1)], axis=1)

        pos_paragraph_f = pd.concat(
            [pos_paragraph_f.drop(['pos_paragraph_f'], axis=1),
             pos_paragraph_f['pos_paragraph_f'].apply(pd.Series).fillna(0).sort_index(axis=1)], axis=1)

        pos_document_m = pd.concat(
            [pos_document_m.drop(['pos_document_m'], axis=1),
             pos_document_m['pos_document_m'].apply(pd.Series).fillna(0).sort_index(axis=1)], axis=1)

        pos_document_f = pd.concat(
            [pos_document_f.drop(['pos_document_f'], axis=1),
             pos_document_f['pos_document_f'].apply(pd.Series).fillna(0).sort_index(axis=1)], axis=1)

        pos_document_mf = pd.concat(
            [pos_document_mf.drop(['pos_document_mf'], axis=1),
             pos_document_mf['pos_document_mf'].apply(pd.Series).fillna(0).sort_index(axis=1)], axis=1)

        # Keep only elements in the white list or remove elements in the black list
        if self.white_list != [] and self.black_list != []:
            raise Exception(
                'Both: black and white list where given. Please just enter one.')

        elif self.black_list != [] and self.white_list == []:

            pos_sentence_m = pos_sentence_m.drop(columns=self.black_list)
            pos_sentence_f = pos_sentence_f.drop(columns=self.black_list)

            pos_paragraph_m = pos_paragraph_m.drop(columns=self.black_list)
            pos_paragraph_f = pos_paragraph_f.drop(columns=self.black_list)

            pos_document_m = pos_document_m.drop(columns=self.black_list)
            pos_document_f = pos_document_f.drop(columns=self.black_list)
            pos_document_mf = pos_document_mf.drop(columns=self.black_list)

        elif self.white_list != [] and self.black_list == []:

            pos_sentence_m = pos_sentence_m[[
                'couple_id', 'speaker', 'gender', 'document_id', 'paragraph_id',
                'sentence_id'] + self.white_list]
            pos_sentence_f = pos_sentence_f[[
                'couple_id', 'speaker', 'gender', 'document_id', 'paragraph_id',
                'sentence_id'] + self.white_list]

            pos_paragraph_m = pos_paragraph_m[[
                'couple_id', 'speaker', 'gender', 'document_id',
                                                  'paragraph_id'] + self.white_list]
            pos_paragraph_f = pos_paragraph_f[[
                'couple_id', 'speaker', 'gender', 'document_id',
                                                  'paragraph_id'] + self.white_list]

            pos_document_m = pos_document_m[[
                'document_id', 'couple_id', ] + self.white_list]
            pos_document_f = pos_document_f[[
                'document_id', 'couple_id', ] + self.white_list]
            pos_document_mf = pos_document_mf[[
                'document_id', 'couple_id', ] + self.white_list]

        # Aggregate over sentences and paragraphs to get document features and drop gender/speaker

        pos_sentence_m = pos_sentence_m.drop(
            columns=['gender', 'speaker', 'paragraph_id', 'sentence_id'])
        pos_sentence_f = pos_sentence_f.drop(
            columns=['gender', 'speaker', 'paragraph_id', 'sentence_id'])
        pos_paragraph_m = pos_paragraph_m.drop(
            columns=['gender', 'speaker', 'paragraph_id'])
        pos_paragraph_f = pos_paragraph_f.drop(
            columns=['gender', 'speaker', 'paragraph_id'])

        pos_sentence_m = pos_sentence_m.groupby(
            ['document_id', 'couple_id']).agg('mean').reset_index()
        pos_sentence_f = pos_sentence_f.groupby(
            ['document_id', 'couple_id']).agg('mean').reset_index()
        pos_paragraph_m = pos_paragraph_m.groupby(
            ['document_id', 'couple_id']).agg('mean').reset_index()
        pos_paragraph_f = pos_paragraph_f.groupby(
            ['document_id', 'couple_id']).agg('mean').reset_index()

        return {POS_DOCUMENT_F: pos_document_f, POS_DOCUMENT_M: pos_document_m, POS_DOCUMENT_MF: pos_document_mf,
                POS_PARAGRAPH_F: pos_paragraph_f, POS_PARAGRAPH_M: pos_paragraph_m,
                POS_SENTENCE_F: pos_sentence_f, POS_SENTENCE_M: pos_sentence_m}

    def visualise(self, created: Dict[Extension, pd.DataFrame], queryable: Queryable):

        pos_document_mf = POS_DOCUMENT_MF.load_from(storage=created)
        pos_document_f = POS_DOCUMENT_F.load_from(storage=created)
        pos_document_m = POS_DOCUMENT_M.load_from(storage=created)

        is_depressed_group_labels = queryable.execute(
            level=TextBody.DOCUMENT).is_depressed_group

        pos_document_mf['is_depressed_group'] = is_depressed_group_labels
        pos_document_f['is_depressed_group'] = is_depressed_group_labels
        pos_document_m['is_depressed_group'] = is_depressed_group_labels

        for cat in pos_document_mf.drop(
                columns=['couple_id', 'document_id', 'is_depressed_group']).columns.values:

            if (cat not in pos_document_f.columns or cat not in pos_document_m.columns):
                continue

            cat_document_mf = pos_document_mf[['couple_id', 'is_depressed_group', cat]]
            cat_document_f = pos_document_f[['couple_id', 'is_depressed_group', cat]]
            cat_document_m = pos_document_m[['couple_id', 'is_depressed_group', cat]]
            fig, ax = plt.subplots(2, 2, figsize=(30, 15))

            # First barplot: depr/non_depr couples
            df = pd.DataFrame({'depressed couple': cat_document_mf[cat_document_mf['is_depressed_group'] == True][cat].mean(
            ), 'non-depressed couple': cat_document_mf[cat_document_mf['is_depressed_group'] == False][cat].mean()}, index=[cat])
            df.plot.bar(rot=0, ax=ax[0, 0])
            ax[0, 0].set_title('Part of Speech - ' + cat + ' - mean')

            # Second barplot: Female/Male in depr/non_depr couples
            df = pd.DataFrame({'depressed couple': cat_document_mf[cat_document_mf['is_depressed_group'] == True][cat].to_numpy(
            ), 'non-depressed couple': cat_document_mf[cat_document_mf['is_depressed_group'] == False][cat].to_numpy()})
            df.boxplot(ax=ax[1, 0])
            ax[1, 0].set_title('Part of Speech - ' + cat + ' - all')

            # First boxplot: depr/non_depr couples
            df = pd.DataFrame({'Female': [cat_document_f[cat_document_f['is_depressed_group'] == True][cat].mean(),
                                          cat_document_f[cat_document_f['is_depressed_group'] == False][cat].mean()],
                               'Male': [cat_document_m[cat_document_m['is_depressed_group'] == True][cat].mean(),
                                        cat_document_m[cat_document_m['is_depressed_group'] == False][cat].mean()]},
                              index=['depressed couple', 'non-depressed couple'])
            df.plot.bar(rot=0, ax=ax[0, 1])
            ax[0, 1].set_title('Part of Speech - ' + cat + ' - mean')

            # Second boxplot: Female/Male in depr/non_depr couples
            df = pd.DataFrame({'depressed couple - Female': cat_document_f[cat_document_f['is_depressed_group'] == True][cat].to_numpy(),
                               'depressed couple - Male': cat_document_m[cat_document_m['is_depressed_group'] == True][cat].to_numpy(),
                               'non-depressed couple - Female ': cat_document_f[cat_document_f['is_depressed_group'] == False][cat].to_numpy(),
                               'non-depressed couple - Male ': cat_document_m[cat_document_m['is_depressed_group'] == False][cat].to_numpy()})
            df.boxplot(ax=ax[1, 1])
            ax[1, 1].set_title('Part of Speech - ' + cat + ' - all')
    '''     
    def visualise(self, created: Dict[Extension, pd.DataFrame], queryable: Queryable):
        df = POS_DOCUMENT_MF.load_from(storage=created)
        metadata_doc = queryable.execute(level=TextBody.DOCUMENT)

        wmtd = pd.merge(
            metadata_doc[["couple_id", "is_depressed_group"]],
            df, on=Parser.COUPLE_ID, how="inner")

        depressed = wmtd.loc[wmtd["is_depressed_group"]][
            wmtd.columns.difference(["document_id", "is_depressed_group"])].mean()
        non_depressed = wmtd.loc[~wmtd["is_depressed_group"]][
            wmtd.columns.difference(["document_id", "is_depressed_group"])].mean()

        subbed = depressed.subtract(non_depressed).sort_values()
        # display(depressed)
        # display(non_depressed)

        subbed.plot.bar(
            subplots=True, figsize=(20, 10),
            title="Subtracted means of LIWC values for all corpi")
    '''

    def _get_pos(self, word_list, nlp):

        # Create Spacy doc from tokens and get a ordered dictionary of the tags with the counts
        df_as_string = ' '.join(word_list)
        doc = nlp(df_as_string)
        pos_list = [word.tag_ for word in doc]

        pos_list = dict(Counter(pos_list))
        pos_list = OrderedDict(sorted(pos_list.items()))
        return pos_list

    def _get_pos_doc(self, word_list, nlp):

        # Join the lists into one list
        res = [s for str_list in word_list for s in str_list]

        # Create Spacy doc from tokens and get a ordered dictionary of the tags with the counts
        df_as_string = ' '.join(res)
        doc = nlp(df_as_string)
        pos_list = [word.tag_ for word in doc]
        pos_list = dict(Counter(pos_list))
        pos_list = OrderedDict(sorted(pos_list.items()))
        return pos_list
