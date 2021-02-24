import collections
from collections import Counter
from typing import Dict, List

import liwc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.pipelinelib.component import Component
from src.pipelinelib.extension import Extension
from src.pipelinelib.querying import Queryable
from src.pipelinelib.text_body import TextBody
from src.sigmund.extensions import (LIWC_DOCUMENT_F, LIWC_DOCUMENT_M, LIWC_DOCUMENT_MF,
                                    LIWC_INVERSE, LIWC_PARAGRAPH_F, LIWC_PARAGRAPH_M,
                                    LIWC_PLOT, LIWC_SENTENCE_F, LIWC_SENTENCE_M,
                                    LIWC_TREND, TOKENS_PARAGRAPH, TOKENS_SENTENCE)


class Liwc(Component):
    def __init__(
            self, white_list=[],
            black_list=[],
            token_parser_path="./data/German_LIWC2001_Dictionary.dic"):
        super(Liwc, self).__init__(
            Liwc.__name__,
            required_extensions=[TOKENS_PARAGRAPH, TOKENS_SENTENCE],
            creates_extensions=[LIWC_DOCUMENT_F, LIWC_DOCUMENT_M, LIWC_DOCUMENT_MF,
                                LIWC_PARAGRAPH_F, LIWC_PARAGRAPH_M,
                                LIWC_SENTENCE_F, LIWC_SENTENCE_M]
        )
        # self.parse, self.category_names = liwc.load_token_parser(token_parser_path)#Liwc(token_parser_path)
        self.white_list = white_list
        self.black_list = black_list
        self.token_parser_path = token_parser_path

    def apply(self, storage: Dict[Extension, pd.DataFrame],
              queryable: Queryable) -> Dict[Extension, pd.DataFrame]:

        tokens_sent = TOKENS_SENTENCE.load_from(storage=storage)
        tokens_par = TOKENS_PARAGRAPH.load_from(storage=storage)
        # To be able to split gender the paragraph layer is needed for document
        tokens_doc = TOKENS_PARAGRAPH.load_from(storage=storage)

        doc_count = tokens_doc['document_id'].max()
        couple_ids = tokens_doc['couple_id'].unique()
        liwc_sentence = tokens_sent
        liwc_paragraph = tokens_par
        liwc_document = tokens_doc

        # Get the liwc parser/categories
        parse, category_names = liwc.load_token_parser(self.token_parser_path)

        # Get LIWC for sentence, paragraph
        liwc_sentence['tokens_sentence'] = liwc_sentence['tokens_sentence'].apply(
            liwc_parser, parse=parse, category=category_names)
        liwc_sentence = liwc_sentence.rename(columns={'tokens_sentence': 'liwc_sentences'})

        liwc_paragraph['tokens_paragraph'] = liwc_paragraph['tokens_paragraph'].apply(
            liwc_parser, parse=parse, category=category_names)
        liwc_paragraph = liwc_paragraph.rename(columns={'tokens_paragraph': 'liwc_paragraph'})

        # Get LIWC for document with gender split and joined
        liwc_document_A_B = liwc_document.groupby(['document_id', 'gender'])['tokens_paragraph'].apply(list).apply(
            liwc_parser_doc, parse=parse, category=category_names)
        liwc_document_AB = liwc_document.groupby(['document_id'])['tokens_paragraph'].apply(list).apply(
            liwc_parser_doc, parse=parse, category=category_names)

        liwc_document_A_B = liwc_document_A_B.to_numpy()
        liwc_document_AB = liwc_document_AB.to_numpy()

        # Build Dataframe with columns: document_id, couple_id, liwc_female, liwc_male, liwc_both
        values = np.concatenate(
            (np.arange(0, doc_count + 1, dtype=int),
             np.array(couple_ids),
             liwc_document_A_B[0:: 2],
             liwc_document_A_B[1:: 2],
             liwc_document_AB),
            axis=0).reshape(
            (5, 10)).transpose()

        liwc_document = pd.DataFrame(values, columns=['document_id', 'couple_id', 'liwc_document_m', 'liwc_document_f', 'liwc_document_mf'])
        liwc_document['document_id'] = liwc_document['document_id'].astype(np.int64)
        liwc_document['couple_id'] = liwc_document['couple_id'].astype(np.int64)

        # Split Dataframe by gender
        # Rename colums with respect to gender
        liwc_sentence_m = liwc_sentence[liwc_sentence['gender'] == 'M']
        liwc_sentence_f = liwc_sentence[liwc_sentence['gender'] == 'W']
        liwc_sentence_m = liwc_sentence_m.rename(columns={'liwc_sentences': 'liwc_sentence_m'})
        liwc_sentence_f = liwc_sentence_f.rename(columns={'liwc_sentences': 'liwc_sentence_f'})

        liwc_paragraph_m = liwc_paragraph[liwc_paragraph['gender'] == 'M']
        liwc_paragraph_f = liwc_paragraph[liwc_paragraph['gender'] == 'W']
        liwc_paragraph_m = liwc_paragraph_m.rename(columns={'liwc_paragraph': 'liwc_paragraph_m'})
        liwc_paragraph_f = liwc_paragraph_f.rename(columns={'liwc_paragraph': 'liwc_paragraph_f'})

        # Drop redundant columns
        liwc_sentence_m = liwc_sentence_m.drop(columns=['is_depressed_group'])
        liwc_sentence_f = liwc_sentence_f.drop(columns=['is_depressed_group'])
        liwc_paragraph_m = liwc_paragraph_m.drop(columns=['is_depressed_group'])
        liwc_paragraph_f = liwc_paragraph_f.drop(columns=['is_depressed_group'])

        liwc_document_m = liwc_document.drop(columns=['liwc_document_f', 'liwc_document_mf'])
        liwc_document_f = liwc_document.drop(columns=['liwc_document_m', 'liwc_document_mf'])
        liwc_document_mf = liwc_document.drop(columns=['liwc_document_m', 'liwc_document_f'])

        # Convert dictionary of LIWC (with counts) in a dataframe to separate columns and concatenate with the labels + NaN -> 0
        liwc_sentence_m = pd.concat(
            [liwc_sentence_m.drop(['liwc_sentence_m'], axis=1),
             liwc_sentence_m['liwc_sentence_m'].apply(pd.Series).fillna(0).sort_index(axis=1)], axis=1)

        liwc_sentence_f = pd.concat(
            [liwc_sentence_f.drop(['liwc_sentence_f'], axis=1),
             liwc_sentence_f['liwc_sentence_f'].apply(pd.Series).fillna(0).sort_index(axis=1)], axis=1)

        liwc_paragraph_m = pd.concat(
            [liwc_paragraph_m.drop(['liwc_paragraph_m'], axis=1),
             liwc_paragraph_m['liwc_paragraph_m'].apply(pd.Series).fillna(0).sort_index(axis=1)], axis=1)

        liwc_paragraph_f = pd.concat(
            [liwc_paragraph_f.drop(['liwc_paragraph_f'], axis=1),
             liwc_paragraph_f['liwc_paragraph_f'].apply(pd.Series).fillna(0).sort_index(axis=1)], axis=1)

        liwc_document_m = pd.concat(
            [liwc_document_m.drop(['liwc_document_m'], axis=1),
             liwc_document_m['liwc_document_m'].apply(pd.Series).fillna(0).sort_index(axis=1)], axis=1)

        liwc_document_f = pd.concat(
            [liwc_document_f.drop(['liwc_document_f'], axis=1),
             liwc_document_f['liwc_document_f'].apply(pd.Series).fillna(0).sort_index(axis=1)], axis=1)

        liwc_document_mf = pd.concat(
            [liwc_document_mf.drop(['liwc_document_mf'], axis=1),
             liwc_document_mf['liwc_document_mf'].apply(pd.Series).fillna(0).sort_index(axis=1)], axis=1)

        # Keep only elements in the white list or remove elements in the black list
        if self.white_list != [] and self.black_list != []:
            raise Exception(
                'Both: black and white list where given. Please just enter one.')

        elif self.black_list != [] and self.white_list == []:

            liwc_sentence_m = liwc_sentence_m.drop(columns=self.black_list)
            liwc_sentence_f = liwc_sentence_f.drop(columns=self.black_list)

            liwc_paragraph_m = liwc_paragraph_m.drop(columns=self.black_list)
            liwc_paragraph_f = liwc_paragraph_f.drop(columns=self.black_list)

            liwc_document_m = liwc_document_m.drop(columns=self.black_list)
            liwc_document_f = liwc_document_f.drop(columns=self.black_list)
            liwc_document_mf = liwc_document_mf.drop(columns=self.black_list)

        elif self.white_list != [] and self.black_list == []:

            liwc_sentence_m = liwc_sentence_m[[
                'couple_id', 'speaker', 'gender', 'document_id', 'paragraph_id', 'sentence_id'] + self.white_list]
            liwc_sentence_f = liwc_sentence_f[[
                'couple_id', 'speaker', 'gender', 'document_id', 'paragraph_id', 'sentence_id'] + self.white_list]

            liwc_paragraph_m = liwc_paragraph_m[[
                'couple_id', 'speaker', 'gender', 'document_id', 'paragraph_id'] + self.white_list]
            liwc_paragraph_f = liwc_paragraph_f[[
                'couple_id', 'speaker', 'gender', 'document_id', 'paragraph_id'] + self.white_list]

            liwc_document_m = liwc_document_m[[
                'document_id', 'couple_id', ] + self.white_list]
            liwc_document_f = liwc_document_f[[
                'document_id', 'couple_id', ] + self.white_list]
            liwc_document_mf = liwc_document_mf[[
                'document_id', 'couple_id', ] + self.white_list]

        # Aggregate over sentences and paragraphs to get document features and drop gender/speaker

        liwc_sentence_m = liwc_sentence_m.drop(
            columns=['gender', 'speaker', 'paragraph_id', 'sentence_id'])
        liwc_sentence_f = liwc_sentence_f.drop(
            columns=['gender', 'speaker', 'paragraph_id', 'sentence_id'])
        liwc_paragraph_m = liwc_paragraph_m.drop(
            columns=['gender', 'speaker', 'paragraph_id'])
        liwc_paragraph_f = liwc_paragraph_f.drop(
            columns=['gender', 'speaker', 'paragraph_id'])

        liwc_sentence_m = liwc_sentence_m.groupby(
            ['document_id', 'couple_id']).agg('mean').reset_index()
        liwc_sentence_f = liwc_sentence_f.groupby(
            ['document_id', 'couple_id']).agg('mean').reset_index()
        liwc_paragraph_m = liwc_paragraph_m.groupby(
            ['document_id', 'couple_id']).agg('mean').reset_index()
        liwc_paragraph_f = liwc_paragraph_f.groupby(
            ['document_id', 'couple_id']).agg('mean').reset_index()

        return {LIWC_DOCUMENT_M: liwc_document_m, LIWC_DOCUMENT_F: liwc_document_f, LIWC_DOCUMENT_MF: liwc_document_mf,
                LIWC_PARAGRAPH_M: liwc_paragraph_m, LIWC_PARAGRAPH_F: liwc_paragraph_f,
                LIWC_SENTENCE_M: liwc_sentence_m, LIWC_SENTENCE_F: liwc_sentence_f}

    def visualise(self, created: Dict[Extension, pd.DataFrame], queryable: Queryable):

        liwc_document_mf = LIWC_DOCUMENT_MF.load_from(storage=created)
        liwc_document_f = LIWC_DOCUMENT_F.load_from(storage=created)
        liwc_document_m = LIWC_DOCUMENT_M.load_from(storage=created)

        is_depressed_group_labels = queryable.execute(
            level=TextBody.DOCUMENT).is_depressed_group

        liwc_document_mf['is_depressed_group'] = is_depressed_group_labels
        liwc_document_f['is_depressed_group'] = is_depressed_group_labels
        liwc_document_m['is_depressed_group'] = is_depressed_group_labels

        for cat in liwc_document_mf.drop(
                columns=['couple_id', 'document_id', 'is_depressed_group']).columns.values:

            if (cat not in liwc_document_f.columns or cat not in liwc_document_m.columns):
                continue

            cat_document_mf = liwc_document_mf[['couple_id', 'is_depressed_group', cat]]
            cat_document_f = liwc_document_f[['couple_id', 'is_depressed_group', cat]]
            cat_document_m = liwc_document_m[['couple_id', 'is_depressed_group', cat]]
            fig, ax = plt.subplots(2, 2, figsize=(30, 15))

            # First barplot: depr/non_depr couples
            df = pd.DataFrame({'depressed couple': cat_document_mf[cat_document_mf['is_depressed_group'] == True][cat].mean(
            ), 'non-depressed couple': cat_document_mf[cat_document_mf['is_depressed_group'] == False][cat].mean()}, index=[cat])
            df.plot.bar(rot=0, ax=ax[0, 0])
            ax[0, 0].set_title('LIWC - ' + cat + ' - mean')

            # Second barplot: Female/Male in depr/non_depr couples
            df = pd.DataFrame({'depressed couple': cat_document_mf[cat_document_mf['is_depressed_group'] == True][cat].to_numpy(
            ), 'non-depressed couple': cat_document_mf[cat_document_mf['is_depressed_group'] == False][cat].to_numpy()})
            df.boxplot(ax=ax[1, 0])
            ax[1, 0].set_title('LIWC - ' + cat + ' - all')

            # First boxplot: depr/non_depr couples
            df = pd.DataFrame({'Female': [cat_document_f[cat_document_f['is_depressed_group'] == True][cat].mean(),
                                          cat_document_f[cat_document_f['is_depressed_group'] == False][cat].mean()],
                               'Male': [cat_document_m[cat_document_m['is_depressed_group'] == True][cat].mean(),
                                        cat_document_m[cat_document_m['is_depressed_group'] == False][cat].mean()]},
                              index=['depressed couple', 'non-depressed couple'])
            df.plot.bar(rot=0, ax=ax[0, 1])
            ax[0, 1].set_title('LIWC - ' + cat + ' - mean')

            # Second boxplot: Female/Male in depr/non_depr couples
            df = pd.DataFrame({'depressed couple - Female': cat_document_f[cat_document_f['is_depressed_group'] == True][cat].to_numpy(),
                               'depressed couple - Male': cat_document_m[cat_document_m['is_depressed_group'] == True][cat].to_numpy(),
                               'non-depressed couple - Female ': cat_document_f[cat_document_f['is_depressed_group'] == False][cat].to_numpy(),
                               'non-depressed couple - Male ': cat_document_m[cat_document_m['is_depressed_group'] == False][cat].to_numpy()})
            df.boxplot(ax=ax[1, 1])
            ax[1, 1].set_title('LIWC - ' + cat + ' - all')


def liwc_parser(tokens, parse, category):

    # Get a dictionary of all liwc category with ther respect counts
    liwc_cats = Counter(category for token in tokens for category in parse(token))
    liwc_cats = dict(liwc_cats)
    liwc_cats = collections.OrderedDict(sorted(liwc_cats.items()))
    return liwc_cats


def liwc_parser_doc(tokens, parse, category):

    # Join the lists into one list
    res = []
    for str_list in tokens:
        res += str_list

    # Get a dictionary of all liwc category with ther respect counts
    liwc_cats = Counter(category for token in res for category in parse(token))
    liwc_cats = dict(liwc_cats)
    liwc_cats = collections.OrderedDict(sorted(liwc_cats.items()))
    return liwc_cats  # images = [
#     Image.open(os.path.join(pipeline._plot_output, ext.filename() + ".png"))
#     for ext, _ in features
# ]
# captions = [ext.name for ext in plots]

# st.image(images, caption=captions, width=None, clear_figure=False)
class Liwc_Inverse(Component):
    def __init__(
            self, category=[],
            token_parser_path="./data/German_LIWC2001_Dictionary.dic"):
        super(Liwc_Inverse, self).__init__(
            Liwc_Inverse.__name__,
            required_extensions=[TOKENS_SENTENCE],
            creates_extensions=[LIWC_INVERSE]
        )
        # self.parse, self.category_names = liwc.load_token_parser(token_parser_path)#Liwc(token_parser_path)
        self.category = category
        self.token_parser_path = token_parser_path

    def apply(self, storage: Dict[Extension, pd.DataFrame],
              queryable: Queryable) -> Dict[Extension, pd.DataFrame]:

        # Get the liwc parser/categories
        parse, category_names = liwc.load_token_parser(self.token_parser_path)

        liwc_inverse_sentence = TOKENS_SENTENCE.load_from(storage=storage)
        for cat in self.category:
            # Get LIWC for sentence
            liwc_inverse_sentence[cat] = liwc_inverse_sentence['tokens_sentence'].apply(
                liwc_inverse_parser, parse=parse, category=category_names, search=cat)

            # Convert 0 to NaN in LIWC categories
            liwc_inverse_sentence[cat].replace(0, np.nan, inplace=True)

        liwc_inverse_sentence = liwc_inverse_sentence.drop(columns=['speaker'])

        # Replaye tokens with whole sentence
        liwc_inverse_sentence['tokens_sentence'] = queryable.execute(level=TextBody.SENTENCE)[
            'text']
        liwc_inverse_sentence = liwc_inverse_sentence.rename(
            columns={'tokens_sentence': 'sentence'})

        # Drop rows not including the liwc category
        liwc_inverse_sentence = liwc_inverse_sentence.dropna(
            subset=self.category, thresh=1)

        # Convert NaN to empty list in LIWC categories
        liwc_inverse_sentence[self.category] = liwc_inverse_sentence[self.category].apply(
            lambda s: s.fillna({i: [] for i in liwc_inverse_sentence[self.category].index}))
        # display(liwc_inverse_sentence.reset_index(drop=True))

        return {LIWC_INVERSE: liwc_inverse_sentence.reset_index(drop=True)}

    def visualise(self, created: Dict[Extension, pd.DataFrame], queryable: Queryable):
        display(LIWC_INVERSE.load_from(storage=created))


def liwc_inverse_parser(tokens, parse, category, search):

    # Get a dictionary of all liwc category with their respect counts
    cat_token = []
    for token in tokens:
        if search in list(parse(token)):
            cat_token.append(token)

    return cat_token if len(cat_token) != 0 else 0


class Liwc_Trend(Component):
    def __init__(
            self, category=[],
            token_parser_path="./data/German_LIWC2001_Dictionary.dic"):
        super(Liwc_Trend, self).__init__(
            Liwc_Trend.__name__,
            required_extensions=[TOKENS_PARAGRAPH],
            creates_extensions=[LIWC_TREND]
        )
        self.category = category
        self.token_parser_path = token_parser_path

    def apply(self, storage: Dict[Extension, pd.DataFrame],
              queryable: Queryable) -> Dict[Extension, pd.DataFrame]:
        # Get the liwc parser/categories
        parse, category_names = liwc.load_token_parser(self.token_parser_path)

        liwc_trend_paragraph = TOKENS_PARAGRAPH.load_from(storage=storage)

        for cat in self.category:
            # Get LIWC score for each paragraph to get the trend over the discussion
            liwc_trend_paragraph[cat] = liwc_trend_paragraph['tokens_paragraph'].apply(
                liwc_trend_parser, parse=parse, category=category_names, search=cat)

        # Drop redundant columns
        liwc_trend_paragraph = liwc_trend_paragraph.drop(
            columns=['speaker', 'tokens_paragraph'])

        return {LIWC_TREND: liwc_trend_paragraph.reset_index(drop=True)}

    def visualise(self, created: Dict[Extension, pd.DataFrame], queryable: Queryable):

        liwc_trend_paragraph = LIWC_TREND.load_from(storage=created)
        count = len(set(liwc_trend_paragraph.couple_id.tolist()))
        height, width = 5 * count, 20

        for cat in self.category:
            fig, axes = plt.subplots(count, 1, figsize=(width, height))
            i = 0

            for couple in set(liwc_trend_paragraph.couple_id.tolist()):

                # Get LIWC scores for each paragraph for one couple
                liwc_trend_couple = liwc_trend_paragraph.loc[liwc_trend_paragraph['couple_id'] == couple][cat].reset_index(drop=True)

                # Get is depressed information for the current couple
                liwc_trend_is_depressed = set(liwc_trend_paragraph.loc[liwc_trend_paragraph['couple_id'] == couple]['is_depressed_group']).pop()
                liwc_trend_is_depressed = "(depressed)" if liwc_trend_is_depressed == True else "(non-depressed)"

                # Plot trend
                liwc_trend_couple.plot.line(ax=axes[i], xlabel='Paragraphs', ylabel='Count')
                axes[i].set_ylim([0, 15])
                axes[i].set_title('Couple-ID ' + str(couple) + ' ' + liwc_trend_is_depressed + ' - ' + cat + ' (wordcount) per paragraph')
                i += 1  # display for next couple


def liwc_trend_parser(tokens, parse, category, search):

    # Get the number words per paragraph having the desired liwc category
    cat_token_count = 0
    for token in tokens:
        if search in list(parse(token)):
            cat_token_count += 1

    return cat_token_count
