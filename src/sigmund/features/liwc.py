import collections
from collections import Counter
from typing import Dict, List

import liwc
import numpy as np
import pandas as pd

from src.pipelinelib.component import Component
from src.pipelinelib.extension import Extension
from src.pipelinelib.querying import Queryable
from src.pipelinelib.text_body import TextBody
from src.sigmund.extensions import (LIWC_DOCUMENT_F, LIWC_DOCUMENT_M, LIWC_DOCUMENT_MF,
                                    LIWC_PARAGRAPH_F, LIWC_PARAGRAPH_M, LIWC_SENTENCE_F,
                                    LIWC_SENTENCE_M, TOKENS_PARAGRAPH, TOKENS_SENTENCE)


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
        liwc_sentence = liwc_sentence.rename(
            columns={'tokens_sentence': 'liwc_sentences'})

        liwc_paragraph['tokens_paragraph'] = liwc_paragraph['tokens_paragraph'].apply(
            liwc_parser, parse=parse, category=category_names)
        liwc_paragraph = liwc_paragraph.rename(
            columns={'tokens_paragraph': 'liwc_paragraph'})

        # Get LIWC for document with gender split and joined
        liwc_document_A_B = liwc_document.groupby(
            ['document_id', 'gender'])['tokens_paragraph'].apply(list).apply(
            liwc_parser_doc, parse=parse, category=category_names)
        liwc_document_AB = liwc_document.groupby(
            ['document_id'])['tokens_paragraph'].apply(list).apply(
            liwc_parser_doc, parse=parse, category=category_names)

        liwc_document_A_B = liwc_document_A_B.to_numpy()
        liwc_document_AB = liwc_document_AB.to_numpy()

        # Add is_depressed_group label
        # document = queryable.execute(level=TextBody.DOCUMENT)
        # is_depressed_group = document['is_depressed_group'].to_numpy()

        # Build Dataframe with columns: document_id, couple_id, liwc_female, liwc_male, liwc_both
        values = np.concatenate(
            (np.arange(0, doc_count + 1, dtype=int),
             np.array(couple_ids),
             liwc_document_A_B[0:: 2],
             liwc_document_A_B[1:: 2],
             liwc_document_AB),
            axis=0).reshape(
            (5, 10)).transpose()

        liwc_document = pd.DataFrame(
            values,
            columns=['document_id', 'couple_id', 'liwc_document_m', 'liwc_document_f', 'liwc_document_mf'])
        liwc_document['document_id'] = liwc_document['document_id'].astype(np.int64)
        liwc_document['couple_id'] = liwc_document['couple_id'].astype(np.int64)

        # Split Dataframe by gender
        # Rename colums with respect to gender
        liwc_sentence_m = liwc_sentence[liwc_sentence['gender'] == 'M']
        liwc_sentence_f = liwc_sentence[liwc_sentence['gender'] == 'W']
        liwc_sentence_m = liwc_sentence_m.rename(
            columns={'liwc_sentences': 'liwc_sentence_m'})
        liwc_sentence_f = liwc_sentence_f.rename(
            columns={'liwc_sentences': 'liwc_sentence_f'})

        liwc_paragraph_m = liwc_paragraph[liwc_paragraph['gender'] == 'M']
        liwc_paragraph_f = liwc_paragraph[liwc_paragraph['gender'] == 'W']
        liwc_paragraph_m = liwc_paragraph_m.rename(
            columns={'liwc_paragraph': 'liwc_paragraph_m'})
        liwc_paragraph_f = liwc_paragraph_f.rename(
            columns={'liwc_paragraph': 'liwc_paragraph_f'})

        # Drop redundant columns
        liwc_sentence_m = liwc_sentence_m.drop(columns=['is_depressed_group'])
        liwc_sentence_f = liwc_sentence_f.drop(columns=['is_depressed_group'])
        liwc_paragraph_m = liwc_paragraph_m.drop(columns=['is_depressed_group'])
        liwc_paragraph_f = liwc_paragraph_f.drop(columns=['is_depressed_group'])

        liwc_document_m = liwc_document.drop(
            columns=['liwc_document_f', 'liwc_document_mf'])
        liwc_document_f = liwc_document.drop(
            columns=['liwc_document_m', 'liwc_document_mf'])
        liwc_document_mf = liwc_document.drop(
            columns=['liwc_document_m', 'liwc_document_f'])

        # Convert dictionary of LIWC (with counts) in a dataframe to separate columns and concatenate with the labels + NaN -> 0
        liwc_sentence_m = pd.concat(
            [liwc_sentence_m.drop(['liwc_sentence_m'],
                                  axis=1),
             liwc_sentence_m['liwc_sentence_m'].apply(pd.Series).fillna(0).sort_index(
                 axis=1)],
            axis=1)

        liwc_sentence_f = pd.concat(
            [liwc_sentence_f.drop(['liwc_sentence_f'],
                                  axis=1),
             liwc_sentence_f['liwc_sentence_f'].apply(pd.Series).fillna(0).sort_index(
                 axis=1)],
            axis=1)

        liwc_paragraph_m = pd.concat([liwc_paragraph_m.drop(
            ['liwc_paragraph_m'],
            axis=1),
            liwc_paragraph_m['liwc_paragraph_m'].
            apply(pd.Series).fillna(0).sort_index(
            axis=1)],
            axis=1)

        liwc_paragraph_f = pd.concat([liwc_paragraph_f.drop(
            ['liwc_paragraph_f'],
            axis=1),
            liwc_paragraph_f['liwc_paragraph_f'].
            apply(pd.Series).fillna(0).sort_index(
            axis=1)],
            axis=1)

        liwc_document_m = pd.concat(
            [liwc_document_m.drop(['liwc_document_m'],
                                  axis=1),
             liwc_document_m['liwc_document_m'].apply(pd.Series).fillna(0).sort_index(
                 axis=1)],
            axis=1)

        liwc_document_f = pd.concat(
            [liwc_document_f.drop(['liwc_document_f'],
                                  axis=1),
             liwc_document_f['liwc_document_f'].apply(pd.Series).fillna(0).sort_index(
                 axis=1)],
            axis=1)

        liwc_document_mf = pd.concat([liwc_document_mf.drop(
            ['liwc_document_mf'],
            axis=1),
            liwc_document_mf['liwc_document_mf'].
            apply(pd.Series).fillna(0).sort_index(
            axis=1)],
            axis=1)

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

        #Aggregate over sentences and paragraphs to get document features and drop gender/speaker

        liwc_sentence_m = liwc_sentence_m.drop(columns=['gender', 'speaker', 'paragraph_id', 'sentence_id'])
        liwc_sentence_f = liwc_sentence_f.drop(columns=['gender', 'speaker', 'paragraph_id', 'sentence_id'])
        liwc_paragraph_m = liwc_paragraph_m.drop(columns=['gender', 'speaker', 'paragraph_id'])
        liwc_paragraph_f = liwc_paragraph_f.drop(columns=['gender', 'speaker', 'paragraph_id'])

        liwc_sentence_m = liwc_sentence_m.groupby(['document_id', 'couple_id']).agg('max').reset_index()
        liwc_sentence_f = liwc_sentence_f.groupby(['document_id', 'couple_id']).agg('max').reset_index()
        liwc_paragraph_m = liwc_paragraph_m.groupby(['document_id', 'couple_id']).agg('max').reset_index()
        liwc_paragraph_f = liwc_paragraph_f.groupby(['document_id', 'couple_id']).agg('max').reset_index()

        return {LIWC_DOCUMENT_M: liwc_document_m, LIWC_DOCUMENT_F: liwc_document_f, LIWC_DOCUMENT_MF: liwc_document_mf,
                LIWC_PARAGRAPH_M: liwc_paragraph_m, LIWC_PARAGRAPH_F: liwc_paragraph_f,
                LIWC_SENTENCE_M: liwc_sentence_m, LIWC_SENTENCE_F: liwc_sentence_f}


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
    return liwc_cats
