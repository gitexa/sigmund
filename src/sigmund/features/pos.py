import operator
from collections import Counter, OrderedDict
from typing import Dict, List

import numpy as np
import pandas as pd
from spacy.tokens import Doc

from src.pipelinelib.component import Component
from src.pipelinelib.extension import Extension
from src.pipelinelib.querying import Queryable
from src.pipelinelib.text_body import TextBody
from src.sigmund.extensions import (POS_DOCUMENT_F, POS_DOCUMENT_M, POS_DOCUMENT_MF,
                                    POS_PARAGRAPH_F, POS_PARAGRAPH_M, POS_SENTENCE_F,
                                    POS_SENTENCE_M, TOKENS_PARAGRAPH, TOKENS_SENTENCE)
from src.sigmund.preprocessing.pos import PartOfSpeechTags


class PartOfSpeech(Component):
    """
    This component calculates part of speech features by using Spacy's POS component to calc the counts per word-category 
    """

    def __init__(self, white_list=[], black_list=[]):
        super(PartOfSpeech, self).__init__(
            PartOfSpeech.__name__,
            required_extensions=[TOKENS_PARAGRAPH, TOKENS_SENTENCE],
            creates_extensions=[POS_DOCUMENT_F, POS_DOCUMENT_M, POS_DOCUMENT_MF,
                                POS_PARAGRAPH_F, POS_PARAGRAPH_M,
                                POS_SENTENCE_F, POS_SENTENCE_M, ]
        )
        self.white_list = white_list
        self.black_list = black_list
        
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
        pos_sentence['tokens_sentence'] = pos_sentence['tokens_sentence'].apply(self._get_pos, nlp=queryable.nlp())
        pos_sentence = pos_sentence.rename(columns={'tokens_sentence': 'pos_sentence'})

        pos_paragraph['tokens_paragraph'] = pos_paragraph['tokens_paragraph'].apply(self._get_pos, nlp=queryable.nlp())
        pos_paragraph = pos_paragraph.rename(columns={'tokens_paragraph': 'pos_paragraph'})

        # Get POS for document with gender split and joined
        pos_document_A_B = pos_document.groupby(['document_id', 'gender'])['tokens_paragraph'].apply(list).apply(self._get_pos_doc, nlp=queryable.nlp())
        pos_document_AB = pos_document.groupby(['document_id'])['tokens_paragraph'].apply(list).apply(self._get_pos_doc, nlp=queryable.nlp())

        pos_document_A_B = pos_document_A_B.to_numpy()
        pos_document_AB = pos_document_AB.to_numpy()

        # Add is_depressed_group label
        document = queryable.execute(level=TextBody.DOCUMENT)
        is_depressed_group = document['is_depressed_group'].to_numpy()

        #Build Dataframe with columns: document_id, couple_id, is_depressed_group, pos_female, pos_male, pos_both
        values = np.concatenate(
            (np.arange(0, doc_count + 1, dtype=int),
             np.array(couple_ids),
             is_depressed_group,
             pos_document_A_B[0:: 2],
             pos_document_A_B[1:: 2],
             pos_document_AB),
            axis=0).reshape(
            (6, 10)).transpose()

        pos_document = pd.DataFrame(values, columns=['document_id', 'couple_id', 'is_depressed_group',
                                                     'pos_document_m', 'pos_document_f', 'pos_document_mf'])
        # Convert type to int              
        pos_document['document_id'] = pos_document['document_id'].astype(np.int64)
        pos_document['couple_id'] = pos_document['couple_id'].astype(np.int64)

        # Split Dataframe by gender
        pos_sentence_m = pos_sentence[pos_sentence['gender'] == 'M']
        pos_sentence_f = pos_sentence[pos_sentence['gender'] == 'W']

        pos_paragraph_m = pos_paragraph[pos_paragraph['gender'] == 'M']
        pos_paragraph_f = pos_paragraph[pos_paragraph['gender'] == 'W']

        pos_document_m = pos_document.drop(columns=['pos_document_f', 'pos_document_mf'])
        pos_document_f = pos_document.drop(columns=['pos_document_m', 'pos_document_mf'])
        pos_document_mf = pos_document.drop(columns=['pos_document_m', 'pos_document_f'])

        # Rename colums with respect to gender
        pos_sentence_m = pos_sentence_m.rename(columns={'pos_sentence': 'pos_sentence_m'})
        pos_sentence_f = pos_sentence_f.rename(columns={'pos_sentence': 'pos_sentence_f'})

        pos_paragraph_m = pos_paragraph_m.rename(columns={'pos_paragraph': 'pos_paragraph_m'})
        pos_paragraph_f = pos_paragraph_f.rename(columns={'pos_paragraph': 'pos_paragraph_f'})

        # Convert dictionary of POS (with counts) in a dataframe to separate columns and concatenate with the labels + NaN -> 0
        pos_sentence_m = pd.concat([pos_sentence_m.drop(['pos_sentence_m'], axis=1),
                                    pos_sentence_m['pos_sentence_m'].apply(pd.Series).fillna(0).sort_index( axis=1)], axis=1)

        pos_sentence_f = pd.concat([pos_sentence_f.drop(['pos_sentence_f'], axis=1),
                                    pos_sentence_f['pos_sentence_f'].apply(pd.Series).fillna(0).sort_index(axis=1)], axis=1)

        pos_paragraph_m = pd.concat([pos_paragraph_m.drop(['pos_paragraph_m'], axis=1),
                                    pos_paragraph_m['pos_paragraph_m'].apply(pd.Series).fillna(0).sort_index( axis=1)], axis=1)

        pos_paragraph_f = pd.concat([pos_paragraph_f.drop(['pos_paragraph_f'], axis=1),
                                    pos_paragraph_f['pos_paragraph_f'].apply(pd.Series).fillna(0).sort_index(axis=1)], axis=1)
        
        pos_document_m = pd.concat([pos_document_m.drop(['pos_document_m'], axis=1),
                                     pos_document_m['pos_document_m'].apply(pd.Series).fillna(0).sort_index(axis=1)], axis=1)

        pos_document_f = pd.concat([pos_document_f.drop(['pos_document_f'], axis=1),
                                     pos_document_f['pos_document_f'].apply(pd.Series).fillna(0).sort_index(axis=1)], axis=1)

        pos_document_mf = pd.concat([pos_document_mf.drop(['pos_document_mf'], axis=1),
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

            pos_sentence_m = pos_sentence_m[['couple_id', 'speaker', 'gender', 'is_depressed_group', 'document_id', 'paragraph_id', 'sentence_id'] + self.white_list]
            pos_sentence_f = pos_sentence_f[['couple_id', 'speaker', 'gender', 'is_depressed_group', 'document_id', 'paragraph_id', 'sentence_id'] + self.white_list]

            pos_paragraph_m = pos_paragraph_m[['couple_id', 'speaker', 'gender', 'is_depressed_group', 'document_id', 'paragraph_id'] + self.white_list]
            pos_paragraph_f = pos_paragraph_f[['couple_id', 'speaker', 'gender', 'is_depressed_group', 'document_id', 'paragraph_id'] + self.white_list]

            pos_document_m = pos_document_m[['document_id', 'couple_id', 'is_depressed_group'] + self.white_list]
            pos_document_f = pos_document_f[['document_id', 'couple_id', 'is_depressed_group'] + self.white_list]
            pos_document_mf = pos_document_mf[['document_id', 'couple_id', 'is_depressed_group'] + self.white_list]


        return {POS_DOCUMENT_F: pos_document_f, POS_DOCUMENT_M: pos_document_m, POS_DOCUMENT_MF: pos_document_mf,
                POS_PARAGRAPH_F: pos_paragraph_f, POS_PARAGRAPH_M: pos_paragraph_m,
                POS_SENTENCE_F: pos_sentence_f, POS_SENTENCE_M: pos_sentence_m}

    def _get_pos(self, word_list, nlp):

        #Create Spacy doc from tokens and get a ordered dictionary of the tags with the counts
        df_as_string = ' '.join(word_list)
        doc = nlp(df_as_string)
        pos_list = []
        [pos_list.append(word.tag_) for word in doc]

        pos_list = dict(Counter(pos_list))
        pos_list = OrderedDict(sorted(pos_list.items()))
        return pos_list

    def _get_pos_doc(self, word_list, nlp):
    
        # Join the lists into one list
        res = []
        for str_list in word_list:
            res += str_list

        #Create Spacy doc from tokens and get a ordered dictionary of the tags with the counts
        df_as_string = ' '.join(res)
        doc = nlp(df_as_string)
        pos_list = []
        [pos_list.append(word.tag_) for word in doc]

        pos_list = dict(Counter(pos_list))
        pos_list = OrderedDict(sorted(pos_list.items()))
        return pos_list



class ReadinessToAction(Component):
    R2D = Extension("r2d", float)

    def __init__(self):
        super(ReadinessToAction, self).__init__(
            ReadinessToAction.__name__,
            required_extensions=[PartOfSpeechTags.VERB_COUNT, PartOfSpeechTags.NOUN_COUNT],
            creates_extensions=[ReadinessToAction.R2D]
        )

    def apply(self, doc: Doc) -> Doc:
        verb_count = operator.attrgetter(PartOfSpeechTags.VERB_COUNT.name)(doc._)
        noun_count = operator.attrgetter(PartOfSpeechTags.NOUN_COUNT.name)(doc._)

        doc._.r2d = verb_count / noun_count
        return doc


class PronominalisationIndex(Component):
    PRON_INDEX = Extension("pron_index", float)

    def __init__(self):
        super(PronominalisationIndex, self).__init__(
            PronominalisationIndex.__name__,
            required_extensions=[PartOfSpeechTags.PRON_COUNT, PartOfSpeechTags.NOUN_COUNT,
                                 PartOfSpeechTags.PROPN_COUNT],
            creates_extensions=[PronominalisationIndex.PRON_INDEX]
        )

    def apply(self, doc: Doc) -> Doc:
        pron_count = operator.attrgetter(PartOfSpeechTags.PRON_COUNT.name)(doc._)
        noun_count = operator.attrgetter(PartOfSpeechTags.NOUN_COUNT.name)(doc._)
        propn_count = operator.attrgetter(PartOfSpeechTags.PROPN_COUNT.name)(doc._)

        doc._.pron_index = pron_count / (pron_count + noun_count + propn_count)
        return doc
