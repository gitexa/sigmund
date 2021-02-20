import string
from itertools import filterfalse
from typing import Dict, List

import numpy as np
import pandas as pd
import pyphen
import spacy
from spacy.tokens import Doc

from src.pipelinelib.component import Component
from src.pipelinelib.extension import Extension
from src.pipelinelib.querying import Queryable
from src.pipelinelib.text_body import TextBody
from src.sigmund.extensions import (FRE_DOCUMENT_F, FRE_DOCUMENT_M, FRE_DOCUMENT_MF,
                                    FRE_PARAGRAPH_F, FRE_PARAGRAPH_M, FRE_SENTENCE_F,
                                    FRE_SENTENCE_M)
from src.utils.feature_annotator import reading_ease_german, syllable_counter


class FleschExtractor(Component):
    """
    Calculates the Flesch-Reading-Ease from text and stores these under FRE_DOCUMENT, FRE_PARAGRAPH, FRE_SENTENCE
    """

    def __init__(self):
        super().__init__(name=FleschExtractor.__name__, required_extensions=[],
                         creates_extensions=[FRE_DOCUMENT_M, FRE_DOCUMENT_F, FRE_DOCUMENT_MF, FRE_PARAGRAPH_M, FRE_PARAGRAPH_F, FRE_SENTENCE_M, FRE_SENTENCE_F])
        self.dic = pyphen.Pyphen(lang='de')

    def apply(self, storage: Dict[Extension, pd.DataFrame],
              queryable: Queryable) -> Dict[Extension, pd.DataFrame]:

        fre_sentence = queryable.execute(level=TextBody.SENTENCE)
        fre_paragraph = queryable.execute(level=TextBody.PARAGRAPH)
        fre_document = queryable.execute(level=TextBody.DOCUMENT)

        is_depressed_group = fre_document['is_depressed_group'].to_numpy()
        
        #Filter columns
        fre_sentence = fre_sentence[['document_id', 'couple_id', 'paragraph_id', 'sentence_id', 'gender', 'is_depressed_group', 'text']]
        fre_paragraph = fre_paragraph[['document_id', 'couple_id', 'paragraph_id', 'gender', 'is_depressed_group', 'text']]
        fre_document = fre_paragraph[['document_id', 'couple_id', 'paragraph_id', 'gender', 'is_depressed_group','text']]

        doc_count = fre_document['document_id'].max()
        couple_ids = fre_document['couple_id'].unique()
        
        #Calculate the Flesch-Reading-Ease
        fre_sentence['text'] = fre_sentence['text'].apply(reading_ease_german)
        fre_paragraph['text'] = fre_paragraph['text'].apply(reading_ease_german)

        fre_document_M_F = fre_document.groupby(['document_id', 'gender'])['text'].apply(process_fre)
        fre_document_MF = fre_document.groupby(['document_id'])['text'].apply(process_fre)

        fre_document_M_F = fre_document_M_F.to_numpy()
        fre_document_MF = fre_document_MF.to_numpy()

        #Build Dataframe with columns: document_id, couple_id, is_depressed_group, fre_female, fre_male, fre_both
        values = np.concatenate(
            (np.arange(0, doc_count + 1, dtype=int),
             np.array(couple_ids),
             is_depressed_group,
             fre_document_M_F[0:: 2],
             fre_document_M_F[1:: 2],
             fre_document_MF),
            axis=0).reshape(
            (6, 10)).transpose()

        fre_document = pd.DataFrame(values, columns=['document_id', 'couple_id','is_depressed_group' ,'fre_document_m', 'fre_document_f', 'fre_document_mf'])
        fre_document['document_id'] = fre_document['document_id'].astype(np.int64)
        fre_document['couple_id'] = fre_document['couple_id'].astype(np.int64)
        fre_document['is_depressed_group'] = fre_document['is_depressed_group'].astype(bool)

        # Split Dataframe by gender#
        # Rename colums with respect to gender
        fre_sentence_m = fre_sentence[fre_sentence['gender'] == 'M']
        fre_sentence_f = fre_sentence[fre_sentence['gender'] == 'W']

        fre_sentence_m = fre_sentence_m.rename(columns={'text': 'fre_sentence_m'})
        fre_sentence_f = fre_sentence_f.rename(columns={'text': 'fre_sentence_f'})

        fre_paragraph_m = fre_paragraph[fre_paragraph['gender'] == 'M']
        fre_paragraph_f = fre_paragraph[fre_paragraph['gender'] == 'W']

        fre_paragraph_m = fre_paragraph_m.rename(columns={'text': 'fre_paragraph_m'})
        fre_paragraph_f = fre_paragraph_f.rename(columns={'text': 'fre_paragraph_f'})

        fre_document_m = fre_document.drop(columns= ['fre_document_f', 'fre_document_mf'])
        fre_document_f = fre_document.drop(columns= ['fre_document_m', 'fre_document_mf'])
        fre_document_mf = fre_document.drop(columns= ['fre_document_m', 'fre_document_f'])

        return {FRE_DOCUMENT_M: fre_document_m, FRE_DOCUMENT_F: fre_document_f, FRE_DOCUMENT_MF: fre_document_mf,
                FRE_PARAGRAPH_M: fre_paragraph_m, FRE_PARAGRAPH_F: fre_paragraph_f,
                FRE_SENTENCE_M: fre_sentence_m, FRE_SENTENCE_F: fre_sentence_f}


def process_fre(text):
    return reading_ease_german(' '.join(list(text)))
