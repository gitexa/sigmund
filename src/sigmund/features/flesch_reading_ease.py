import string
from typing import Dict

import numpy as np
import pandas as pd
import pyphen

from src.pipelinelib.component import Component
from src.pipelinelib.extension import Extension
from src.pipelinelib.querying import Queryable
from src.pipelinelib.text_body import TextBody
from src.sigmund.extensions import (FRE_DOCUMENT_F, FRE_DOCUMENT_M, FRE_DOCUMENT_MF,
                                    FRE_PARAGRAPH_F, FRE_PARAGRAPH_M, FRE_SENTENCE_F,
                                    FRE_SENTENCE_M)


class FleschExtractor(Component):
    '''
    Calculates the Flesch-Reading-Ease from the different text layers (document,paragraphs,sentences) 
    and stores these under FRE_DOCUMENT, FRE_PARAGRAPH, FRE_SENTENCE.
    The Flesch-Reading-Ease evaluats the readability of the text.
    A score indicates simple text structure and readability and higher scores indicate more complex text structures.
    '''

    def __init__(self):
        super().__init__(name=FleschExtractor.__name__, required_extensions=[],
                         creates_extensions=[FRE_DOCUMENT_M, FRE_DOCUMENT_F, FRE_DOCUMENT_MF, FRE_PARAGRAPH_M,
                                             FRE_PARAGRAPH_F, FRE_SENTENCE_M, FRE_SENTENCE_F])
        self.dic = pyphen.Pyphen(lang='de')

    def apply(self, storage: Dict[Extension, pd.DataFrame],
              queryable: Queryable) -> Dict[Extension, pd.DataFrame]:

        fre_sentence = queryable.execute(level=TextBody.SENTENCE)
        fre_paragraph = queryable.execute(level=TextBody.PARAGRAPH)
        # fre_document = queryable.execute(level=TextBody.DOCUMENT)

        # Filter columns
        fre_sentence = fre_sentence[['document_id', 'couple_id', 'paragraph_id', 'sentence_id', 'gender', 'text']]
        fre_paragraph = fre_paragraph[['document_id', 'couple_id', 'paragraph_id', 'gender', 'text']]
        fre_document = fre_paragraph[['document_id', 'couple_id', 'paragraph_id', 'gender', 'text']]

        doc_count = fre_document['document_id'].max()
        couple_ids = fre_document['couple_id'].unique()

        # Calculate the Flesch-Reading-Ease
        fre_sentence['text'] = fre_sentence['text'].apply(self._reading_ease_german, nlp=queryable.nlp())
        fre_paragraph['text'] = fre_paragraph['text'].apply(self._reading_ease_german, nlp=queryable.nlp())

        fre_document_M_F = fre_document.groupby(['document_id', 'gender'])['text'].apply(self._process_fre)
        fre_document_MF = fre_document.groupby(['document_id'])['text'].apply(self._process_fre)

        fre_document_M_F = fre_document_M_F.to_numpy()
        fre_document_MF = fre_document_MF.to_numpy()

        # Build Dataframe with columns: document_id, couple_id, is_depressed_group, fre_female, fre_male, fre_both
        values = np.concatenate(
            (np.arange(0, doc_count + 1, dtype=int),
             np.array(couple_ids),
             fre_document_M_F[0:: 2],
             fre_document_M_F[1:: 2],
             fre_document_MF),
            axis=0).reshape(
            (5, 10)).transpose()

        fre_document = pd.DataFrame(values, columns=['document_id', 'couple_id', 'fre_document_m', 'fre_document_f',
                                                     'fre_document_mf'])
        fre_document['document_id'] = fre_document['document_id'].astype(np.int64)
        fre_document['couple_id'] = fre_document['couple_id'].astype(np.int64)

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

        fre_document_m = fre_document.drop(columns=['fre_document_f', 'fre_document_mf'])
        fre_document_f = fre_document.drop(columns=['fre_document_m', 'fre_document_mf'])
        fre_document_mf = fre_document.drop(columns=['fre_document_m', 'fre_document_f'])

        # Aggregate over sentences and paragraphs to get document features and drop gender

        fre_sentence_m = fre_sentence_m.drop(columns=['gender', 'paragraph_id', 'sentence_id'])
        fre_sentence_f = fre_sentence_f.drop(columns=['gender', 'paragraph_id', 'sentence_id'])
        fre_paragraph_m = fre_paragraph_m.drop(columns=['gender', 'paragraph_id'])
        fre_paragraph_f = fre_paragraph_f.drop(columns=['gender', 'paragraph_id'])

        fre_sentence_m = fre_sentence_m.groupby(['document_id', 'couple_id']).agg('mean').reset_index()
        fre_sentence_f = fre_sentence_f.groupby(['document_id', 'couple_id']).agg('mean').reset_index()
        fre_paragraph_m = fre_paragraph_m.groupby(['document_id', 'couple_id']).agg('mean').reset_index()
        fre_paragraph_f = fre_paragraph_f.groupby(['document_id', 'couple_id']).agg('mean').reset_index()

        return {FRE_DOCUMENT_M: fre_document_m, FRE_DOCUMENT_F: fre_document_f, FRE_DOCUMENT_MF: fre_document_mf,
                FRE_PARAGRAPH_M: fre_paragraph_m, FRE_PARAGRAPH_F: fre_paragraph_f,
                FRE_SENTENCE_M: fre_sentence_m, FRE_SENTENCE_F: fre_sentence_f}

    def _process_fre(self, text, nlp):
        return self._reading_ease_german(' '.join(list(text)), nlp)

    def _syllable_counter(self, text, nlp):
        doc = nlp(text)
        syllable_count = 0
        word_count = 0
        sent_count = len(list(doc.sents))

        for token in doc:
            if not str(token) in string.punctuation:
                word_count += 1
                syllable_count += len(self.dic.inserted(str(token)).split("-"))

        return word_count, syllable_count, sent_count

    # Adopted formula to calculate Flesh reading ease for german

    def _reading_ease_german(self, text, nlp):
        word_count, syllable_count, sent_count = self._syllable_counter(text, nlp)
        if syllable_count == 0 or word_count == 0 or sent_count == 0:
            return 0
        score = 180 - (word_count / sent_count) - (58.5 * (syllable_count / word_count))
        return score
