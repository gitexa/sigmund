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
from src.sigmund.extensions import FRE_DOCUMENT, FRE_PARAGRAPH, FRE_SENTENCE
from src.utils.feature_annotator import reading_ease_german, syllable_counter


class FleschExtractor(Component):
    """
    Calculates the Flesch-Reading-Ease from text and stores these under doc._.flesch_sentence
    """
    
    def __init__(self):
        super().__init__(name=FleschExtractor.__name__, required_extensions=[],
                         creates_extensions=[FRE_SENTENCE, FRE_PARAGRAPH, FRE_DOCUMENT])
        self.dic = pyphen.Pyphen(lang='de')

    def apply(self, storage: Dict[Extension, pd.DataFrame],
              queryable: Queryable) -> Dict[Extension, pd.DataFrame]:  # Sentence

        fre_sentence = queryable.execute(level=TextBody.SENTENCE)
        fre_paragraph = queryable.execute(level=TextBody.PARAGRAPH)
        fre_document = queryable.execute(level=TextBody.DOCUMENT)

        fre_sentence = fre_sentence[['document_id', 'couple_id', 'paragraph_id', 'sentence_id', 'speaker', 'text']]
        fre_paragraph = fre_paragraph[['document_id', 'couple_id', 'paragraph_id', 'speaker', 'text']]
        fre_document = fre_paragraph[['document_id', 'couple_id', 'paragraph_id', 'speaker', 'text']]

        doc_count = fre_document['document_id'].max()
        couple_ids = fre_document['couple_id'].unique()
        #print(np.array(couple_ids))
        fre_sentence['text'] = fre_sentence['text'].apply(reading_ease_german)
        fre_paragraph['text'] = fre_paragraph['text'].apply(reading_ease_german)

        display(fre_document)
        fre_document_A_B = fre_document.groupby(['document_id', 'speaker'])['text'].apply(process_fre)
        fre_document_AB = fre_document.groupby(['document_id'])['text'].apply(process_fre)
        
        #display(fre_document_AB)
        fre_document_A_B = fre_document_A_B.to_numpy()
        fre_document_AB = fre_document_AB.to_numpy()

        #print(fre_document_AB)
        #print(fre_document[0::2])
        #print(fre_document[1::2])
        values = np.concatenate((np.arange(0, doc_count + 1, dtype=int), np.array(couple_ids) ,fre_document_A_B[0::2], fre_document_A_B[1::2], fre_document_AB), axis=0).reshape((5, 10)).transpose()
        #print(values)

        fre_document = pd.DataFrame(values, columns=['document_id', 'couple_id', 'FRE_A', 'FRE_B', 'FRE_AB'])
        fre_document['document_id'] = fre_document['document_id'].astype(np.int64)
        fre_document['couple_id'] = fre_document['couple_id'].astype(np.int64)

        fre_sentence = fre_sentence.rename(columns={'text': 'FRE_sentence'})
        fre_paragraph = fre_paragraph.rename(columns={'text': 'FRE_paragraph'})

        return {FRE_SENTENCE: fre_sentence, FRE_PARAGRAPH: fre_paragraph, FRE_DOCUMENT: fre_document}


def process_fre(text):
    print("pro apply:::::",text)
    return reading_ease_german(' '.join(list(text)))
