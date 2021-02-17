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
from src.sigmund.extensions import FLESCHREADINGEASE, TOKENS
from src.utils.feature_annotator import reading_ease_german, syllable_counter


class FleschExtractor(Component):
    """
    Calculates the Flesch-Reading-Ease from text and stores these under doc._.flesch_sentence
    """
    FLESCH_SENTENCE = Extension(name="flesch_sentence")

    def __init__(self):
        super().__init__(name=FleschExtractor.__name__, required_extensions=[],
                         creates_extensions=[FleschExtractor.FLESCH_SENTENCE])
        self.dic = pyphen.Pyphen(lang='de')

    def apply(self, storage: Dict[Extension, pd.DataFrame],
              queryable: Queryable) -> Dict[Extension, pd.DataFrame]:  # Sentence
        # print(doc.text)
        tokens_df = TOKENS.load_from(storage=storage)
        display(tokens_df)

        # Can be replaced when access to TOKENS is clear
        fre = queryable.execute(level=TextBody.SENTENCE)
        fre = fre[['document_id', 'paragraph_id',
                   'sentence_id', 'speaker', 'text']]

        fre['text'] = fre['text'].apply(reading_ease_german)
        fre = fre.rename(columns={'text': 'FleschReadingEase'})

        return {FLESCHREADINGEASE: fre}


def tokenize_df(sentence: str, nlp) -> List[str]:
    tokens = nlp(sentence)
    res = []
    # Go through tokens and check if it is inside the punctuation set
    # If this is the case it will be ignored
    for token in map(str, tokens):
        if not any(p in token for p in string.punctuation):
            res.append(token.lower())

    return res


'''
    def apply(self, doc: Doc) -> Doc:  # Paragraph

        doc._.flesch_sentence = reading_ease_german(doc)
        return doc

    def apply(self, doc: Doc) -> Doc:  # Dialog

        doc._.flesch_sentence = reading_ease_german(doc)
        return doc
'''
