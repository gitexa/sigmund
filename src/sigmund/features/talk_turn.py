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
from src.sigmund.extensions import TALKTURN, TOKENS

nlp = spacy.load("de_core_news_md")


class TalkTurnExtractor(Component):
    """
    Extracts Talk turn from text and stores these under doc._.talkturn
    """
    talkturn = Extension(name="talkturn")

    def __init__(self):
        super().__init__(name=TalkTurnExtractor.__name__, required_extensions=[],
                         creates_extensions=[TalkTurnExtractor.talkturn])
        self.dic = pyphen.Pyphen(lang='de')

    # def apply(self, doc: Doc) -> Doc:
    def apply(self, storage: Dict[Extension, pd.DataFrame],
              queryable: Queryable) -> Dict[Extension, pd.DataFrame]:

        # Can be replaced when access to TOKENS is clear
        tokens = queryable.execute(level=TextBody.PARAGRAPH)
        tokens = tokens[['document_id', 'paragraph_id', 'speaker', 'text']]
        tokens['text'] = tokens['text'].apply(tokenize_df, nlp=queryable.nlp)

        doc_count = tokens['document_id'].max()
        #paragraph_count_per_doc = tokens.groupby(['document_id'])['paragraph_id'].max()

        tokens['text'] = tokens['text'].apply(len)
        tokens['text'] = tokens['text'].apply(lambda x: 1 if x > 5 else 0)
        tokens = tokens.groupby(['document_id', 'speaker'])['text'].sum()
        tokens = tokens.to_dict()

        talkturns = []
        for x in range(doc_count+1):
            talkturns.append(
                round(
                    tokens[(x, 'A')] / (tokens[(x, 'A')] + tokens[(x, 'B')]),
                    2))

        values = np.concatenate(
            (np.arange(0, doc_count + 1).astype(int),
             np.asarray(talkturns)),
            axis=0).reshape((2, 10)).transpose()

        talkturns = pd.DataFrame(values,
                                 columns=['document_id', 'TalkTurn'])

        return {TALKTURN: talkturns}


def tokenize_df(sentence: str, nlp) -> List[str]:
    tokens = nlp(sentence)
    res = []
    # Go through tokens and check if it is inside the punctuation set
    # If this is the case it will be ignored
    for token in map(str, tokens):
        if not any(p in token for p in string.punctuation):
            res.append(token)

    return res
