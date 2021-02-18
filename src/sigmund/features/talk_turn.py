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
from src.sigmund.extensions import TALKTURN, TOKENS_PARAGRAPH

nlp = spacy.load("de_core_news_md")


class TalkTurnExtractor(Component):
    """
    Extracts Talk turn from text and stores these under doc._.talkturn
    """
    #talkturn = Extension(name="talkturn")

    def __init__(self):
        super().__init__(name=TalkTurnExtractor.__name__, required_extensions=[TOKENS_PARAGRAPH],
                         creates_extensions=[TALKTURN])
        self.dic = pyphen.Pyphen(lang='de')

    # def apply(self, doc: Doc) -> Doc:
    def apply(self, storage: Dict[Extension, pd.DataFrame],
              queryable: Queryable) -> Dict[Extension, pd.DataFrame]:

        tokens_par = TOKENS_PARAGRAPH.load_from(storage=storage)
        doc_count = tokens_par['document_id'].max()
        couple_ids = tokens_par['couple_id'].unique()

        #paragraph_count_per_doc = tokens.groupby(['document_id'])['paragraph_id'].max()

        tokens_par['text'] = tokens_par['text'].apply(len)
        tokens_par['text'] = tokens_par['text'].apply(lambda x: 1 if x > 5 else 0)
        tokens_par = tokens_par.groupby(['document_id', 'speaker'])['text'].sum()
        tokens_par = tokens_par.to_dict()

        talkturns = []
        for x in range(doc_count+1):
            talkturns.append(
                round(
                    tokens_par[(x, 'A')] / (tokens_par[(x, 'A')] + tokens_par[(x, 'B')]),
                    2))

        values = np.concatenate(
            (np.arange(0, doc_count + 1, dtype=np.int64), couple_ids,
             np.asarray(talkturns)),
            axis=0).reshape((3, 10)).transpose()

        talkturns = pd.DataFrame(values,
                                 columns=['document_id', 'couple_id', 'TalkTurn'])

        return {TALKTURN: talkturns}
