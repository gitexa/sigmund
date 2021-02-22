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
    Extracts Talk turn from text and stores these under TALKTURN
    """

    def __init__(self):
        super().__init__(name=TalkTurnExtractor.__name__,
                         required_extensions=[TOKENS_PARAGRAPH],
                         creates_extensions=[TALKTURN])
        self.dic = pyphen.Pyphen(lang='de')

    def apply(self, storage: Dict[Extension, pd.DataFrame],
              queryable: Queryable) -> Dict[Extension, pd.DataFrame]:

        tokens_par = TOKENS_PARAGRAPH.load_from(storage=storage)
        doc_count = tokens_par['document_id'].max()
        couple_ids = tokens_par['couple_id'].unique()

        # If a paragraph has more then 5 tokens it is count as a talking turn
        tokens_par['tokens_paragraph'] = tokens_par['tokens_paragraph'].apply(len)
        tokens_par['tokens_paragraph'] = tokens_par['tokens_paragraph'].apply(lambda x: 1 if x > 5 else 0)
        tokens_par = tokens_par.groupby(['document_id', 'speaker'])['tokens_paragraph'].sum()
        tokens_par = tokens_par.to_dict()

        # Calculate Talkturn ratio (with respect to the value for man) for all the documents (Man/ (Man+Woman))
        talkturns = []
        for x in range(doc_count + 1):
            talkturns.append(
                round(tokens_par[(x, 'A')] / (tokens_par[(x, 'A')] + tokens_par[(x, 'B')]), 2))

        # Add is_depressed_group label
        document = queryable.execute(level=TextBody.DOCUMENT)
        is_depressed_group = document['is_depressed_group'].to_numpy()

        # Build Dataframe with columns: document_id, couple_id, is_depressed_group, TalkTurn
        values = np.concatenate(
            (np.arange(0, doc_count + 1, dtype=np.int64),
                couple_ids,
                is_depressed_group,
                np.asarray(talkturns)),
            axis=0).reshape((4, 10)).transpose()

        talkturns = pd.DataFrame(values,
                                 columns=['document_id', 'couple_id', 'is_depressed_group', 'TalkTurn'])

        talkturns['is_depressed_group'] = talkturns['is_depressed_group'].astype(bool)

        return {TALKTURN: talkturns}
