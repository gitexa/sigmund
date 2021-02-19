import re
import string
from collections import Counter
from itertools import filterfalse
from typing import Dict, List

import liwc
import numpy as np
import pandas as pd
import pyphen
import spacy
from spacy.tokens import Doc

from src.pipelinelib.component import Component
from src.pipelinelib.extension import Extension
from src.pipelinelib.querying import Queryable
from src.pipelinelib.text_body import TextBody
from src.sigmund.extensions import AGREEMENTSCORE, TOKENS_PARAGRAPH


class AgreementScoreExtractor(Component):
    """
    Extracts the Agreement-Score turn from text and stores these under AGREEMENTSCORE
    """

    def __init__(self, dictionary_path="./data/German_LIWC2001_Dictionary.dic"):
        super().__init__(
            name=AgreementScoreExtractor.__name__,
            required_extensions=[TOKENS_PARAGRAPH],
            creates_extensions=[AGREEMENTSCORE])
        self.dic = pyphen.Pyphen(lang='de')
        self._dictionary_path = dictionary_path

    def apply(self, storage: Dict[Extension, pd.DataFrame],
              queryable: Queryable) -> Dict[Extension, pd.DataFrame]:

        # Load LIWC Dictionary provided by path
        parse, category_names = liwc.load_token_parser(self._dictionary_path)
        # neg : negate(7) ,negemo(16),  23	Discrep,  45	Excl
        disagreement_cat = ['Negate', 'Negemo', 'Discrep', 'Excl']

        tokens_par = TOKENS_PARAGRAPH.load_from(storage=storage)

        # Get number of documents
        doc_count = tokens_par['document_id'].max()
        # Get list of couple_ids
        couple_ids = tokens_par['couple_id'].unique()

        paragraph_count_per_doc = tokens_par.groupby(
            ['document_id'])['paragraph_id'].max()

        tokens_par = tokens_par.groupby(['document_id'])['text'].apply(list).to_dict()

        agr_score = [0, 0]
        agr_score[0] = np.zeros(doc_count + 1)
        agr_score[1] = paragraph_count_per_doc.to_numpy()

        for x in range(doc_count+1):
            for y in range(len(tokens_par[x])):
                if len(tokens_par[x][y]) > 5:

                    for category in disagreement_cat:
                        if Counter(category for token in tokens_par[x][y][:5]
                                   for category in parse(token))[category] >= 1:
                            # print(x, y, category, tokens[x][y][:5])
                            agr_score[0][x] += 1
                            break
                else:
                    for category in disagreement_cat:
                        if Counter(category for token in tokens_par[x][y]
                                   for category in parse(token))[category] >= 1:
                            # print(x, y, category, tokens[x][y][:5])
                            agr_score[0][x] += 1
                            break

        agr_score = 1 - np.around(agr_score[0] / agr_score[1], decimals=2)

        # Prepare data for Dataframe
        values = np.concatenate(
            (np.arange(0, doc_count + 1, dtype=np.int64), couple_ids,
             agr_score),
            axis=0).reshape((3, 10)).transpose()

        agr_score = pd.DataFrame(values,
                                 columns=['document_id', 'couple_id', 'AgreementScore'])

        return {AGREEMENTSCORE: agr_score}
