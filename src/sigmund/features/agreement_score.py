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

        # List of negative emotions which will be a trigger for disagreement if one word in this category occurs in the paragraph
        disagreement_cat = ['Negate', 'Negemo', 'Discrep', 'Excl']

        tokens_par = TOKENS_PARAGRAPH.load_from(storage=storage)

        # Get number of documents
        doc_count = tokens_par['document_id'].max()
        # Get list of couple_ids
        couple_ids = tokens_par['couple_id'].unique()

        paragraph_count_per_doc = tokens_par.groupby(['document_id'])['paragraph_id'].max()

        tokens_par = tokens_par.groupby(['document_id'])['tokens_paragraph'].apply(list).to_dict()

        agr_score = [0, 0]
        agr_score[0] = np.zeros(doc_count + 1)
        # Get total paragraph size for the normalization of the agreementscore
        agr_score[1] = paragraph_count_per_doc.to_numpy()

        for x in range(doc_count+1):
            for y in range(len(tokens_par[x])):
                if len(tokens_par[x][y]) > 5:
                    # Check if any of the first 5 tokens are in the category of disagreement, set counter +1 if true
                    for category in disagreement_cat:
                        if Counter(category for token in tokens_par[x][y][:5]
                                   for category in parse(token))[category] >= 1:
                            # print(x, y, category, tokens[x][y][:5])
                            agr_score[0][x] += 1
                            break
                else:
                    # Check if any of the tokens are in the category of disagreement, set counter +1 if true
                    for category in disagreement_cat:
                        if Counter(category for token in tokens_par[x][y]
                                   for category in parse(token))[category] >= 1:
                            # print(x, y, category, tokens[x][y][:5])
                            agr_score[0][x] += 1
                            break

        #Calculate disagreement ratio with respect to total paragraph size and substract the result from 1 to get the agreementscore
        agr_score = 1 - np.around(agr_score[0] / agr_score[1], decimals=2)

        # Add is_depressed_group label
        document = queryable.execute(level=TextBody.DOCUMENT)
        is_depressed_group = document['is_depressed_group'].to_numpy()

        #Build Dataframe with columns: document_id, couple_id, is_depressed_group, AgreementScore
        values = np.concatenate(
            (np.arange(0, doc_count + 1, dtype=np.int64),
            couple_ids,
            is_depressed_group,
            agr_score),
            axis=0).reshape((4, 10)).transpose()

        agr_score = pd.DataFrame(values,
                                 columns=['document_id', 'couple_id', 'is_depressed_group', 'AgreementScore'])
        agr_score['is_depressed_group'] = agr_score['is_depressed_group'].astype(bool)

        return {AGREEMENTSCORE: agr_score}
