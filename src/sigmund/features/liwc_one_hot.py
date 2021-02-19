import operator
import string
from collections import Counter
from typing import Dict, List

import liwc
import numpy as np
import pandas as pd
from spacy.tokens.doc import Doc

from src.pipelinelib.component import Component
from src.pipelinelib.extension import Extension
from src.pipelinelib.querying import Queryable
from src.pipelinelib.text_body import TextBody
from src.sigmund.extensions import (LOH_DOCUMENT, LOH_PARAGRAPH, LOH_SENTENCE,
                                    TOKENS_PARAGRAPH, TOKENS_SENTENCE)
from src.sigmund.preprocessing.words import Tokenizer
from src.utils.liwc import Liwc


class LiwcOneHot(Component):

    def __init__(self, token_parser_path="./data/German_LIWC2001_Dictionary.dic"):
        super(LiwcOneHot, self).__init__(
            LiwcOneHot.__name__,
            required_extensions=[TOKENS_PARAGRAPH, TOKENS_SENTENCE],
            creates_extensions=[LOH_SENTENCE, LOH_PARAGRAPH, LOH_DOCUMENT]
        )
        # self.parse, self.category_names = liwc.load_token_parser(token_parser_path)#Liwc(token_parser_path)

    def apply(self, storage: Dict[Extension, pd.DataFrame],
              queryable: Queryable) -> Dict[Extension, pd.DataFrame]:  # Sentence

        tokens_sent = TOKENS_SENTENCE.load_from(storage=storage)
        tokens_par = TOKENS_PARAGRAPH.load_from(storage=storage)
        tokens_doc = TOKENS_PARAGRAPH.load_from(storage=storage)

        doc_count = tokens_doc['document_id'].max()
        couple_ids = tokens_doc['couple_id'].unique()

        cat_per_tokens_sent = tokens_sent
        cat_per_tokens_par = tokens_par
        cat_per_tokens_doc = tokens_doc
        parse, category_names = liwc.load_token_parser(
            "./data/German_LIWC2001_Dictionary.dic")

        cat_per_tokens_sent['text'] = cat_per_tokens_sent['text'].apply(
            liwc_parser, parse=parse, category=category_names)
        cat_per_tokens_sent = cat_per_tokens_sent.rename(
            columns={'text': 'LOH_sentences'})

        cat_per_tokens_par['text'] = cat_per_tokens_par['text'].apply(
            liwc_parser, parse=parse, category=category_names)
        cat_per_tokens_par = cat_per_tokens_par.rename(
            columns={'text': 'LOH_paragraph'})

        LOH_document_A_B = cat_per_tokens_doc.groupby(['document_id', 'speaker'])[
            'text'].apply(list).apply(liwc_parser_doc, parse=parse,
                                      category=category_names)

        LOH_document_AB = cat_per_tokens_doc.groupby(['document_id'])[
            'text'].apply(list).apply(liwc_parser_doc, parse=parse,
                                      category=category_names)

        LOH_document_A_B = LOH_document_A_B.to_numpy()
        LOH_document_AB = LOH_document_AB.to_numpy()

        values = np.concatenate(
            (np.arange(0, doc_count + 1, dtype=int),
             np.array(couple_ids),
             LOH_document_A_B[0:: 2],
             LOH_document_A_B[1:: 2],
             LOH_document_AB),
            axis=0).reshape(
            (5, 10)).transpose()

        LOH_document = pd.DataFrame(
            values, columns=['document_id', 'couple_id', 'LOH_A', 'LOH_B', 'LOH_AB'])
        LOH_document['document_id'] = LOH_document['document_id'].astype(np.int64)
        LOH_document['couple_id'] = LOH_document['couple_id'].astype(np.int64)

        return {LOH_SENTENCE: cat_per_tokens_sent, LOH_PARAGRAPH: cat_per_tokens_par, LOH_DOCUMENT: LOH_document}


def liwc_parser(tokens, parse, category):

    liwc_cats = Counter(category for token in tokens for category in parse(token))
    return sorted(liwc_cats.items(), key=lambda item: item[1], reverse=True)


def liwc_parser_doc(tokens, parse, category):

    #counter = 0
    # for x in range(len(tokens)):

    #    x -= counter
    #    if tokens[x] == []:
    #        del tokens[x]
    #        counter += 1
    # print(tokens)
    res = []
    for str_list in tokens:
        res += str_list

    liwc_cats = Counter(category for token in res for category in parse(token))
    return sorted(liwc_cats.items(), key=lambda item: item[1], reverse=True)
