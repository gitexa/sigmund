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
            creates_extensions=[LOH_PARAGRAPH, LOH_SENTENCE]
        )
        # self.parse, self.category_names = liwc.load_token_parser(token_parser_path)#Liwc(token_parser_path)

    def apply(self, storage: Dict[Extension, pd.DataFrame],
              queryable: Queryable) -> Dict[Extension, pd.DataFrame]:  # Sentence

        tokens_sent = TOKENS_SENTENCE.load_from(storage=storage)
        tokens_par  = TOKENS_PARAGRAPH.load_from(storage=storage)
        tokens_doc  = TOKENS_PARAGRAPH.load_from(storage=storage)

        doc_count = tokens_doc['document_id'].max()
        couple_ids = tokens_doc['couple_id'].unique()

        cat_per_tokens_sent = tokens_sent
        cat_per_tokens_par = tokens_par
        cat_per_tokens_doc = tokens_doc
        parse, category_names = liwc.load_token_parser("./data/German_LIWC2001_Dictionary.dic")

        display(cat_per_tokens_doc)

        cat_per_tokens_sent['text'] = cat_per_tokens_sent['text'].apply(
            liwc_parser, parse=parse, category=category_names)
        cat_per_tokens_sent = cat_per_tokens_sent.rename(columns={'text': 'LOH_sentences'})

        cat_per_tokens_par['text'] = cat_per_tokens_par['text'].apply(
            liwc_parser, parse=parse, category=category_names)
        cat_per_tokens_par = cat_per_tokens_par.rename(columns={'text': 'LOH_paragraph'})

        #display(cat_per_tokens_doc.groupby(['document_id', 'speaker'])['text'].to_dict())
        #cat_per_tokens_doc = cat_per_tokens_doc[cat_per_tokens_doc.text != []]
        indexNames = cat_per_tokens_doc[ cat_per_tokens_doc['text'] == [] ].index
        #cat_per_tokens_doc = cat_per_tokens_doc.drop(cat_per_tokens_doc[cat_per_tokens_doc.text.empty].index)
        fre_document_A_B = cat_per_tokens_doc.groupby(['document_id', 'speaker'])['text'].apply(list).apply(process_fre)#liwc_parser, parse=parse, category=category_names)
        display(fre_document_A_B)
        #fre_document_A_B = fre_document_A_B.groupby(['document_id', 'speaker'])['text'].apply( liwc_parser, parse=parse, category=category_names)

        #display(fre_document_A_B)
        #fre_document_AB = cat_per_tokens_doc.groupby(['document_id'])['text'].apply(
        #    liwc_parser, parse=parse, category=category_names)

        #display(fre_document_A_B)
        #display(fre_document_AB)
        #fre_document_A_B = fre_document_A_B.to_numpy()
        #fre_document_AB = fre_document_AB.to_numpy()

        return {LOH_SENTENCE: cat_per_tokens_sent, LOH_PARAGRAPH: cat_per_tokens_par}


def liwc_parser(tokens, parse, category):
    # Gets category for each token

    #print("per apply::::",tokens)
    #tokens = np.array(tokens).flatten()
    #print(tokens)
    liwc_cats = Counter(category for token in tokens for category in parse(token))
    return liwc_cats

def process_fre(text):
    #for 
    #text = np.array(text).flatten().flatten().flatten()
    #print("per apply::::::::",text)
    return text


def tokenize_df(sentence: str, nlp) -> List[str]:
    tokens = nlp(sentence)
    res = []
    # Go through tokens and check if it is inside the punctuation set
    # If this is the case it will be ignored
    for token in map(str, tokens):
        if not any(p in token for p in string.punctuation):
            res.append(token.lower())

    return res
