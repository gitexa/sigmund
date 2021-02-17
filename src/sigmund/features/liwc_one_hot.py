import operator
import string
from collections import Counter
from typing import Dict, List

import liwc
import pandas as pd
from spacy.tokens.doc import Doc

from src.pipelinelib.component import Component
from src.pipelinelib.extension import Extension
from src.pipelinelib.querying import Queryable
from src.pipelinelib.text_body import TextBody
from src.sigmund.extensions import LIWCONEHOT
#from src.sigmund.extensions import LEMMATIZED, STEMMED, TOKENS
from src.sigmund.preprocessing.words import Tokenizer
from src.utils.liwc import Liwc


class LiwcOneHot(Component):
    LIWC_ONE_HOT_VEC = Extension("loh")  # , dict())

    def __init__(self, token_parser_path="./data/German_LIWC2001_Dictionary.dic"):
        super(LiwcOneHot, self).__init__(
            LiwcOneHot.__name__,
            required_extensions=[],  # Tokenizer.TOKENS],
            creates_extensions=[LiwcOneHot.LIWC_ONE_HOT_VEC]
        )
        #self.parse, self.category_names = liwc.load_token_parser(token_parser_path)#Liwc(token_parser_path)

    # def apply(self, doc: Doc) -> Doc:
    #    import operator
    #    getter = operator.attrgetter(Tokenizer.TOKENS.name)
    #    tokens = getter(doc._)

    #    doc._.loh = self.liwc.parse(tokens)

    #    return doc

    def apply(self, storage: Dict[Extension, pd.DataFrame],
              queryable: Queryable) -> Dict[Extension, pd.DataFrame]:  # Sentence

        # Can be replaced when access to TOKENS is clear
        tokens = queryable.execute(level=TextBody.SENTENCE)
        tokens = tokens[['document_id', 'paragraph_id',
                         'sentence_id', 'speaker', 'text']]
        tokens['text'] = tokens['text'].apply(tokenize_df, nlp=queryable.nlp)

        cat_per_tokens = tokens
        parse, category_names = liwc.load_token_parser("./data/German_LIWC2001_Dictionary.dic")
        cat_per_tokens['text'] = cat_per_tokens['text'].apply(liwc_parser, parse = parse, category = category_names)
        cat_per_tokens = cat_per_tokens.rename(columns={'text': 'LiwcOneHot'})

        return {LIWCONEHOT: cat_per_tokens}


def liwc_parser(tokens, parse, category):
    #Gets category for each token
    #licw_dict_per_token = {}
    #for token in tokens:
        #licw_dict_per_token[token] = list(parse(token))
        #print(parse(token))

    liwc_cats = Counter(category for token in tokens for category in parse(token))
    return liwc_cats

def tokenize_df(sentence: str, nlp) -> List[str]:
    tokens = nlp(sentence)
    res = []
    # Go through tokens and check if it is inside the punctuation set
    # If this is the case it will be ignored
    for token in map(str, tokens):
        if not any(p in token for p in string.punctuation):
            res.append(token.lower())

    return res
