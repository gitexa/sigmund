import string
from typing import Dict

import pandas as pd
from nltk.stem.snowball import GermanStemmer
from spacy.tokens import Token

from src.pipelinelib.component import Component
from src.pipelinelib.extension import Extension
from src.pipelinelib.text_body import TextBody
from src.sigmund.extensions import LEMMATIZED, STEMMED, TOKENS
from src.utils.querying import Queryable


class Tokenizer(Component):
    """
    Extracts tokens without punctuation from texts.
    """

    def __init__(self):
        super().__init__(Tokenizer.__name__, required_extensions=[],
                         creates_extensions=[TOKENS])

    def apply(self, storage: Dict[Extension, pd.DataFrame],
              queryable: Queryable) -> Dict[Extension, pd.DataFrame]:
        sentences = queryable.execute(level=TextBody.SENTENCE)

        tokens = (
            (sid, token) for sid, sentence in sentences[["uid", "text"]]
            for token in sentence)
        tokens = list((sid, token) for(sid, token) in tokens
                      if not any(p in token for p in string.punctuation))

        tokens_df = pd.DataFrame(tokens, columns=("sentence_id", "token"))
        return {TOKENS: tokens_df}


class Stemmer(Component):
    """
    Performs stemming on the tokens
    """

    def __init__(self, stemmer=GermanStemmer()):
        super(
            Stemmer, self).__init__(
            Stemmer.__name__, required_extensions=[TOKENS],
            creates_extensions=[STEMMED])
        self._stemmer = stemmer

    def apply(self, storage: Dict[Extension, pd.DataFrame],
              queryable: Queryable) -> Dict[Extension, pd.DataFrame]:
        tokens_df = TOKENS.load_from(storage=storage)
        stemmed_df = tokens_df.copy()

        stemmed_df["token"].apply(self._stemmer.stem)
        return {STEMMED: stemmed_df}


class Lemmatizer(Component):
    """
    Lemmatizes the tokens
    """

    def __init__(self):
        super(
            Lemmatizer, self).__init__(
            Lemmatizer.__name__, required_extensions=[TOKENS],
            creates_extensions=[LEMMATIZED])

    def apply(self, storage: Dict[Extension, pd.DataFrame],
              queryable: Queryable) -> Dict[Extension, pd.DataFrame]:
        tokens_df = TOKENS.load_from(storage=storage)
        lemma_df = tokens_df.copy()

        lemma_df["token"] = list(queryable.nlp()(" ".join(lemma_df["token"])))
        lemma_df["token"] = lemma_df["token"].apply(self._f)

        return {LEMMATIZED: lemma_df}

    def _f(self, token: Token) -> list:
        return token.lemma_ if not token.pos_ == "PRON" else token.text
