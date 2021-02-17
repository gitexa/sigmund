import string
from typing import Dict, List

import pandas as pd
import spacy
from nltk.stem.snowball import GermanStemmer
from spacy.tokens import Token

from src.pipelinelib.component import Component
from src.pipelinelib.extension import Extension
from src.pipelinelib.querying import Queryable
from src.pipelinelib.text_body import TextBody
from src.sigmund.extensions import (LEMMATIZED, STEMMED, TOKENS_DOCUMENT,
                                    TOKENS_PARAGRAPHS, TOKENS_SENTENCE)


class Tokenizer(Component):
    """
    Extracts tokens without punctuation from texts.
    """

    def __init__(self):
        super().__init__(Tokenizer.__name__, required_extensions=[],
                         creates_extensions=[TOKENS_SENTENCE])

    def apply(self, storage: Dict[Extension, pd.DataFrame],
              queryable: Queryable) -> Dict[Extension, pd.DataFrame]:
        '''
        # Document
        tokens_doc = queryable.execute(level=TextBody.DOCUMENT)
        tokens_doc = tokens_doc[['document_id', 'text']]
        tokens_doc['text'] = tokens_doc['text'].apply(
            tokenize_df, nlp=queryable.nlp())

        # Paragraphs
        tokens_para = queryable.execute(level=TextBody.PARAGRAPH)
        tokens_para = tokens_para[['document_id', 'paragraph_id', 'text']]
        tokens_para['text'] = tokens_para['text'].apply(
            tokenize_df, nlp=queryable.nlp())
        '''

        # Sentence
        tokens_sent = queryable.execute(level=TextBody.SENTENCE)
        tokens_sent = tokens_sent[['document_id', 'paragraph_id',
                                   'sentence_id', 'text']]
        display(tokens_sent['text'])
        tokens_sent['text'] = tokens_sent['text'].apply(
            tokenize_df, nlp=queryable.nlp(), axis=1)

        # , TOKENS_PARAGRAPHS: tokens_para, TOKENS_DOCUMENT: tokens_doc
        return {TOKENS_SENTENCE: tokens_sent}


def tokenize_df(sentence: str, nlp) -> List[str]:
    tokens = nlp(sentence)
    res = []
    # Go through tokens and check if it is inside the punctuation set
    # If this is the case it will be ignored
    for token in map(str, tokens):
        if not any(p in token for p in string.punctuation):
            res.append(token)

    return res


class Stemmer(Component):
    """
    Performs stemming on the tokens
    """

    def __init__(self, stemmer=GermanStemmer()):
        super(
            Stemmer, self).__init__(
            Stemmer.__name__, required_extensions=[TOKENS_SENTENCE],
            creates_extensions=[STEMMED])
        self._stemmer = stemmer

    def apply(self, storage: Dict[Extension, pd.DataFrame],
              queryable: Queryable) -> Dict[Extension, pd.DataFrame]:
        tokens_df = TOKENS_SENTENCE.load_from(storage=storage)
        stemmed_df = tokens_df.copy()

        stemmed_df["text"] = stemmed_df["text"].apply(self._stemmer.stem)
        return {STEMMED: stemmed_df}


class Lemmatizer(Component):
    """
    Lemmatizes the tokens
    """

    def __init__(self):
        super(
            Lemmatizer, self).__init__(
            Lemmatizer.__name__, required_extensions=[TOKENS_SENTENCE],
            creates_extensions=[LEMMATIZED])

    def apply(self, storage: Dict[Extension, pd.DataFrame],
              queryable: Queryable) -> Dict[Extension, pd.DataFrame]:
        tokens_df = TOKENS_SENTENCE.load_from(storage=storage)
        lemma_df = tokens_df.copy()

        lemma_df["token"] = list(queryable.nlp()(" ".join(lemma_df["token"])))
        lemma_df["token"] = lemma_df["token"].apply(self._f)

        return {LEMMATIZED: lemma_df}

    def _f(self, token: Token) -> list:
        return token.lemma_ if not token.pos_ == "PRON" else token.text
