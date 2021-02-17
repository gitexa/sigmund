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
from src.sigmund.extensions import (LEMMATIZED, STEMMED_DOCUMENT, STEMMED_PARAGRAPH,
                                    STEMMED_SENTENCE, TOKENS_DOCUMENT, TOKENS_PARAGRAPH,
                                    TOKENS_SENTENCE)


class Tokenizer(Component):
    """
    Extracts tokens without punctuation from texts.
    """

    def __init__(self):
        super().__init__(Tokenizer.__name__, required_extensions=[],
                         creates_extensions=[TOKENS_SENTENCE, TOKENS_PARAGRAPH, TOKENS_DOCUMENT])

    def apply(self, storage: Dict[Extension, pd.DataFrame],
              queryable: Queryable) -> Dict[Extension, pd.DataFrame]:

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

        # Sentence
        tokens_sent = queryable.execute(level=TextBody.SENTENCE)
        tokens_sent = tokens_sent[['document_id', 'paragraph_id',
                                   'sentence_id', 'text']]
        tokens_sent['text'] = tokens_sent['text'].apply(
            tokenize_df, nlp=queryable.nlp())

        return {TOKENS_SENTENCE: tokens_sent, TOKENS_PARAGRAPH: tokens_para, TOKENS_DOCUMENT: tokens_doc}


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
        super().__init__(
            Stemmer.__name__,
            required_extensions=[TOKENS_SENTENCE, TOKENS_PARAGRAPH, TOKENS_DOCUMENT],
            creates_extensions=[STEMMED_DOCUMENT, STEMMED_PARAGRAPH, STEMMED_SENTENCE])

        self._stemmer = stemmer

    def apply(self, storage: Dict[Extension, pd.DataFrame],
              queryable: Queryable) -> Dict[Extension, pd.DataFrame]:

        # Document
        df_tokens_document = TOKENS_DOCUMENT.load_from(storage=storage)
        df_stemmed_document = df_tokens_document.copy()
        df_stemmed_document["text"] = df_stemmed_document["text"].apply(
            self._stemmer.stem)

        # Paragraph
        df_tokens_paragraph = TOKENS_PARAGRAPH.load_from(storage=storage)
        df_stemmed_paragraph = df_tokens_paragraph.copy()
        df_stemmed_paragraph["text"] = df_stemmed_paragraph["text"].apply(
            self._stemmer.stem)

        # Sentence
        df_tokens_sentence = TOKENS_SENTENCE.load_from(storage=storage)
        df_stemmed_sentence = df_tokens_sentence.copy()
        df_stemmed_sentence["text"] = df_stemmed_sentence["text"].apply(
            self._stemmer.stem)

        return {STEMMED_SENTENCE: df_stemmed_sentence, STEMMED_PARAGRAPH: df_stemmed_paragraph, STEMMED_DOCUMENT: df_stemmed_document}


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
