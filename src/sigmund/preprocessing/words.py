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
from src.sigmund.extensions import (LEMMATIZED_DOCUMENT, LEMMATIZED_PARAGRAPH,
                                    LEMMATIZED_SENTENCE, STEMMED_DOCUMENT,
                                    STEMMED_PARAGRAPH, STEMMED_SENTENCE,
                                    TOKENS_DOCUMENT, TOKENS_PARAGRAPH, TOKENS_SENTENCE)


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
        tokens_doc = tokens_doc[['couple_id',
                                 'is_depressed_group', 'document_id', 'text']]
        tokens_doc['text'] = tokens_doc['text'].apply(
            self.tokenize_df, nlp=queryable.nlp())

        # Paragraphs
        tokens_para = queryable.execute(level=TextBody.PARAGRAPH)
        tokens_para = tokens_para[['couple_id', 'speaker', 'gender',
                                   'is_depressed_group', 'document_id', 'paragraph_id', 'text']]
        tokens_para['text'] = tokens_para['text'].apply(
            self.tokenize_df, nlp=queryable.nlp())

        # Sentence
        tokens_sent = queryable.execute(level=TextBody.SENTENCE)
        tokens_sent = tokens_sent[['couple_id', 'speaker', 'gender',
                                   'is_depressed_group', 'document_id', 'paragraph_id',
                                   'sentence_id', 'text']]
        tokens_sent['text'] = tokens_sent['text'].apply(
            self.tokenize_df, nlp=queryable.nlp())

        tokens_sent = tokens_sent.rename(columns={'text': 'tokens_sentence'})
        tokens_para = tokens_para.rename(columns={'text': 'tokens_paragraph'})
        tokens_doc = tokens_doc.rename(columns={'text': 'tokens_document'})

        return {TOKENS_SENTENCE: tokens_sent, TOKENS_PARAGRAPH: tokens_para, TOKENS_DOCUMENT: tokens_doc}

    def tokenize_df(self, text: str, nlp) -> List[str]:
        tokens = nlp(text)
        res = []
        # Go through tokens and check if it is inside the punctuation set
        # If this is the case it will be ignored
        for token in map(str, tokens):
            if (not any(p in token for p in string.punctuation)) and len(token) > 1:
                res.append(token.lower())
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
        df_stemmed_document = TOKENS_DOCUMENT.load_from(storage=storage)
        df_stemmed_document["text"] = df_stemmed_document["text"].apply(
            self.stemm_df, stemmer=self._stemmer)

        # Paragraph
        df_stemmed_paragraph = TOKENS_PARAGRAPH.load_from(storage=storage)
        df_stemmed_paragraph["text"] = df_stemmed_paragraph["text"].apply(
            self.stemm_df, stemmer=self._stemmer)

        # Sentence
        df_stemmed_sentence = TOKENS_SENTENCE.load_from(storage=storage)
        df_stemmed_sentence["text"] = df_stemmed_sentence["text"].apply(
            self.stemm_df, stemmer=self._stemmer)

        return {STEMMED_SENTENCE: df_stemmed_sentence, STEMMED_PARAGRAPH: df_stemmed_paragraph, STEMMED_DOCUMENT: df_stemmed_document}

    def stemm_df(self, text: str, stemmer) -> List[str]:
        res = []
        for token in map(str, text):
            res.append(stemmer.stem(token))
        return res


class Lemmatizer(Component):
    """
    Lemmatizes the tokens
    """

    def __init__(self):
        super().__init__(
            Lemmatizer.__name__,
            required_extensions=[TOKENS_SENTENCE, TOKENS_PARAGRAPH, TOKENS_DOCUMENT],
            creates_extensions=[LEMMATIZED_SENTENCE, LEMMATIZED_PARAGRAPH,
                                LEMMATIZED_DOCUMENT])

    def apply(self, storage: Dict[Extension, pd.DataFrame],
              queryable: Queryable) -> Dict[Extension, pd.DataFrame]:

        # Document
        df_lemmatized_document = TOKENS_DOCUMENT.load_from(storage=storage)
        df_lemmatized_document["text"] = df_lemmatized_document["text"].apply(
            self.lemmatize_df, nlp=queryable.nlp())

        # Paragraph
        df_lemmatized_paragraph = TOKENS_PARAGRAPH.load_from(storage=storage)
        df_lemmatized_paragraph["text"] = df_lemmatized_paragraph["text"].apply(
            self.lemmatize_df, nlp=queryable.nlp())

        # Sentence
        df_lemmatized_sentence = TOKENS_SENTENCE.load_from(storage=storage)
        df_lemmatized_sentence["text"] = df_lemmatized_sentence["text"].apply(
            self.lemmatize_df, nlp=queryable.nlp())

        return {LEMMATIZED_SENTENCE: df_lemmatized_sentence, LEMMATIZED_PARAGRAPH: df_lemmatized_paragraph, LEMMATIZED_DOCUMENT: df_lemmatized_document}

    def lemmatize_df(self, text: str, nlp) -> list:
        res = []
        # for nlp a list does not work, it expects a string, therefore this strange casting here
        text_nlp = nlp(' '.join(token for token in text))
        for token in text_nlp:
            res.append(token.lemma_ if not token.pos_ == "PRON" else token.text)
        return res
