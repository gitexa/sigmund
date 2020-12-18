import operator
import string
from itertools import filterfalse

from nltk.stem.snowball import GermanStemmer
from spacy.tokens import Doc

from pipelinelib.component import Component
from pipelinelib.extension import Extension


class Tokenizer(Component):
    """
    Extracts words from texts. Stores the words and their count under
    doc._.words and doc._.word_count respectively
    """

    TOKENS = Extension("tokens", list())
    TOKEN_COUNT = Extension("token_count", int())

    def __init__(self):
        super().__init__(Tokenizer.__name__, required_extensions=[],
                         creates_extensions=[
                             Tokenizer.TOKENS, Tokenizer.TOKEN_COUNT
        ])

    def apply(self, doc: Doc) -> Doc:
        tokens = map(str, doc)
        doc._.tokens = list(filterfalse(string.punctuation.__contains__, tokens))
        doc._.token_count = len(doc._.tokens)
        return doc


class Stemmer(Component):
    """
    Performs stemming on the tokens
    """

    STEMMED = Extension("stemmed", list())

    def __init__(self, stemmer=GermanStemmer()):
        super(
            Stemmer, self).__init__(
            Stemmer.__name__, required_extensions=[Tokenizer.TOKENS],
            creates_extensions=[Stemmer.STEMMED])
        self._stemmer = stemmer

    def apply(self, doc: Doc) -> Doc:
        doc._.stemmed = self._apply(doc)
        return doc

    def _apply(self, doc: Doc) -> list:
        return [self._stemmer.stem(token) for token in doc._.tokens]


class Lemmatizer(Component):
    """
    Lemmatizes the tokens
    """
    LEMMATIZED = Extension("lemmatized", list())

    def __init__(self):
        super(
            Lemmatizer, self).__init__(
            Lemmatizer.__name__, required_extensions=[Tokenizer.TOKENS],
            creates_extensions=[Lemmatizer.LEMMATIZED])

    def apply(self, doc: Doc) -> Doc:
        doc._.lemmatized = self._apply(doc)
        return doc

    def _apply(self, doc: Doc) -> list:
        return [token.lemma_ if not token.pos_ == "PRON" else token.text for token in doc]


class StemmedAndLemmatized(Component):
    """
    Stem and lemmatize the tokens
    """
    STEM_AND_LEMMATIZE = Extension("stem_and_lemma", list())

    def __init__(self, stemmer=GermanStemmer()):
        super(
            StemmedAndLemmatized, self).__init__(
            StemmedAndLemmatized.__name__, required_extensions=[],
            creates_extensions=[StemmedAndLemmatized.STEM_AND_LEMMATIZE])
        self._stemmer = stemmer

    def apply(self, doc: Doc) -> Doc:
        valid_tokens = filterfalse(
            lambda t: string.punctuation.__contains__(t.text), doc)
        doc._.stem_and_lemma = list(
            self._stemmer.stem(token.lemma_) for token in valid_tokens
        )
        return doc
