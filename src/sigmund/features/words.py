import operator
import re
import string
from collections import Counter
from itertools import filterfalse

import liwc
from spacy.tokens import Doc

from pipelinelib.component import Component
from pipelinelib.extension import Extension
from sigmund.preprocessing.syllables import SyllableExtractor
from sigmund.preprocessing.words import WordExtractor


class LiwcScores(Component):
    """
    TODO: @greenhippo
    """

    SCORES = Extension("liwc_scores", dict())

    def __init__(self, dictionary_path: str):
        super().__init__(LiwcScores.__name__, required_extensions=[
            WordExtractor.WORDS],
            creates_extensions=[LiwcScores.SCORES])
        self._dictionary_path = dictionary_path

    def apply(self, doc: Doc) -> Doc:
        tokens = list(self._tokenize_and_lower(doc))
        parse, category_names = liwc.load_token_parser(self._dictionary_path)

        liwc_counts = Counter(category for token in tokens for category in parse(token))
        doc._.liwc_scores = {
            token: score / len(tokens) for token, score in liwc_counts.items()
        }
        return doc

    def _tokenize_and_lower(self, doc: Doc):
        getter = operator.attrgetter(WordExtractor.WORDS.name)
        return [word.lower() for word in getter(doc._)]
