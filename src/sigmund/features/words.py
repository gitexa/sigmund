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
    This component provides features for classification by using the LIWC-Approach. LIWC (spoken "LUKE") stands for "Linguistic Inquiry of Word Counts"
    LIWC is based on a dictionary that assigns Words to specific categories, for example "Posemo" (Positive Emotions), Negemo (Negative Emotions), Pronouns et cetera
    A text-segment is analyzed by counting the occurrence of words per individual categories
    LIWC provides a Desktop-Application that calculates the word counts. However there also exists a Python-Library that takes in .dic files (Dictionary files) to calculate
    LIWC-Scores. That's the purpose of this component.
    """

    SCORES = Extension("liwc_scores", dict())

    def __init__(self, dictionary_path: str):
        super().__init__(LiwcScores.__name__, required_extensions=[
            WordExtractor.WORDS],
                         creates_extensions=[LiwcScores.SCORES])
        self._dictionary_path = dictionary_path

    def apply(self, doc: Doc) -> Doc:
        # Tokenize Tokens inside of Doc
        tokens = list(self._tokenize_and_lower(doc))
        # Load LIWC Dictionary provided by path
        parse, category_names = liwc.load_token_parser(self._dictionary_path)

        """ Calculate counts per word-category and divide by number of tokens, append dictionary of liwc-scores to document-
         object """
        liwc_counts = Counter(category for token in tokens for category in parse(token))
        doc._.liwc_scores = {
            token: score / len(tokens) for token, score in liwc_counts.items()
        }
        return doc

    def _tokenize_and_lower(self, doc: Doc):
        getter = operator.attrgetter(WordExtractor.WORDS.name)
        return [word.lower() for word in getter(doc._)]
