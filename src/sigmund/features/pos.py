import operator
import re
import string
from collections import Counter
from itertools import filterfalse

import liwc
import pandas as pd
from spacy.tokens import Doc

from pipelinelib.component import Component
from pipelinelib.extension import Extension
from sigmund.preprocessing.syllables import SyllableExtractor
from sigmund.preprocessing.words import WordExtractor


class PartOfSpeech(Component):
    """
    This component provides features for classification by using Spacy's POS component
    """

    POS = Extension("part_of_speech", dict())

    def __init__(self):
        super().__init__(PartOfSpeech.__name__,
                         required_extensions=[WordExtractor.WORDS],
                         creates_extensions=[PartOfSpeech.SCORES])

    def apply(self, doc: Doc) -> Doc:
        # Load LIWC Dictionary provided by path
        #parse, category_names = liwc.load_token_parser(self._dictionary_path)

        # """ Calculate counts per word-category and divide by number of tokens, append dictionary of liwc-scores to document-
        # object """
        #liwc_counts = Counter(category for token in tokens for category in parse(token))

        # Tokenize Tokens inside of Doc
        tokens = list(self._tokenize_and_lower(doc))
        pos_feature = self._get_pos_as_feature(self._get_pos(tokens))
        doc._.part_of_speech = pos_feature

        return doc

    def _tokenize_and_lower(self, doc: Doc):
        getter = operator.attrgetter(WordExtractor.WORDS.name)
        return [word.lower() for word in getter(doc._)]

    def _get_pos_as_feature(pos_list):
        pos_shares = pd.DataFrame(
            pos_list).apply(
            pd.value_counts).div(
            len(pos_list)).sort_index()

        return dict(zip(pos_shares.index, pos_shares.values[:, 0]))

    def _get_pos(word_list):
        df_as_string = ' '.join(word_list)
        doc = nlp(df_as_string)
        pos_list = []
        [pos.append(word.tag_) for word in doc]

        return pos_list
