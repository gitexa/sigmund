import string
from itertools import filterfalse

import pandas as pd
import pyphen
import spacy
from spacy.tokens import Doc
from utils.feature_annotator import reading_ease_german, syllable_counter

from pipelinelib.component import Component
from pipelinelib.extension import Extension

nlp = spacy.load("de_core_news_md")


class FleschExtractor(Component):
    """
    Calculates the Flesch-Reading-Ease from text and stores these under doc._.flesch_sentence
    """
    FLESCH_SENTENCE = Extension(name="flesch_sentence", default_type=list())

    def __init__(self):
        super().__init__(name=FleschExtractor.__name__, required_extensions=[],
                         creates_extensions=[FleschExtractor.FLESCH_SENTENCE])
        self.dic = pyphen.Pyphen(lang='de')

    def apply(self, doc: Doc) -> Doc:  # Sentence
        # print(doc.text)
        doc._.flesch_sentence = reading_ease_german(doc.text)
        return doc


'''
    def apply(self, doc: Doc) -> Doc:  # Paragraph

        doc._.flesch_sentence = reading_ease_german(doc)
        return doc

    def apply(self, doc: Doc) -> Doc:  # Dialog

        doc._.flesch_sentence = reading_ease_german(doc)
        return doc
'''
