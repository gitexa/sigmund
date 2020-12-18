import string
from itertools import filterfalse

from spacy.tokens import Doc

from pipelinelib.component import Component
from pipelinelib.extension import Extension


class WordExtractor(Component):
    """
    Extracts words from texts. Stores the words and their count under
    doc._.words and doc._.word_count respectively
    """

    WORDS = Extension("words", list())
    WORD_COUNT = Extension("word_count", int())

    def __init__(self):
        super().__init__(WordExtractor.__name__, required_extensions=[],
                         creates_extensions=[
            WordExtractor.WORDS, WordExtractor.WORD_COUNT
        ])

    def apply(self, doc: Doc) -> Doc:
        tokens = map(str, doc)
        doc._.words = list(filterfalse(string.punctuation.__contains__, tokens))
        doc._.word_count = len(doc._.words)
        return doc
