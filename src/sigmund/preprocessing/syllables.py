import string

import pyphen
from spacy.tokens import Doc

from pipelinelib.component import Component
from pipelinelib.extension import Extension

import string

class SyllableExtractor(Component):
    """
    Extracts syllables from text and stores these under doc._.syllables
    """
    SYLLABLES = Extension(name="syllables", default_type=list())

    def __init__(self):
        super().__init__(name=SyllableExtractor.__name__, required_extensions=list(),
                         creates_extensions=[SyllableExtractor.SYLLABLES])
        self.dic = pyphen.Pyphen(lang='de')

    def apply(self, doc: Doc) -> Doc:
        for token in doc:
            if not str(token) in string.punctuation:
                doc._.syllables.extend(self.dic.inserted(str(token)).split("-"))
        return doc
