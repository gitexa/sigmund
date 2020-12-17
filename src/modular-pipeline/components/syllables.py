import pyphen
import string
from blocks.component import Component
from blocks.extension import Extension
from spacy.tokens import Doc, Token


class SyllableExtractor(Component):
    SYLLABLES = Extension(name="syllables", default_type=list())

    def __init__(self):
        super().__init__(name=SyllableExtractor.__name__, required_extensions=list(), creates_extensions=[
            SyllableExtractor.SYLLABLES
        ])
        self.dic = pyphen.Pyphen(lang='de')

    def method(self, doc: Doc):
        doc.set_extension(self.SYLLABLES.name, default=self.SYLLABLES.default_type)
        for token in doc:
            if not str(token) in string.punctuation:
                doc._.syllables.extend(self.dic.inserted(str(token)).split("-"))

        return doc
