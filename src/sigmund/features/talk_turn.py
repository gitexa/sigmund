import string

import pyphen
from spacy.tokens import Doc

from pipelinelib.component import Component
from pipelinelib.extension import Extension


class TalkTurnExtractor(Component):
    """
    Extracts Talk turn from text and stores these under doc._.talkturn
    """
    talkturn = Extension(name="talkturn", default_type=list())

    def __init__(self):
        super().__init__(name=TalkTurnExtractor.__name__, required_extensions=list(),
                         creates_extensions=[TalkTurnExtractor.talkturn])
        self.dic = pyphen.Pyphen(lang='de')

    def apply(self, doc: Doc) -> Doc:

        return doc
