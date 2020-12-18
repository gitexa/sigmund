import string

import pyphen
from spacy.tokens import Doc

from pipelinelib.component import Component
from pipelinelib.extension import Extension


class AgreementScoreExtractor(Component):
    """
    Extracts the Agreement-Score turn from text and stores these under doc._.agreementscore
    """
    agreementscore = Extension(name="agreementscore", default_type=list())

    def __init__(self):
        super().__init__(name=TalkTurnExtractor.__name__, required_extensions=list(),
                         creates_extensions=[AgreementScoreExtractor.agreementscore])
        self.dic = pyphen.Pyphen(lang='de')

    def apply(self, doc: Doc) -> Doc:

        return doc
