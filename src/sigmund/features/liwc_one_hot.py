from collections import Counter

from spacy.tokens.doc import Doc

from src.pipelinelib.component import Component
from src.pipelinelib.extension import Extension
from src.sigmund.preprocessing.words import Tokenizer
from src.utils.liwc import Liwc


class LiwcOneHot(Component):
    LIWC_ONE_HOT_VEC = Extension("loh", dict())

    def __init__(self, token_parser_path="./data/German_LIWC2001_Dictionary.dic"):
        super(LiwcOneHot, self).__init__(
            LiwcOneHot.__name__,
            required_extensions=[Tokenizer.TOKENS],
            creates_extensions=[LiwcOneHot.LIWC_ONE_HOT_VEC]
        )
        self.liwc = Liwc(token_parser_path)

    def apply(self, doc: Doc) -> Doc:
        import operator
        getter = operator.attrgetter(Tokenizer.TOKENS.name)
        tokens = getter(doc._)

        doc._.loh = self.liwc.parse(tokens)

        return doc
