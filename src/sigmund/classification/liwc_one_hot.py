from collections import Counter

import liwc
from spacy.tokens.doc import Doc

from src.pipelinelib.component import Component
from src.pipelinelib.extension import Extension


class LiwcOneHot(Component):
    LIWC_ONE_HOT_VEC = Extension("loh", dict())

    def __init__(self, token_parser_path="./data/German_LIWC2001_Dictionary.dic"):
        super(LiwcOneHot, self).__init__(
            LiwcOneHot.__name__,
            required_extensions=[],
            creates_extensions=[LiwcOneHot.LIWC_ONE_HOT_VEC]
        )
        self._parse, self._cat_names = liwc.load_token_parser(token_parser_path)

    def apply(self, doc: Doc) -> Doc:
        liwc_scores = dict.fromkeys(self._cat_names)

        without_puncs = filter(lambda token: not token.is_punct, doc)
        categories = dict(Counter(
            category for token in without_puncs
            for category in self._parse(token.text.lower())
        ))

        liwc_counts = {**liwc_scores, **categories}
        liwc_counts = {key: item or 0 for key, item in liwc_counts.items()}

        doc._.loh = liwc_counts
        return doc
