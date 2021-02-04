import string
from itertools import filterfalse

import pyphen
from spacy.tokens import Doc

from src.pipelinelib.component import Component
from src.pipelinelib.extension import Extension


class Cleaner(Component):
    """
    Lowercases, removes stop words and symbols,
    and lemmatises pronouns
    """
    CLEANED = Extension("cleaned", list())

    def __init__(self):
        super().__init__(
            name=Cleaner.__name__, required_extensions=[],
            creates_extensions=[Cleaner.CLEANED])

    def apply(self, doc: Doc) -> Doc:
        # TODO: Consider using custom stop words
        candidates = list(filterfalse(lambda t: t.is_stop, doc))

        candidates = list(filter(lambda t: t.is_alpha or t.text.isdigit(), candidates))
        candidates = list(filterfalse(lambda t:
                                      t.text in string.whitespace, candidates))

        lemmas = list(map(lambda t: t.lemma_ if not t.pos_ ==
                          "PRON" else t.text, candidates))

        doc._.cleaned = list(lemmas)
        return doc
