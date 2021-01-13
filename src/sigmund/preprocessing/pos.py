from collections import defaultdict

from spacy.tokens.doc import Doc

from src.pipelinelib.component import Component
from src.pipelinelib.extension import Extension


class PartOfSpeechTags(Component):
    POS_COUNT = Extension("pos_counts", dict())
    NOUN_COUNT = Extension("noun_count", int)
    AUXN_COUNT = Extension("auxn_count", int)
    PRON_COUNT = Extension("pron_count", int)
    VERB_COUNT = Extension("verb_count", int)
    PROPN_COUNT = Extension("propn_count", int)

    def __init__(self):
        super().__init__(PartOfSpeechTags.__name__,
                         required_extensions=[],
                         creates_extensions=[PartOfSpeechTags.POS_COUNT, PartOfSpeechTags.VERB_COUNT,
                                             PartOfSpeechTags.NOUN_COUNT, PartOfSpeechTags.AUXN_COUNT,
                                             PartOfSpeechTags.PRON_COUNT])

    def apply(self, doc: Doc) -> Doc:
        doc._.pos_count = defaultdict(int)
        for token in doc:
            doc._.pos_count[token.pos_] += 1

        doc._.verb_count = doc._.pos_count["VERB"] + doc._.pos_count["AUX"]
        doc._.noun_count = doc._.pos_count["NOUN"] + doc._.pos_count["PROPN"] + doc._.pos_count["PRON"]
        doc._.auxn_count = doc._.pos_count["AUX"]
        doc._.pron_count = doc._.pos_count["PRON"]
        doc._.propn_count = doc._.pos_count["PROPN"]

        return doc