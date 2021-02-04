import operator

import pandas as pd
from spacy.tokens import Doc

from pipelinelib.component import Component
from pipelinelib.extension import Extension
from sigmund.preprocessing.words import WordExtractor
from src.sigmund.preprocessing.pos import PartOfSpeechTags


class PartOfSpeech(Component):
    """
    This component provides features for classification by using Spacy's POS component
    """

    POS = Extension("part_of_speech", dict())

    def __init__(self):
        super().__init__(PartOfSpeech.__name__,
                         required_extensions=[WordExtractor.WORDS],
                         creates_extensions=[PartOfSpeech.POS])

    def apply(self, doc: Doc) -> Doc:
        # Load LIWC Dictionary provided by path
        # parse, category_names = liwc.load_token_parser(self._dictionary_path)

        # """ Calculate counts per word-category and divide by number of tokens, append dictionary of liwc-scores to document-
        # object """
        # liwc_counts = Counter(category for token in tokens for category in parse(token))

        # Tokenize Tokens inside of Doc
        tokens = list(self._tokenize_and_lower(doc))
        pos_feature = self._get_pos_as_feature(self._get_pos(tokens))
        doc._.part_of_speech = pos_feature

        return doc

    def _tokenize_and_lower(self, doc: Doc):
        getter = operator.attrgetter(WordExtractor.WORDS.name)
        return [word.lower() for word in getter(doc._)]

    def _get_pos_as_feature(self, pos_list):
        pos_shares = pd.DataFrame(
            pos_list).apply(
            pd.value_counts).div(
            len(pos_list)).sort_index()

        return dict(zip(pos_shares.index, pos_shares.values[:, 0]))

    def _get_pos(self, word_list):
        df_as_string = ' '.join(word_list)
        doc = nlp(df_as_string)
        pos_list = []
        [pos.append(word.tag_) for word in doc]

        return pos_list


class ReadinessToAction(Component):
    R2D = Extension("r2d", float)

    def __init__(self):
        super(ReadinessToAction, self).__init__(
            ReadinessToAction.__name__,
            required_extensions=[PartOfSpeechTags.VERB_COUNT, PartOfSpeechTags.NOUN_COUNT],
            creates_extensions=[ReadinessToAction.R2D]
        )

    def apply(self, doc: Doc) -> Doc:
        verb_count = operator.attrgetter(PartOfSpeechTags.VERB_COUNT.name)(doc._)
        noun_count = operator.attrgetter(PartOfSpeechTags.NOUN_COUNT.name)(doc._)

        doc._.r2d = verb_count / noun_count
        return doc


class PronominalisationIndex(Component):
    PRON_INDEX = Extension("pron_index", float)

    def __init__(self):
        super(PronominalisationIndex, self).__init__(
            PronominalisationIndex.__name__,
            required_extensions=[PartOfSpeechTags.PRON_COUNT, PartOfSpeechTags.NOUN_COUNT,
                                 PartOfSpeechTags.PROPN_COUNT],
            creates_extensions=[PronominalisationIndex.PRON_INDEX]
        )

    def apply(self, doc: Doc) -> Doc:
        pron_count = operator.attrgetter(PartOfSpeechTags.PRON_COUNT.name)(doc._)
        noun_count = operator.attrgetter(PartOfSpeechTags.NOUN_COUNT.name)(doc._)
        propn_count = operator.attrgetter(PartOfSpeechTags.PROPN_COUNT.name)(doc._)

        doc._.pron_index = pron_count / (pron_count + noun_count + propn_count)
        return doc
