import operator
import unittest

import spacy

from src.pipelinelib.pipeline import Pipeline
from src.sigmund.features.liwc_one_hot import LiwcOneHot as LOHFeature
from src.sigmund.preprocessing.words import Tokenizer
from src.utils.corpus_manager import DialogueCorpusManager


class LiwcOneHot(unittest.TestCase):
    def setUp(self):
        self.nlp = spacy.load("de_core_news_md")
        self.pipeline = Pipeline(model=self.nlp, empty_pipeline=True)

    def test_category_counts(self):
        self.pipeline.add_component(Tokenizer())
        self.pipeline.add_component(LOHFeature())

        sentence = " ".join(['positiv',
                             'gefühle',
                             'ja',
                             'positiv',
                             'gefühle',
                             'können',
                             'bei',
                             'mir',
                             'auslösen',
                             'wenn',
                             'du',
                             'mich',
                             'gut',
                             'behandeln',
                             'fühlen',
                             'ich',
                             'mich',
                             'echt',
                             'tollen',
                             'wenn',
                             'du',
                             'zärtlich',
                             'mit',
                             'mir',
                             'umgehen',
                             'zärtlich',
                             'reden',
                             'zuwendung',
                             'das',
                             'sein',
                             'ja',
                             'zuwendung',
                             'wir',
                             'beide',
                             'wenn',
                             'du',
                             'dich',
                             'mir',
                             'zuwenden',
                             'dann',
                             'fühlen',
                             'ich',
                             'mich',
                             'wohl',
                             'das',
                             'tun',
                             'mir',
                             'gut',
                             'verteilung',
                             'der',
                             'haushaltsaufgaben',
                             'das',
                             'machen',
                             'wir',
                             'sowieso',
                             'ich',
                             'glaub',
                             'da',
                             'brauchen',
                             'wir',
                             'da',
                             'fühlen',
                             'wir',
                             'uns',
                             'beide',
                             'wohl',
                             'da',
                             'muss',
                             'man',
                             'nichts',
                             'sagen',
                             'dazu',
                             'gleich',
                             'vorstellung',
                             'bei',
                             'der',
                             'kindererziehung',
                             'ich',
                             'glaub'])
        expected_liwc = {'Affect': 12,
                         'Posemo': 12,
                         'Posfeel': 4,
                         'Assent': 2,
                         'Occup': 5,
                         'Achieve': 3,
                         'Preps': 3,
                         'Space': 5,
                         'Pronoun': 22,
                         'I': 11,
                         'Self': 16,
                         'Cogmech': 10,
                         'Discrep': 3,
                         'You': 4,
                         'Social': 12,
                         'Othref': 10,
                         'School': 2,
                         'Insight': 4,
                         'Incl': 1,
                         'Article': 5,
                         'Other': 1,
                         'We': 5,
                         'Leisure': 1,
                         'Home': 1,
                         'Past': 1,
                         'Optim': 2,
                         'Certain': 2,
                         'Metaph': 2,
                         'Relig': 2,
                         'Cause': 3,
                         'Excl': 3,
                         'Present': 1,
                         'Negate': 1,
                         'Comm': 2,
                         'Time': 1}
        result_doc = self.pipeline.execute_on(sentence)
        vec_getter = operator.attrgetter(LOHFeature.LIWC_ONE_HOT_VEC.name)
        liwc_encoding = vec_getter(result_doc._)

        self.assertDictEqual(expected_liwc, liwc_encoding)


if __name__ == '__main__':
    unittest.main()
