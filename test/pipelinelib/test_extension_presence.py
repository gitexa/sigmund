import unittest

import spacy

from src.pipelinelib.component import Component
from src.pipelinelib.extension import Extension
from src.pipelinelib.pipeline import Pipeline as PAPI
from src.pipelinelib.querying import Parser, Queryable

CLASHING_EXTENSION = Extension("clash-with-me")


class Pipeline(unittest.TestCase):
    NLP = spacy.load("de_core_news_sm", disable=["ner", "parser"])

    def setUp(self):
        self.pipeline = PAPI(
            queryable=Queryable.from_parser(
                parser=Parser(
                    nlp=Pipeline.NLP,
                    metadata_path="./data/transcripts/Kopie von Transkriptionspaare_Daten.xls")))

    def test_trigger_overwrite_failure(self):
        """
        Two components write to the same component
        """
        self.pipeline.add_component(C1())

        self.assertRaises(Exception, self.pipeline.add_component, C2())

    def test_trigger_missing_failure(self):
        """
        Component references non-existent extension
        """
        self.assertRaises(Exception, self.pipeline.add_component, D1())

    def test_trigger_nothing(self):
        """
        Successful case: C1 add extensions, D1 references it
        """
        try:
            self.pipeline.add_component(C1())
            self.pipeline.add_component(D1())
        except Exception as e:
            self.fail(e)


class C1(Component):
    def __init__(self):
        super().__init__(C1.__name__, required_extensions=[],
                         creates_extensions=[CLASHING_EXTENSION])

    def apply():
        pass


class C2(Component):
    def __init__(self):
        super().__init__(C2.__name__, required_extensions=[],
                         creates_extensions=[CLASHING_EXTENSION])

    def apply(self):
        pass


class D1(Component):
    def __init__(self):
        super().__init__(D1.__name__, required_extensions=[CLASHING_EXTENSION],
                         creates_extensions=[])

    def apply(self):
        pass


if __name__ == '__main__':
    unittest.main()
