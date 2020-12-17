import unittest

import spacy

from pipelinelib import Component


class Pipeline(unittest.TestCase):
    NLP = spacy.load("de_core_news_sm", disable=["ner", "parser"])

    def setUp(self):
        self.pipeline = Pipeline(model=Pipeline.NLP)

    def trigger_overwrite_failure(self):
        """
        Two components write to the same component
        """
        c1 = Component()

    def trigger_missing_failure(self):
        pass

    def trigger_nothing(self):
        pass


if __name__ == '__main__':
    unittest.main()
