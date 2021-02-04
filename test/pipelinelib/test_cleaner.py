import unittest

import spacy

from src.pipelinelib.pipeline import Pipeline
from src.sigmund.preprocessing.cleaner import Cleaner as PCleaner
from src.utils.corpus_manager import DialogueCorpusManager


class Cleaner(unittest.TestCase):
    NLP = spacy.load("de_core_news_sm", disable=["ner", "parser"])

    def setUp(self):
        cleaner = PCleaner()
        self.pipeline = Pipeline(model=Cleaner.NLP, empty_pipeline=True) \
            .add_component(cleaner)

        self.corpus_manager = DialogueCorpusManager(
            corpus_file="all_preprocessed.csv", nlp=Cleaner.NLP
        )

    def test_observe(self):
        df = self.corpus_manager.get_couple_paragraphs(couple_id=105)
        paragraphs = list(map(lambda tup: tup[1], zip(range(3), df["lemmatized"])))

        for paragraph in paragraphs:
            doc = self.pipeline.execute(paragraph)
            print(paragraph)
            print(doc._.cleaned)


if __name__ == '__main__':
    unittest.main()
