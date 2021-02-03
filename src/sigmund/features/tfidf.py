import operator

from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.tokens import Doc

from pipelinelib.component import Component
from pipelinelib.extension import Extension
from sigmund.features.words import Tokenizer


class TfIdf(Component):
    TFIDF = Extension("tfidf", 0)

    def __init__(self):
        super().__init__(name=TfIdf.__name__, required_extensions=[
            Tokenizer.TOKENS], creates_extensions=[TfIdf.TFIDF])
        self.vectorizer = TfidfVectorizer()

    def apply(self, doc: Doc) -> Doc:
        tokens = operator.attrgetter(Tokenizer.TOKENS.name)(doc._)
        doc._.tfidf = self.vectorizer.fit_transform(tokens)

        return doc
