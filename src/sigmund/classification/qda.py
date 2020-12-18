import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from spacy.tokens import Doc

from pipelinelib.component import Component
from pipelinelib.extension import Extension
from sigmund.features.words import LiwcScores


class QDA_ON_LIWC(Component):
    def __init__(self, training_set):
        super().__init__(name=QDA_ON_LIWC.__name__, required_extensions=[
            LiwcScores.SCORES], creates_extensions=[])
        self.qda = QuadraticDiscriminantAnalysis()
        self.qda.fit(training_set)

    def apply(self, doc: Doc) -> Doc:
        posemo = doc._.liwc_scores.get("Posemo", 0.0)
        negemo = doc._.liwc_scores.get("Negemo", 0.0)
        inhib = doc._.liwc_scores.get("Inhib", 0.0)

        feature_vector = [posemo, negemo, inhib]
        self.qda.predict(feature_vector)

    def _extract_label(self, doc: Doc):
        pass

    def _extract_features(self, doc: Doc):
        pass
