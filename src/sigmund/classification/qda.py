import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from spacy.tokens import Doc

from pipelinelib.component import Component
from pipelinelib.extension import Extension
from sigmund.features.words import LiwcScores
import numpy as np


class QDA_ON_LIWC(Component):
    QDA_ON_LIWC = Extension("QDA_ON_LIWC", 0)

    def __init__(self, X_train, y_train):
        super().__init__(name=QDA_ON_LIWC.__name__, required_extensions=[
            LiwcScores.SCORES], creates_extensions=[QDA_ON_LIWC.QDA_ON_LIWC])
        self.qda = LinearDiscriminantAnalysis()
        self.qda.fit(X_train, y_train)

    def apply(self, doc: Doc) -> Doc:
        posemo = doc._.liwc_scores.get("Posemo", 0.0)
        negemo = doc._.liwc_scores.get("Negemo", 0.0)
        inhib = doc._.liwc_scores.get("Inhib", 0.0)

        feature_vector = np.array([posemo, negemo, inhib]).reshape(1, -1)
        qda_prediction = self.qda.predict(feature_vector)
        doc._.QDA_ON_LIWC = qda_prediction[0]

        return doc
