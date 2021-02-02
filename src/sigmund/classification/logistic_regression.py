from sklearn.linear_model import LogisticRegression
from spacy.tokens import Doc

from pipelinelib.component import Component
from pipelinelib.extension import Extension


class LogRegr(Component):
    LOG_REGR = Extension("log_regr")

    def __init__(self, x_train, y_train):
        super().__init__(name=LogRegr.__name__, required_extensions=[
            # TODO: what do we work on here
        ], creates_extensions=[LogRegr.LOG_REGR])
        self.clf = LogisticRegression()
        self.clf.fit(x_train, y_train)

    def apply(self, doc: Doc) -> Doc:
        # TODO: Load feature vector
        feature_vector = None
        prediction = self.clf.predict(feature_vector)
        doc._.log_regr = prediction[0]

        return doc
