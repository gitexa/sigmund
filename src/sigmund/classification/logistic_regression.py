import operator

from sklearn.linear_model import LogisticRegression
from spacy.tokens import Doc

from pipelinelib.component import Component
from pipelinelib.extension import Extension
from sigmund.features.tfidf import TfIdf


class LogRegrOnTfIdf(Component):
    """
    This component performs Logistic Regression on TfIdf. 
    Logistic regression is used to classify the TfIdf data and to show the relationship between the variables by 
    modeling the distribution of discrete variables.
    """
    
    LOG_REGR = Extension("log_regr")

    def __init__(self, x_train, y_train):
        super().__init__(name=LogRegrOnTfIdf.__name__, required_extensions=[
            TfIdf.TFIDF], creates_extensions=[LogRegrOnTfIdf.LOG_REGR])
        self.clf = LogisticRegression()
        self.clf.fit(x_train, y_train)

    def apply(self, doc: Doc) -> Doc:
        feature_vector = operator.attrgetter(TfIdf.TFIDF.name)(doc._)
        prediction = self.naive_bayes.predict(feature_vector)

        doc._.log_regr = prediction[0]
        return doc
