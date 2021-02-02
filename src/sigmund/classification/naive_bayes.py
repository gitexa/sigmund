from sklearn.naive_bayes import MultinomialNB
from spacy.tokens import Doc

from pipelinelib.component import Component
from pipelinelib.extension import Extension


class NaiveBayes(Component):
    BAYES = Extension("gaussian_nb", int)

    def __init__(self, x_train, y_train):
        super().__init__(name=NaiveBayes.__name__, required_extensions=[
            # TODO: https://scikit-learn.org/stable/modules/naive_bayes.html
            # TODO: what do we work on here? TFIDF and word vector counts
            # TODO: are apparently known to work
        ], creates_extensions=[NaiveBayes.BAYES])
        self.naive_bayes = MultinomialNB()
        self.naive_bayes.fit(x_train, y_train)

    def apply(self, doc: Doc) -> Doc:
        # TODO: Load feature vector
        feature_vector = None
        prediction = self.naive_bayes.predict(feature_vector)
        doc._.gaussian_nb = prediction[0]

        return doc
