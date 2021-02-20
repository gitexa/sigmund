import operator

from sklearn.naive_bayes import MultinomialNB
from spacy.tokens import Doc

from pipelinelib.component import Component
from pipelinelib.extension import Extension
from sigmund.features.tfidf import TfIdf


class NaiveBayes(Component):
    """
    Performs naive bayes on a feature dataframe 
    """

    def __init__(self):
        super().__init__(
            NaiveBayes.__name__,
            required_extensions=[FEATURE_VECTOR],
            creates_extensions=[CLASSIFICATION_NAIVE_BAYES])

    def apply(self, storage: Dict[Extension, pd.DataFrame],
              queryable: Queryable) -> Dict[Extension, pd.DataFrame]:

        # Document
        df_feature_vector = FEATURE_VECTOR.load_from(storage=storage)

        return {CLASSIFICATION_NAIVE_BAYES: df_stemmed_sentence}


class NaiveBayesOnTfIdf(Component):
    BAYES = Extension("tfidf_gaussian_nb", int)

    def __init__(self, x_train, y_train):
        super().__init__(name=NaiveBayesOnTfIdf.__name__, required_extensions=[
            TfIdf.TFIDF], creates_extensions=[NaiveBayesOnTfIdf.BAYES])
        self.naive_bayes = MultinomialNB()
        self.naive_bayes.fit(x_train, y_train)

    def apply(self, doc: Doc) -> Doc:
        feature_vector = operator.attrgetter(TfIdf.TFIDF.name)(doc._)
        prediction = self.naive_bayes.predict(feature_vector)

        doc._.tfidf_gaussian_nb = prediction[0]
        return doc
