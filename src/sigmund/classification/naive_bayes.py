import operator
from typing import Dict

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from spacy.tokens import Doc

from src.pipelinelib.component import Component
from src.pipelinelib.extension import Extension
from src.pipelinelib.querying import Queryable
from src.sigmund.extensions import CLASSIFICATION_NAIVE_BAYES, FEATURE_VECTOR


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

        # get features
        df_feature_vector = FEATURE_VECTOR.load_from(storage=storage)

        couple_id = df_feature_vector.iloc[:, 0]
        labels = df_feature_vector.iloc[:, 1].astype(int)
        features = df_feature_vector.iloc[:, 2:]

        features_train, features_test, label_train, label_test, indices_train, indices_test = train_test_split(
            features, labels, features.index.values, test_size=0.20, random_state=42)

        # fit classifier
        classifier = MultinomialNB()
        classifier.fit(features_train, label_train)

        # predict
        predicted = classifier.predict(features_test)
        display(label_test)
        display(predicted)

        # evaluate classifier
        accuracy = ((predicted == label_test).sum())/len(label_test)
        display(accuracy)

        return {CLASSIFICATION_NAIVE_BAYES: predicted}


'''
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
'''
