import operator

import pandas as pd
from src.pipelinelib.querying import Queryable
from typing import Dict
from src.sigmund.extensions import CLASSIFICATION_NAIVE_BAYES, FEATURE_VECTOR

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from spacy.tokens import Doc

from src.pipelinelib.component import Component
from src.pipelinelib.extension import Extension


class PCA(Component):
    """
    Performs PCA on the feature dataframe 
    """

    def __init__(self):
        super().__init__(
            PCA.__name__,
            required_extensions=[FEATURE_VECTOR],
            creates_extensions=[PCA])

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
