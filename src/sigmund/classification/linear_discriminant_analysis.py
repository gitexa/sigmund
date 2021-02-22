import operator
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from spacy.tokens import Doc

from src.pipelinelib.component import Component
from src.pipelinelib.extension import Extension
from src.pipelinelib.querying import Queryable
from src.pipelinelib.text_body import TextBody
from src.sigmund.extensions import *


class LinearDiscriminantAnalysis(Component):
    """
    Performs linear discriminant analysis on a feature vector 
    """

    def __init__(self):
        super().__init__(
            LinearDiscriminantAnalysis.__name__,
            required_extensions=[FEATURE_VECTOR],
            creates_extensions=[CLASSIFICATION_LINEAR_DISCRIMINANT_ANALYSIS])

    def apply(self, storage: Dict[Extension, pd.DataFrame],
              queryable: Queryable) -> Dict[Extension, pd.DataFrame]:

        # get features
        df_feature_vector = FEATURE_VECTOR.load_from(storage=storage)

        metadata = queryable.execute(level=TextBody.DOCUMENT)
        df_feature_vector = pd.merge(
            metadata[['couple_id', 'is_depressed_group']],
            df_feature_vector, on='couple_id', how='inner')

        display(df_feature_vector)

        couple_id = df_feature_vector["couple_id"]
        labels = df_feature_vector["is_depressed_group"].astype(int)
        features = df_feature_vector[df_feature_vector.columns.difference(
            ["couple_id", "is_depressed_group"], sort=False)]

        features_train, features_test, label_train, label_test, indices_train, indices_test = train_test_split(
            features, labels, features.index.values, test_size=0.20, random_state=42)

        # fit classifier
        classifier = LinearDiscriminantAnalysis(n_components=2)
        classifier.fit(features_train, label_train)

        # predict
        predicted = classifier.predict(features_test)
        display(label_test)
        display(predicted)

        # evaluate classifier
        accuracy = ((predicted == label_test).sum()) / len(label_test)
        display(accuracy)

        # Using cross validation
        cv = StratifiedKFold(n_splits=5, random_state=42)
        scores = cross_val_score(classifier, features, labels, cv=cv)
        display(np.mean(scores))

        # visualize with reduced dimensionality
        self.visualize(plt, classifier, features, labels)

        return {CLASSIFICATION_LINEAR_DISCRIMINANT_ANALYSIS: predicted}

    def visualize(self, plt, classifier, all_features, labels):

        # dimensionality reduction
        embedded_features = classifier.transform(all_features)

        # plot the projected points
        plt.scatter(
            embedded_features[:, 0],
            embedded_features[:, 1],
            c=labels, s=30, cmap='Set1')
