import operator
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from spacy.tokens import Doc

from src.pipelinelib.component import Component
from src.pipelinelib.extension import Extension
from src.pipelinelib.querying import Queryable
from src.pipelinelib.text_body import TextBody
from src.sigmund.extensions import *


class LinearDiscriminantAnalysisClassifier(Component):
    """
    Performs linear discriminant analysis on a feature vector 
    """

    def __init__(self):
        super().__init__(
            LinearDiscriminantAnalysisClassifier.__name__,
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
        classifier = LinearDiscriminantAnalysis()
        classifier.fit(features_train, label_train)

        # predict
        predicted_test = classifier.predict(features_test)
        df_predicted_test = pd.DataFrame(
            data=predicted_test, columns=['predicted'],
            index=label_test.index.copy())


        # evaluate classifier
        accuracy = ((predicted_test == label_test).sum()) / len(label_test)

        # aggregate results and build dataframe
        couple_id_test = df_feature_vector.iloc[indices_test, :]['couple_id']
        gt_test = df_feature_vector.iloc[indices_test, :]['is_depressed_group']
        df_prediction_summary = pd.concat(
            [couple_id_test, gt_test, label_test, df_predicted_test], axis=1)

        # Using cross validation
        cv = StratifiedKFold(n_splits=5, random_state=42)
        scores = cross_val_score(classifier, features, labels, cv=cv)

        # Print results
        display(df_prediction_summary)
        display(f'Accuracy on test set: {accuracy}')
        display(f'Accuracy with cross-valiation: {scores} | mean = {np.mean(scores)}')


        # visualize with reduced dimensionality
        # self.visualize(plt, classifier, features, labels)

        return {CLASSIFICATION_LINEAR_DISCRIMINANT_ANALYSIS: df_prediction_summary}

    def visualize(self, plt, classifier, all_features, labels):

        # dimensionality reduction
        embedded_features = classifier.transform(all_features)

        display(all_features)
        display(labels)
        display(embedded_features)

        # plot the projected points
        plt.plot(
            embedded_features[:, 0],
            embedded_features[:, 1],
            c=labels, s=30, cmap='Set1')
