import operator
from functools import reduce
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from spacy.tokens import Doc

from src.pipelinelib.component import Component
from src.pipelinelib.extension import Extension
from src.pipelinelib.querying import Parser, Queryable
from src.pipelinelib.text_body import TextBody
from src.sigmund.extensions import (CLASSIFICATION_LINEAR_DISCRIMINANT_ANALYSIS,
                                    FEATURE_VECTOR)


class LinearDiscriminantAnalysisClassifier(Component):
    """
    Performs linear discriminant analysis on a feature vector 
    """

    def __init__(self, inputs: List[Extension] = None,
                 output: Extension = None):
        self.inputs = inputs or [FEATURE_VECTOR]
        self.output = output or CLASSIFICATION_LINEAR_DISCRIMINANT_ANALYSIS

        super().__init__(
            LinearDiscriminantAnalysisClassifier.__name__,
            required_extensions=self.inputs,
            creates_extensions=[self.output])

    def apply(self, storage: Dict[Extension, pd.DataFrame],
              queryable: Queryable) -> Dict[Extension, pd.DataFrame]:

        if not len(self.inputs):
            return dict()

        # get features
        elif len(self.inputs) == 1:
            df_feature_vector = self.inputs[0].load_from(storage=storage)

        else:
            loaded = map(lambda e: e.load_from(storage=storage), self.inputs)
            df_feature_vector = reduce(lambda left, right: pd.merge(
                left, right, on=Parser.COUPLE_ID, how="inner"), loaded)

        if "is_depressed_group" not in df_feature_vector.columns:
            metadata = queryable.execute(level=TextBody.DOCUMENT)
            df_feature_vector = pd.merge(
                metadata[['couple_id', 'is_depressed_group']],
                df_feature_vector, on='couple_id', how='inner')

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

        return {self.output: df_prediction_summary}

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
