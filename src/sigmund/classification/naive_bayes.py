import operator
from functools import reduce
from typing import Dict, List

import numpy as np
import pandas as pd
from IPython.core.display import display
from sklearn import metrics
from sklearn.model_selection import (StratifiedKFold, cross_val_predict,
                                     cross_val_score, train_test_split)
from sklearn.naive_bayes import MultinomialNB
from spacy.tokens import Doc

from src.pipelinelib.component import Component
from src.pipelinelib.extension import Extension
from src.pipelinelib.querying import Parser, Queryable
from src.pipelinelib.text_body import TextBody
from src.sigmund.extensions import CLASSIFICATION_NAIVE_BAYES, FEATURE_VECTOR


class NaiveBayes(Component):
    """
    Performs naive bayes on a feature vector and prints results
    """

    def __init__(self,
                 inputs: List[Extension] = None,
                 output: Extension = None,
                 voting: bool = False):

        self.inputs = inputs or [FEATURE_VECTOR]
        self.output = output or CLASSIFICATION_NAIVE_BAYES
        self.voting = voting

        super().__init__(
            NaiveBayes.__name__,
            required_extensions=self.inputs,
            creates_extensions=[self.output])

    def apply(self, storage: Dict[Extension, pd.DataFrame],
              queryable: Queryable) -> Dict[Extension, pd.DataFrame]:

        # Read user input:
        # 1) from which features to construct feature vector
        # 2) key to store the results
        # if voting==True, feature vector consists of previous classifier outputs
        if not self.voting:
            if not len(self.inputs):
                return dict()
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
        else:
            if not len(self.inputs):
                raise Exception(
                    'When voting, keys for the classifier results must be used.')
            elif len(self.inputs) == 1:
                df_feature_vector = self.inputs[0].load_from(storage=storage)
            else:
                loaded = map(lambda e: e.load_from(storage=storage), self.inputs)
                df_feature_vector = reduce(
                    lambda left, right: pd.merge(
                        left[['couple_id', 'predicted']],
                        right[['couple_id', 'predicted']],
                        on=Parser.COUPLE_ID, how="inner"),
                    loaded)
                metadata = queryable.execute(level=TextBody.DOCUMENT)
                df_feature_vector = pd.merge(
                    metadata[['couple_id', 'is_depressed_group']],
                    df_feature_vector, on='couple_id', how='inner')

        display(df_feature_vector)

        couple_id = df_feature_vector["couple_id"]
        labels = df_feature_vector["is_depressed_group"].astype(int)
        features = df_feature_vector[df_feature_vector.columns.difference(
            ["couple_id", "is_depressed_group"], sort=False)]

        # Using "normal" validation
        features_train, features_test, label_train, label_test, indices_train, indices_test = train_test_split(
            features, labels, features.index.values, test_size=0.50, random_state=42)

        # fit classifier
        classifier = MultinomialNB()
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
            [couple_id_test, label_test, df_predicted_test], axis=1)

        # Using cross validation
        gt = df_feature_vector['is_depressed_group']
        cv = StratifiedKFold(n_splits=5, random_state=42)

        predictions_test_cv = cross_val_predict(classifier, features, labels, cv=cv)
        df_predicted_test = pd.DataFrame(
            data=predictions_test_cv, columns=['predicted'],
            index=labels.index.copy())

        scores = cross_val_score(classifier, features, labels, cv=cv)
        df_prediction_summary_cv = pd.concat(
            [couple_id, gt, labels, df_predicted_test], axis=1)

        # Print results
        display('Predictions on the test set')
        display(df_prediction_summary)
        display('Cross-valiation')
        display(df_prediction_summary_cv)
        display(f'Accuracy on test set: {accuracy}')
        display(f'Accuracy with cross-valiation: {scores} | mean = {np.mean(scores)}')

        return {self.output: df_prediction_summary_cv}
