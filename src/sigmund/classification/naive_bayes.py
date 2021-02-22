import operator
from typing import Dict

import numpy as np
import pandas as pd
from IPython.core.display import display
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict, train_test_split
from sklearn.naive_bayes import MultinomialNB
from spacy.tokens import Doc

from src.pipelinelib.component import Component
from src.pipelinelib.extension import Extension
from src.pipelinelib.querying import Queryable
from src.pipelinelib.text_body import TextBody
from src.sigmund.extensions import CLASSIFICATION_NAIVE_BAYES, FEATURE_VECTOR


class NaiveBayes(Component):
    """
    Performs naive bayes on a feature vector and prints results
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
            [couple_id_test, gt_test, label_test, df_predicted_test], axis=1)

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

        return {CLASSIFICATION_NAIVE_BAYES: df_prediction_summary}
