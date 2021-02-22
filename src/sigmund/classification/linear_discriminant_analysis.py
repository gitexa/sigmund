import operator
from typing import Dict

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
    Performs linear discriminant analysis on a feature dataframe 
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
        classifier = LinearDiscriminantAnalysis()
        classifier.fit(features_train, label_train)

        # predict
        predicted = classifier.predict(features_test)
        display(label_test)
        display(predicted)

        # evaluate classifier
        accuracy = ((predicted == label_test).sum()) / len(label_test)
        display(accuracy)

        return {CLASSIFICATION_LINEAR_DISCRIMINANT_ANALYSIS: predicted}
