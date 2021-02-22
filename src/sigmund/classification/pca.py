import operator

import pandas as pd
from src.pipelinelib.querying import Queryable
from typing import Dict
from src.sigmund.extensions import CLASSIFICATION_NAIVE_BAYES, FEATURE_VECTOR
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import PCA
from spacy.tokens import Doc

from src.pipelinelib.component import Component
from src.pipelinelib.extension import Extension


class PCAReduction(Component):
    """
    Performs PCA on the feature dataframe 
    """

    def __init__(self):
        super().__init__(
            PCAReduction.__name__,
            required_extensions=[FEATURE_VECTOR],
            creates_extensions=[PCA])

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

        # dimensionality reduction
        pca = PCA(n_components=2, svd_solver='full')
        embedded = pca.fit_transform(features)

        # explained variance 
        explained_variance = pca.explained_variance_ratio_

        # plot 
        plt.scatter(embedded[:, 0], embedded[:, 1], c=labels, s=30, cmap='Set1')



        return {PCA: predicted}
