import operator
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from spacy.tokens import Doc

from src.pipelinelib.component import Component
from src.pipelinelib.extension import Extension
from src.pipelinelib.querying import Queryable
from src.pipelinelib.text_body import TextBody
from src.sigmund.extensions import *


class PCAReduction(Component):
    """
    Performs PCA on the feature dataframe 
    """

    def __init__(self):
        super().__init__(
            PCAReduction.__name__,
            required_extensions=[FEATURE_VECTOR],
            creates_extensions=[PCA_REDUCTION])

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

        # construct pandas df with labels 
        df_embedded = pd.DataFrame(embedded, columns=['dim1', 'dim2'])
        df_embedded = pd.concat([df_embedded, labels], axis=1)
        display(df_embedded)

        # explained variance 
        explained_variance = pca.explained_variance_ratio_
        display(explained_variance)

        return {PCA_REDUCTION: df_embedded}
    
    def visualise(self, created: Dict[Extension, pd.DataFrame], queryable: Queryable):
        
        df_embedded = PCA_REDUCTION.load_from(storage=created)

        plt.scatter(
            df_embedded['dim1'],
            df_embedded['dim2'],
            c=df_embedded['is_depressed_group'],
            s=30, cmap='Set1')
