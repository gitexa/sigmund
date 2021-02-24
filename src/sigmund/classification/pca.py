from functools import reduce
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
from IPython.core.display import display
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score

from src.pipelinelib.component import Component
from src.pipelinelib.extension import Extension
from src.pipelinelib.querying import Parser, Queryable
from src.pipelinelib.text_body import TextBody
from src.sigmund.extensions import FEATURE_VECTOR, PCA_REDUCTION


class PCAReduction(Component):
    """
    Performs Principal Component Analysis on a feature vector.

    PCA is useful when attempting to lower the dimensionality of a
    dataset in order to avoid overfitting, similarly to LDA.

    Unlike LDA, PCA is an unsupervised transformation technique,
    i.e. it ignores class labels.

    Instead, PCA calculates orthogonal vectors in an attempt to detect and
    remove features that are related, and thereby redundant.
    """

    def __init__(self, inputs: List[Extension] = None,
                 output: Extension = None):
        self.inputs = inputs or [FEATURE_VECTOR]
        self.output = output or PCA_REDUCTION

        super().__init__(
            PCAReduction.__name__,
            required_extensions=self.inputs,
            creates_extensions=[self.output])

    def apply(self, storage: Dict[Extension, pd.DataFrame],
              queryable: Queryable) -> Dict[Extension, pd.DataFrame]:

        # load feature from storage
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

        return {self.output: df_embedded}

    def visualise(self, created: Dict[Extension, pd.DataFrame], queryable: Queryable):

        df_embedded = self.output.load_from(storage=created)

        plt.figure()
        plt.scatter(
            df_embedded['dim1'],
            df_embedded['dim2'],
            c=df_embedded['is_depressed_group'],
            s=30, cmap='Set1')
