
from typing import Dict, List

import pandas as pd

from src.pipelinelib.component import Component
from src.pipelinelib.extension import Extension
from src.pipelinelib.querying import Parser, Queryable
from src.sigmund.extensions import FEATURE_VECTOR, TFIDF_DOCUMENT


class FeatureMerger(Component):

    def __init__(self):  # , candidates: List[Extension]):
        #self.candidates = candidates
        super().__init__(FeatureMerger.__name__,
                         required_extensions=[TFIDF_DOCUMENT],  # self.specifics or [],
                         creates_extensions=[FEATURE_VECTOR])

    def apply(self, storage: Dict[Extension, pd.DataFrame],
              queryable: Queryable) -> Dict[Extension, pd.DataFrame]:
        # feature_dfs = list(map(
        #    lambda e: e.load_from(storage=storage).add_prefix(e.name),
        #    self.candidates
        # ))

        #joined = pd.merge(feature_dfs, on=Parser.COUPLE_ID, how="inner")
        joined = TFIDF_DOCUMENT.load_from(storage=storage)
        display(joined)

        return {FEATURE_VECTOR: joined}
