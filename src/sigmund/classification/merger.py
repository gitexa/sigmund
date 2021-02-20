from functools import reduce
from typing import Dict, List

import pandas as pd

from src.pipelinelib.component import Component
from src.pipelinelib.extension import Extension
from src.pipelinelib.querying import Parser, Queryable
from src.sigmund.extensions import FEATURE_VECTOR


class FeatureMerger(Component):
    def __init__(self, feature_exts: List[Extension]):
        self.candidates = feature_exts
        assert all(e.is_feature for e in feature_exts)

        super().__init__(FeatureMerger.__name__,
                         required_extensions=self.candidates,
                         creates_extensions=[FEATURE_VECTOR])

    def apply(self, storage: Dict[Extension, pd.DataFrame],
              queryable: Queryable) -> Dict[Extension, pd.DataFrame]:
        names = list(map(lambda e: e.name, self.candidates))
        print(f"{FeatureMerger.__name__} with {names}")

        names = list(map(lambda e: e.name, storage.keys()))
        print(f"in storage: {names}")

        feature_dfs = list(map(
            lambda e: e.load_from(storage=storage).add_prefix(f"_{e.name}"),
            self.candidates
        ))

        if not feature_dfs:
            return dict()

        if len(feature_dfs) == 1:
            return {FEATURE_VECTOR: feature_dfs[0]}

        for extension, feature_df in zip(self.candidates, feature_dfs):
            if Parser.COUPLE_ID not in feature_df.columns:
                raise AssertionError(
                    f"Missing {Parser.COUPLE_ID} from DataFrame; key = {extension.name}")

        joined = reduce(lambda left, right: pd.merge(
            left, right, on=Parser.COUPLE_ID, how="inner"), feature_dfs)
        return {FEATURE_VECTOR: joined}
