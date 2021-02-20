from functools import reduce
from typing import Dict, List

import pandas as pd

from src.pipelinelib.component import Component
from src.pipelinelib.extension import Extension
from src.pipelinelib.querying import Parser, Queryable
from src.sigmund.extensions import FEATURE_VECTOR


class FeatureMerger(Component):
    def __init__(self, feature_exts: List[Extension] = []):
        self.candidates = feature_exts
        assert not feature_exts or all(e.is_feature for e in feature_exts)

        super().__init__(FeatureMerger.__name__,
                         required_extensions=self.candidates,
                         creates_extensions=[FEATURE_VECTOR])

    def apply(self, storage: Dict[Extension, pd.DataFrame],
              queryable: Queryable) -> Dict[Extension, pd.DataFrame]:
        # Select provided candidates or all feature vector extensions
        candidates = self.candidates or list(filter(lambda e: e.is_feature, storage))

        # Preemptive exit
        if not candidates:
            return dict()

        # Load desired dfs from storage
        feature_dfs = list(map(
            lambda e: e.load_from(storage=storage), candidates
        ))

        # Rename columns introduced by each component by prepending the Extension name
        # This prevents joins from clashing columns
        for extension, df in zip(candidates, feature_dfs):
            to_rename = [column for column in df.columns if column not in Parser.SCHEMA]
            renamed = [f"{extension.name}_{column}" for column in to_rename]
            df.rename(columns=dict(zip(to_rename, renamed)), inplace=True)

        # Assert join will work
        for extension, feature_df in zip(candidates, feature_dfs):
            if Parser.COUPLE_ID not in feature_df.columns:
                raise AssertionError(
                    f"Missing {Parser.COUPLE_ID} from DataFrame: {extension.name} @ {feature_df.columns}")

        # Cannot perform reduce on less than 2 DFs
        if len(feature_dfs) == 1:
            return {FEATURE_VECTOR: feature_dfs[0]}

        joined = reduce(lambda left, right: pd.merge(
            left, right, on=Parser.COUPLE_ID, how="inner"), feature_dfs)
        return {FEATURE_VECTOR: joined}
