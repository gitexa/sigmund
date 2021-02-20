import abc
from typing import Dict, Iterable, Tuple, Union

import pandas as pd
from spacy.tokens import Doc

from src.pipelinelib.querying import Queryable

from .extension import Extension


class Component(metaclass=abc.ABCMeta):
    """
    A class that represents a single step in the analytics pipeline

    Attributes
    ----------
    name: str
        name of this component

    required_extensions: Iterable[str | Extension]
        extensions (or their names) that this step depends on

    creates_extensions:
        extensions that this step will create, and shall not exist prior to this step's application
    """

    def __init__(self, name: str, required_extensions: Iterable[Union[str, Extension]],
                 creates_extensions: Iterable[Extension]):
        self.name = name
        self.required_extensions = required_extensions
        self.creates_extensions = creates_extensions

    def _internal_apply(self, storage: Dict[Extension, pd.DataFrame],
                        queryable: Queryable) -> Dict[Extension, pd.DataFrame]:
        """
        Wrapper around the abstract apply method with output
        """
        print(f"Executing {self.name}")

        results = self.apply(storage=storage, queryable=queryable)
        dfs = results.values()

        # Check that all column names in returned dataframes are unique
        all_column_names = list(column_name for df in dfs for column_name in df.columns)
        unique_column_names = set(all_column_names)

        if len(unique_column_names) != len(all_column_names):
            raise AttributeError(
                f"{self.name} produced dataframes with an non-empty intersection of column names")

        return results

    @abc.abstractmethod
    def apply(self, storage: Dict[Extension, pd.DataFrame],
              queryable: Queryable) -> Dict[Extension, pd.DataFrame]:
        """
        Implement this method in derivatives of this class to perform
        a step in the analytics pipeline
        """
        pass

    @abc.abstractmethod
    def visualise(self, created: Dict[Extension, pd.DataFrame]):
        return None
