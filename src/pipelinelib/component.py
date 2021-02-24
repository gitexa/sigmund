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

    required_extensions: Iterable[Extension]
        extensions that this step depends on

    creates_extensions: Iterable[Extension]
        extensions that this step will create that shall not exist prior to this step's application
    """

    def __init__(self, name: str, required_extensions: Iterable[Extension],
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

        return results

    @abc.abstractmethod
    def apply(self, storage: Dict[Extension, pd.DataFrame],
              queryable: Queryable) -> Dict[Extension, pd.DataFrame]:
        """
        Implement this method to perform a step in the analytics pipeline
        """
        pass

    def visualise(self, created: Dict[Extension, pd.DataFrame], queryable: Queryable):
        """
        Implement this method in order to visualise DataFrames created by the apply method
        (see Component.apply)
        """
        pass
