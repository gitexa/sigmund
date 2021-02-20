from copy import deepcopy
from typing import Dict, Union

import pandas as pd


class Extension:
    """
    A class to represent an extension member of spacy's
    DocInstance type.

    Implements utility methods to read and store data frames to your
    desired storage container.

    Attributes
    ----------
    name: str
        identifier of the extension
    """

    def __init__(self, name: str, is_feature: bool = False):
        self.name = name
        self.is_feature = is_feature

    def load_from(self, storage: Dict["Extension", pd.DataFrame]) -> Union[pd.DataFrame, None]:
        lookup = storage.get(self, None)
        if lookup is None:
            print(
                f"Warning: {Extension.load_from.__name__} returned None for {self.name}")
            return None

        return storage[self].copy(deep=True)

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, Extension):
            return False

        return self.name == o.name and self.is_feature == o.is_feature

    def __hash__(self) -> int:
        return (hash(self.name) << 1) ^ hash(self.is_feature)

    def store_to(self, storage: Dict["Extension", pd.DataFrame], df: pd.DataFrame):
        storage[self] = df

    def __str__(self):
        return self.name
