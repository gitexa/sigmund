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

    def __init__(self, name: str):
        self.name = name

    def load_from(self, storage: Dict["Extension", pd.DataFrame]) -> Union[pd.DataFrame, None]:
        return storage.get(key=self, default=None)

    def store_to(self, storage: Dict["Extension", pd.DataFrame], df: pd.DataFrame):
        storage[self] = df

    def __eq__(self, o: object) -> bool:
        return isinstance(o, Extension) and self.name == o.name

    def __hash__(self) -> int:
        return hash(self.name)
