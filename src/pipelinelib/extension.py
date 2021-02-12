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

    def load_from(self, storage: Dict[str, pd.DataFrame]) -> Union[pd.DataFrame, None]:
        return storage.get(key=self.name, default=None)

    def store_to(self, storage: Dict[str, pd.DataFrame], df: pd.DataFrame):
        storage[self.name] = df
