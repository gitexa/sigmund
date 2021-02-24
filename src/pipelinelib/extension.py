from typing import Dict, Union

import pandas as pd


class ExtensionKind:
    """
    Differentiate between extensions made by preprocessing,
    feature engineering and classification.

    This is especially important when reviewing DataFrames generated by
    different classes of Component.
    """
    PREPROCESSING = "preprocessing"
    FEATURE = "feature"
    CLASSIFIER = "classifier"


class Extension:
    """
    Represent an entry within the lookup structure.

    Implements utility methods to read and store DataFrames from / to
    the storage container.

    Attributes
    ----------
    name: str
        unique identifier of the extension

    kind: ExtensionKind
        the type of extension
    """

    def __init__(self, name: str, kind: ExtensionKind):
        self.name = name
        self.kind = kind

    def load_from(self, storage: Dict["Extension", pd.DataFrame]) -> Union[pd.DataFrame, None]:
        lookup = storage.get(self, None)
        if lookup is None:
            print(
                f"Warning: {Extension.load_from.__name__} returned None for {self.name}")
            return None

        return lookup.copy(deep=True)

    def store_to(self, storage: Dict["Extension", pd.DataFrame], df: pd.DataFrame):
        storage[self] = df

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, Extension):
            return False

        return self.name == o.name and self.kind == o.kind

    def __hash__(self) -> int:
        return (hash(self.name) << 1) ^ hash(self.kind)

    def __str__(self):
        return self.name

    def __repr__(self) -> str:
        return f"{self.name}({self.kind})"
