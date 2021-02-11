from typing import Dict

import pandas as pd
from spacy.tokens import Doc

from src.pipelinelib.component import Component
from src.pipelinelib.extension import Extension
from src.utils.querying import Queryable


class Adapter(Component):
    """
    Creates a register in the Spacy document that can be used for 
    other steps in the pipeline
    """

    def __init__(self, old: Extension, new: Extension):
        super().__init__(
            name=Adapter.__name__, required_extensions=[old],
            creates_extensions=[new])

        self.old = old
        self.new = new

    def apply(self, storage: Dict[str, pd.DataFrame],
              queryable: Queryable) -> Dict[Extension, pd.DataFrame]:
        df = self.old.load_from(storage=storage)
        return {self.new: df.copy(deep=True)}
