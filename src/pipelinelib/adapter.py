from typing import Dict

import pandas as pd

from src.pipelinelib.component import Component
from src.pipelinelib.extension import Extension
from src.pipelinelib.querying import Queryable


class Adapter(Component):
    """
    Creates a new entry within the lookup structure,
    in case DataFrames need to be copied to different registers
    for generic Components to work correctly
    """

    def __init__(self, old: Extension, new: Extension):
        super().__init__(
            name=Adapter.__name__, required_extensions=[old],
            creates_extensions=[new])

        self.old = old
        self.new = new

    def apply(self, storage: Dict[Extension, pd.DataFrame],
              queryable: Queryable) -> Dict[Extension, pd.DataFrame]:
        df = self.old.load_from(storage=storage)
        return {self.new: df.copy(deep=True)}
