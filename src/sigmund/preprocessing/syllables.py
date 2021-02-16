import string
from typing import Dict

import pandas as pd
import pyphen
from spacy.tokens import Doc

from src.pipelinelib.component import Component
from src.pipelinelib.extension import Extension
from src.pipelinelib.querying import Queryable
from src.sigmund.extensions import SYLLABLES, TOKENS


class SyllableExtractor(Component):
    """
    Extracts syllables from text and stores these under doc._.syllables
    """

    def __init__(self):
        super().__init__(name=SyllableExtractor.__name__,
                         required_extensions=list(TOKENS),
                         creates_extensions=[SYLLABLES])
        self.dic = pyphen.Pyphen(lang='de')

    def apply(self, storage: Dict[Extension, pd.DataFrame],
              queryable: Queryable) -> Dict[Extension, pd.DataFrame]:
        #for token in doc:
        #    if not str(token) in string.punctuation:
        #        doc._.syllables.extend(self.dic.inserted(str(token)).split("-"))
        #
        #    SYLLABLES.store_to(storage)
        #return doc
        
        
        return {self.new: }
