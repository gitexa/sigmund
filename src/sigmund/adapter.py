import operator
from copy import deepcopy

from spacy.tokens import Doc

from src.pipelinelib.component import Component
from src.pipelinelib.extension import Extension


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

    def apply(self, doc: Doc) -> Doc:
        getter = operator.attrgetter(self.old.name)
        setattr(doc._, self.new.name, deepcopy(getter(doc._)))

        return doc
