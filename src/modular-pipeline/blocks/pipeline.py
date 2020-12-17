import operator
from itertools import filterfalse
from spacy.tokens import Doc

from .component import Component
from .extension import Extension


class Pipeline:
    def __init__(self, model):
        self.model = model
        self.extensions = dict()

    def add_component(self, component: Component):
        self._is_compatible(component)
        self._register_extensions(component)
        self._register_pipe(component)

        return self

    def execute(self, text: str):
        return self.model(text)

    def _is_compatible(self, component: Component):
        names = map(lambda x: x.name if isinstance(
            x, Extension) else x, component.required_extensions)
        missing_extensions = list(filterfalse(self.extensions.__contains__, names))
        if missing_extensions:
            raise Exception(f"Unable to apply {component.name} to pipeline: missing extensions " +
                            f"from Doc object:\n{', '.join(missing_extensions)}")

        names = map(operator.attrgetter("name"),
                    component.creates_extensions)
        overwritten_extensions = list(filter(self.extensions.__contains__, names))
        if overwritten_extensions:
            raise Exception(f"Unable to apply {component.name} to pipeline: would overwrite extensions " +
                            f"in Doc object:\n{', '.join(overwritten_extensions)}")

    def _register_extensions(self, component: Component):
        for extension in component.creates_extensions:
            self.extensions[extension.name] = extension.default_type

    def _register_pipe(self, component: Component):
        self.model.add_pipe(component.method)
