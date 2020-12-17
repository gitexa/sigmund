import operator
from itertools import filterfalse
from typing import Any, Dict

from spacy.tokens import Doc

from .component import Component
from .extension import Extension


class Pipeline:
    """
    A class to represent a pipeline for training a model

    Attributes
    ----------
    _model
        the spacy model to be trained

    _extensions: dict
        a collection of extensions that the pipeline will create on
        spacy's Doc instance.
    """

    def __init__(self, model):
        self._model = model
        self._extensions: Dict[str, Any] = dict()

    def add_component(self, component: Component):
        """
        Add a component to the pipeline, with checks
        """
        self._is_compatible(component)
        self._register_extensions(component)
        self._register_pipe(component)

        return self

    def execute(self, text: str) -> Doc:
        """
        Execute the pipeline with the registered components
        """
        return self._model(text)

    def _is_compatible(self, component: Component):
        """
        Check that the component will not overwrite a preregistered extension,
        or if it depends on an extension that has not been declared yet
        """

        # differentiate between string and Extension instance
        names = map(lambda x: x.name if isinstance(
            x, Extension) else x, component.required_extensions)

        # depends on non-existent Extension
        missing_extensions = list(filterfalse(self._extensions.__contains__, names))
        if missing_extensions:
            raise Exception(
                f"Unable to apply {component.name} to pipeline: missing extensions " +
                f"from Doc object:\n{', '.join(missing_extensions)}")

        # read names from Extensions
        names = map(operator.attrgetter("name"),
                    component.creates_extensions)
        # would overwrite pre-existing Extensions
        overwritten_extensions = list(filter(self._extensions.__contains__, names))
        if overwritten_extensions:
            raise Exception(
                f"Unable to apply {component.name} to pipeline: would overwrite extensions "
                + f"in Doc object:\n{', '.join(overwritten_extensions)}")

    def _register_extensions(self, component: Component):
        """
        Register extensions declared in component
        """
        for extension in component.creates_extensions:
            self._extensions[extension.name] = extension.default_type

    def _register_pipe(self, component: Component):
        """
        Add the transformation declared by the component to the pipeline
        """
        self._model.add_pipe(component.apply)
