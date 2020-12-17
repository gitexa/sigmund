import abc
from typing import Iterable, Union

from spacy.tokens import Doc

from .extension import Extension


class Component(metaclass=abc.ABCMeta):
    """
    A class that represents a single step in the analytics pipeline

    Attributes
    ----------
    name: str
        name of this component

    required_extensions: Iterable[str | Extension]
        extensions (or their names) that this step depends on

    creates_extensions:
        extensions that this step will create, and shall not exist prior to this step's application
    """
    def __init__(self, name: str, required_extensions: Iterable[Union[str, Extension]],
                 creates_extensions: Iterable[Extension]):
        self.name = name
        self.required_extensions = required_extensions
        self.creates_extensions = creates_extensions

    @abc.abstractmethod
    def apply(self, doc: Doc):
        """
        Implement this method in derivatives of this class to perform
        a step in the analytics pipeline
        """
        pass
