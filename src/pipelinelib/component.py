import abc
import operator
from itertools import filterfalse
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

    def _internal_apply(self, doc: Doc) -> Doc:
        """
        Wrapper around the abstract apply method 
        that automatically intialises declared extensions
        on the doc
        """
        for extension in filterfalse(lambda e: doc.has_extension(e.name), self.creates_extensions):
            doc.set_extension(
                extension.name, default=extension.default_type)

        return self.apply(doc)

    @abc.abstractmethod
    def apply(self, doc: Doc) -> Doc:
        """
        Implement this method in derivatives of this class to perform
        a step in the analytics pipeline
        """
        pass
