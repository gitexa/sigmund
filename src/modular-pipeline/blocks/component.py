import abc
from typing import Callable, Iterable, Tuple, Union
from itertools import filterfalse
import operator

from spacy.tokens import Doc

from .extension import Extension


class Component(metaclass=abc.ABCMeta):
    def __init__(self, name: str, required_extensions: Iterable[Union[str, Extension]], creates_extensions: Iterable[Extension]):
        self.name = name
        self.required_extensions = required_extensions
        self.creates_extensions = creates_extensions

    @abc.abstractmethod
    def method(self, doc: Doc):
        pass
