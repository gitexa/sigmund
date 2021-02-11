import operator

from spacy.tokens import Doc


class Extension:
    """
    A class to represent an extension member of spacy's
    DocInstance type

    Attributes
    ----------
    name: str
        identifier of the extension

    default_type:
        instance of type that the extension shall be
        initialised with.
        This should not change during runtime
    """

    def __init__(self, name: str, default_type):
        self.name = name
        self.default_type = default_type

    def load_from(self, doc: Doc):
        return operator.attrgetter(self.name)(doc._)

    def store_to(self, doc: Doc, value):
        # TODO: operator.attrsetter version
        doc._[self.name] = value
