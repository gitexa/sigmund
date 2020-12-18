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
