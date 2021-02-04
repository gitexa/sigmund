from enum import Enum, unique


@unique
class TextBody(Enum):
    DOCUMENT = 1
    PARAGRAPH = 2
    SENTENCE = 3
