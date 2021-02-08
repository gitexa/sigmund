from enum import OrderedEnum, unique


@unique
class TextBody(OrderedEnum):
    DOCUMENT = 5
    PARAGRAPH = 4
    SENTENCE = 3
