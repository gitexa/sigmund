import abc
from pathlib import Path
from typing import Callable, Union

import docx
from pandas import pd

from src.pipelinelib.text_body import TextBody


class Parser:
    """
    Reads document(s) into pandas DataFrames so
    that they can be queried on
    """

    DEPRESSED = "is_depressed"
    COMMENT = "comment"
    GENDER = "gender"
    DOCUMENT_ID = "document_id"
    PARAGRAPH_ID = "paragraph_id"
    SENTENCE_ID = "sentence_id"
    COUPLE_ID = "couple_id"

    def __init__(self, path2file: str, nlp: Callable):
        self.frame = pd.DataFrame()
        self.nlp = nlp
        self.supp_exts = (".docx", ".csv")

        self._read_into_frame(path2file)

    def _read_into_frame(self, path2file: str):
        if not any(path2file.endswith(ext) for ext in self.supp_exts):
            raise ValueError(
                f"{Parser.__name__} supports reading from {', '.join(self.supp_exts)}")

        modifies_nlp = not self.nlp.has_pipe("sentencizer")
        if modifies_nlp:
            self.nlp.add_pipe(self.nlp.create_pipe("sentencizer"))

        if path2file.endswith(".docx"):
            self._read_docx_into_frame(path2file)
        else:  # path2file.endswith(".csv")
            self._read_csv_into_frame(path2file)

        if modifies_nlp:
            self.nlp.remove_pipe("sentencizer")

    def _read_docx_into_frame(self, path2file: str) -> None:
        pass

    def _read_csv_into_frame(self, path2file: str) -> None:
        pass


class Queryable:
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe
        self.query = list()

        # Metadata options, handled separately from panda queries
        self.with_comments = True
        self.with_gender = True

    def is_depressed(self) -> "Queryable":
        return self._extend_query_with(f"{Parser.DEPRESSED}")

    def by_couple_id(self, couple_id) -> "Queryable":
        return self._extend_query_with(f"{Parser.COUPLE_ID} == {couple_id}")

    def without_comments(self) -> "Queryable":
        self.with_comments = False
        return self

    def without_gender(self) -> "Queryable":
        self.with_gender = False
        return self

    def execute(self) -> pd.DataFrame:
        # Join query together
        preced = map(lambda q: f"({q})", self.query)
        joined = " & ".join(preced)

        # Do not modify dataframe read from document!
        df = self.df.query(joined, inplace=False)

        # Metadata modifications, can occur inplace
        # as we are working with an out-of-place copy
        if not self.with_comments:
            df.drop(Parser.COMMENT, axis=1, inplace=True)
        if not self.with_gender:
            df.drop(Parser.GENDER, axis=1, inplace=True)

        return df

    def _extend_query_with(self, query: str) -> "Queryable":
        self.query.append(query)
        return self


class QueryBuilder:
    def __init__(self, parser: Parser):
        self.parser = parser

    def on_corpus_level(self, level: TextBody) -> Queryable:
        df = self.parser.frame.copy(deep=True)

        # Remove columns that pertain to levels of
        # information not included on the provided TextBody
        if level < TextBody.DOCUMENT:
            df.drop(Parser.DOCUMENT_ID, axis=1, inplace=True)
        if level < TextBody.PARAGRAPH:
            df.drop(Parser.PARAGRAPH_ID, axis=1, inplace=True)
        if level < TextBody.SENTENCE:
            df.drop(Parser.SENTENCE_ID, axis=1, inplace=True)

        return Queryable(dataframe=df)
