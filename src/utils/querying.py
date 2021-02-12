import re
from typing import Callable, Dict, Iterable, Text, Tuple

import docx
import pandas as pd

from src.pipelinelib.text_body import TextBody

# Regexes for extracting strings
_SPEAKER_TEXT_SPLIT_REGEX = re.compile(r"^(([AB])[:;] )(.*)")


class Parser:
    """
    Reads document(s) into pandas DataFrames so
    that they can be queried on
    """

    DOCUMENT_ID = "document_id"
    PARAGRAPH_ID = "paragraph_id"
    SENTENCE_ID = "sentence_id"
    COUPLE_ID = "couple_id"

    IS_DEPRESSED_GROUP = "is_depressed_group"
    DEPRESSED_PERSON = "depressed_person"

    GENDER = "gender"
    SPEAKER = "speaker"

    SENTENCE = "text"

    def __init__(self, nlp: Callable, metadata_path: str):
        self.nlp = nlp
        self.supp_exts = (".docx", ".csv")

        # Loads transcription metadata of institute
        self.metadata = pd.read_excel(metadata_path)

        self.frame = pd.DataFrame(columns=[
            Parser.DOCUMENT_ID,
            Parser.PARAGRAPH_ID,
            Parser.SENTENCE_ID,
            Parser.COUPLE_ID,

            Parser.SPEAKER,
            Parser.GENDER,
            Parser.IS_DEPRESSED_GROUP,
            Parser.DEPRESSED_PERSON,

            Parser.SENTENCE
        ])

    def read_from_files(self, paths: Iterable[str]):
        for path in paths:
            self.update_frame(path)

    def update_frame(self, path2file: str):
        if not any(path2file.endswith(ext) for ext in self.supp_exts):
            raise ValueError(
                f"{Parser.__name__} supports reading from {', '.join(self.supp_exts)}")

        print(f"=== {Parser.__name__}: reading from {path2file} ===")

        modifies_nlp = not self.nlp.has_pipe("sentencizer")
        if modifies_nlp:
            self.nlp.add_pipe(self.nlp.create_pipe("sentencizer"))

        self._read_docx_into_frame(path2file) \
            if path2file.endswith(".docx") \
            else self._read_csv_into_frame(path2file)  # path2file.endswith(".csv")

        if modifies_nlp:
            self.nlp.remove_pipe("sentencizer")

    def _read_docx_into_frame(self, path2file: str, couple_id_pat: str = re.compile(
            r"Paar (\d+)")) -> None:
        # Read document
        document = docx.Document(path2file)
        couple_id = next(re.finditer(couple_id_pat, path2file)).group(1)
        new_document_id = len(self.frame[Parser.DOCUMENT_ID].unique())

        paragraph_w_speakers = [paragraph.text for paragraph in document.paragraphs]
        genders_lookup = {
            "A": _extract_gender(paragraph_w_speakers[1]),
            "B": _extract_gender(paragraph_w_speakers[2])}

        # Skip None, A, B with [3:]
        speaker_paragraphs = [_split_speaker_text(
            p) for p in paragraph_w_speakers][3:]
        paragraph_split = map(list, zip(*speaker_paragraphs))
        paragraph_speakers = next(paragraph_split)
        paragraphs = next(paragraph_split)

        speaker_pid_sentence = [
            (speaker, pid, sentence)
            for pid, (speaker, paragraph) in enumerate(zip(paragraph_speakers, paragraphs))
            for sentence in _split_sentences(paragraph, self.nlp)
        ]
        speaker_pid_sentence = map(list, zip(*speaker_pid_sentence))

        sentence_speakers = next(speaker_pid_sentence)
        paragraph_ids = next(speaker_pid_sentence)
        sentences = next(speaker_pid_sentence)

        sentence_genders = [genders_lookup[speaker] for speaker in sentence_speakers]
        sentence_count = len(sentences)

        couple_metadata = self.metadata.query(f"Paarnummer == {couple_id}")

        # NOTE: our dataset says that group id 1 means W is depressed
        group_id = couple_metadata["Gruppe"].iloc[0]
        is_depressed_group = bool(group_id)
        depressed_person = None if not is_depressed_group else "W"

        collected = {
            Parser.DOCUMENT_ID: [new_document_id] * sentence_count,
            Parser.PARAGRAPH_ID: paragraph_ids,
            Parser.SENTENCE_ID: list(range(sentence_count)),

            Parser.COUPLE_ID: [couple_id] * sentence_count,
            Parser.SPEAKER: sentence_speakers,
            Parser.GENDER: sentence_genders,

            Parser.IS_DEPRESSED_GROUP: [is_depressed_group] * sentence_count,
            Parser.DEPRESSED_PERSON: [depressed_person] * sentence_count,

            Parser.SENTENCE: sentences
        }

        new_df = pd.DataFrame.from_dict(collected, orient="index").transpose()
        self.frame = self.frame.append(new_df)

    def _read_csv_into_frame(self, path2file: str) -> None:
        # Treat these all as new documents, which means offsetting by the amount of assigned document ids
        csv_df = pd.read_csv(path2file)[
            Parser.DOCUMENT_ID] + len(self.frame[Parser.DOCUMENT_ID].unique())
        self.frame = pd.concat(self.frame, csv_df)


def _extract_gender(text: str) -> str:
    without_prefix = re.sub("[AB]: ", "", text)
    without_prefix = re.sub("Er", "M", without_prefix)
    without_prefix = re.sub("Sie", "W", without_prefix)
    return without_prefix


def _split_speaker_text(text: str) -> Tuple[str, str]:
    matches = _SPEAKER_TEXT_SPLIT_REGEX.match(text)
    return (None, None) if not matches else matches.group(2, 3)


def _clean_comments(text: str) -> str:
    """
    Remove Comments from Text
    :type text: str
    :param text: text to remove comments from
    :return: cleaned text
    """
    if text:
        text = re.sub(r"\(.*?\)", "", text)
        text = re.sub(r"\[.*?\]", "", text)
        # Remove Digits and Symbols
        # Remove whitespace
        text = re.sub(' +', ' ', text)
        text = re.sub(r' \.', '.', text)

        return text
    else:
        return " "


def _split_sentences(text: str, nlp) -> list:
    if not text:
        return []
    return [_clean_comments(sentence.text) for sentence in nlp(text).sents
            if sentence and sentence.text]


class Queryable:
    def __init__(self, dataframe: pd.DataFrame, nlp):
        self.df = dataframe.copy(deep=True)
        self.nlp = nlp
        self.query = list()

        # Metadata option defaults
        self.with_gender = True

    def nlp(self):
        return self.nlp

    def from_parser(parser: Parser) -> "Queryable":
        return Queryable(parser.frame, parser.nlp)

    def is_depressed(self) -> "Queryable":
        return self._extend_query_with(f"{Parser.DEPRESSED}")

    def by_couple_id(self, couple_id: int) -> "Queryable":
        return self._extend_query_with(f"{Parser.COUPLE_ID} == {couple_id}")

    def by_speaker(self, speaker: str) -> "Queryable":
        assert speaker in "AB"
        return self._extend_query_with(f"{Parser.SPEAKER} == {speaker}")

    def add_custom_query(self, query: str) -> "Queryable":
        return self._extend_query_with(query=query)

    def without_gender(self) -> "Queryable":
        self.with_gender = False
        return self

    def execute(self, level: TextBody, empty_query=True) -> pd.DataFrame:
        # Join query together
        preced = map(lambda q: f"({q})", self.query)
        joined = " & ".join(preced)

        # Do not modify dataframe read from document!
        df = self.df.query(joined, inplace=False)

        # TODO: Merge columns that pertain to levels of
        # TODO: information not included on the provided TextBody

        # Document merging: Join all sentences of the document
        # according to the level
        if level == TextBody.DOCUMENT:
            group_column, dropped = Parser.DOCUMENT_ID, (
                Parser.PARAGRAPH_ID, Parser.SENTENCE_ID)
        elif level == TextBody.PARAGRAPH:
            group_column, dropped = Parser.PARAGRAPH_ID, (Parser.SENTENCE_ID)
        else:  # level == TextBody.SENTENCE
            group_column, dropped = Parser.SENTENCE_ID, ()

        # TODO: Consider caching these to avoid recalculating these time we execute
        df = df.groupby(group_column, as_index=False)[Parser.SENTENCE].agg(" ".join)
        df.drop(dropped, axis=1, inplace=True)

        # Metadata modifications, can occur inplace
        # as we are working with an out-of-place copy
        if not self.with_gender:
            df.drop(Parser.GENDER, axis=1, inplace=True)

        # Reset query list
        if empty_query:
            self.query.clear()

        return df

    def _extend_query_with(self, query: str) -> "Queryable":
        self.query.append(query)
        return self
