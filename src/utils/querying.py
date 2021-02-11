import re
from typing import Callable, Tuple

import docx
from pandas import pd

from src.pipelinelib.text_body import TextBody

# Regexes for extracting strings
_SPEAKER_TEXT_SPLIT_REGEX = re.compile(r"^(([AB])[:;] )(.*)")


class Parser:
    """
    Reads document(s) into pandas DataFrames so
    that they can be queried on
    """

    DEPRESSED = "is_depressed"
    COMMENT = "comment"
    GENDER = "gender"
    SPEAKER = "speaker"

    DOCUMENT_ID = "document_id"
    PARAGRAPH_ID = "paragraph_id"
    SENTENCE_ID = "sentence_id"
    COUPLE_ID = "couple_id"

    def __init__(self, nlp: Callable, path2file: str):
        self.nlp = nlp
        self.supp_exts = (".docx", ".csv")

        self.frame = pd.DataFrame()
        self._update_frame(path2file)

    def _update_frame(self, path2file: str):
        if not any(path2file.endswith(ext) for ext in self.supp_exts):
            raise ValueError(
                f"{Parser.__name__} supports reading from {', '.join(self.supp_exts)}")

        modifies_nlp = not self.nlp.has_pipe("sentencizer")
        if modifies_nlp:
            self.nlp.add_pipe(self.nlp.create_pipe("sentencizer"))

        df = self._read_docx_into_frame(path2file) \
            if path2file.endswith(".docx") \
            else self._read_csv_into_frame(path2file)  # path2file.endswith(".csv")

        if modifies_nlp:
            self.nlp.remove_pipe("sentencizer")

        return df

    def _read_docx_into_frame(self, path2file: str) -> None:
        # Read document
        document = docx.Document(path2file)
        new_document_id = len(self.frame[Parser.DOCUMENT_ID].unique())

        # Get all paragraphs
        paragraph_w_speakers = [paragraph.text for paragraph in document.paragraphs]
        genders_lookup = {
            "A": _extract_gender(paragraph_w_speakers[1]),
            "B": _extract_gender(paragraph_w_speakers[2])}
        # Speakers and their paragraphs
        speaker_paragraph_tups = [_split_speaker_text(
            p.text) for p in paragraph_w_speakers]
        speakers = [speaker for speaker, _ in speaker_paragraph_tups]
        paragraphs = [paragraph for _, paragraph in speaker_paragraph_tups]

        del speaker_paragraph_tups

        # Deduce genders for speakers
        genders = [genders_lookup[speaker] for speaker in speakers]

        # Extract sentences
        sentences_w_id = [(new_document_id, p_idx, sent_idx)
                          for p_idx, paragraph in enumerate(paragraphs)
                          for sent_idx, sentence in enumerate(_split_sentences(paragraph))]

        document_ids, paragraph_ids, sentence_ids, sentences = zip(*sentences_w_id)

        self.frame = self.frame.append({
            Parser.DOCUMENT_ID: document_ids,
            Parser.PARAGRAPH_ID: paragraph_ids,
            Parser.SENTENCE_ID: sentence_ids,
            Parser.SENTENCES: 
        })

    def _read_csv_into_frame(self, path2file: str) -> None:
        self.frame = pd.read_csv(path2file)


def _extract_gender(text: str) -> str:
    without_prefix = re.sub("[AB]: ", "", text)
    without_prefix = re.sub("Er", "M", without_prefix)
    without_prefix = re.sub("Sie", "W", without_prefix)
    return without_prefix


def _split_speaker_text(text: str) -> Tuple[str, str]:
    matches = _SPEAKER_TEXT_SPLIT_REGEX.match(text)
    return (None, None) if not matches else matches.group(2, 3)


def _split_sentences(text: str, nlp):
    return [sentence for sentence in nlp(text).sents]


class Queryable:
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe
        self.query = list()

        # Metadata options, handled separately from panda queries
        self.with_comments = True
        self.with_gender = True

    def is_depressed(self) -> "Queryable":
        return self._extend_query_with(f"{Parser.DEPRESSED}")

    def by_couple_id(self, couple_id: int) -> "Queryable":
        return self._extend_query_with(f"{Parser.COUPLE_ID} == {couple_id}")

    def by_speaker(self, speaker: str) -> "Queryable":
        assert speaker in "AB"
        return self._extend_query_with(f"{Parser.SPEAKER} == {speaker}")

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

        # TODO: Merge columns that pertain to levels of
        # TODO: information not included on the provided TextBody
        if level < TextBody.DOCUMENT:
            df.drop(Parser.DOCUMENT_ID, axis=1, inplace=True)
        if level < TextBody.PARAGRAPH:
            df.drop(Parser.PARAGRAPH_ID, axis=1, inplace=True)
        if level < TextBody.SENTENCE:
            df.drop(Parser.SENTENCE_ID, axis=1, inplace=True)

        return Queryable(dataframe=df)
