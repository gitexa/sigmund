import functools
import os
import re
from itertools import filterfalse
from os import system
from typing import Callable, Iterable, Tuple

import docx
import numpy as np
import pandas as pd

from src.pipelinelib.text_body import TextBody

# Regexes for extracting strings
_SPEAKER_TEXT_SPLIT_REGEX = re.compile(r"^(([AB])[:;] )(.*)")


class Parser:
    """
    Reads document(s) into pandas DataFrames so that they can be queried on.

    Extracted data and their types can be reviewed in SCHEMA
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
    HAMILTON_SCORE = "hamilton"

    SCHEMA = {
        DOCUMENT_ID: np.int64,
        PARAGRAPH_ID: np.int64,
        SENTENCE_ID: np.int64,
        COUPLE_ID: np.int64,

        SPEAKER: str,
        GENDER: str,
        IS_DEPRESSED_GROUP: np.bool_,
        DEPRESSED_PERSON: str,

        SENTENCE: str,
        HAMILTON_SCORE: np.int64
    }

    def __init__(self, nlp: Callable, metadata_path: str):
        """
        @param nlp: Spacy's natural language processing object, initialised with model for the corresponding language
        @param metadata_path: path to a XLS containing all relevant metadata for the relevant transcripts.
        """
        self.nlp = nlp
        self.supp_exts = (".docx", ".csv")

        # Loads transcription metadata of institute
        self.metadata = pd.read_excel(metadata_path)

        self.frame = pd.DataFrame(columns=Parser.SCHEMA.keys()) \
            .astype(dtype=Parser.SCHEMA)

    def read_from_files(self, paths: Iterable[str]) -> None:
        """
        Loads multiple documents with the appropriate file format into the DataFrame.
        Currently supported extensions are reviewable under self.supp_exts

        @param paths: the paths to the files to be loaded
        """
        print('Read files')
        for path in paths:
            if "T2" in path or "t2" in path:  # TODO improve quickfix
                print(path)
                print("Skipped T2.")
            elif path in [
                    os.path.join(os.getcwd(), "data", "all_transcripts", "Paar_104_IM.docx"),
                    os.path.join(os.getcwd(), "data", "all_transcripts", "Paar_7_IM.docx"),
                    os.path.join(os.getcwd(), "data", "all_transcripts", "Paar_60_T1_IM_FW.docx"),
                    os.path.join(os.getcwd(), "data", "all_transcripts", "Paar_82_T1_IM.docx"),
                    os.path.join(os.getcwd(), "data", "all_transcripts", "Paar_86_IM.docx"),
                    os.path.join(os.getcwd(), "data", "all_transcripts", "Paar_91_IM.docx"),
                    os.path.join(os.getcwd(), "data", "all_transcripts", "Paar_119_T1_IM.docx")]:
                print(f"Skipped: {path}")  # TODO fix paths
            else:
                self.update_frame(path)

    def update_frame(self, path2file: str) -> None:
        """
        Loads an document with the appropriate file format into the DataFrame.
        Currently supported extensions are reviewable under self.supp_exts

        @param path2file: the path to the file to be read
        """
        if not any(path2file.endswith(ext) for ext in self.supp_exts):
            raise ValueError(
                f"{Parser.__name__} supports reading from {', '.join(self.supp_exts)}")

        print(f"=== {Parser.__name__}: reading from {path2file} ===")

        modifies_nlp = not self.nlp.has_pipe("sentencizer")
        if modifies_nlp:
            self.nlp.add_pipe(self.nlp.create_pipe("sentencizer"))

        self._read_docx_into_frame(path2file) \
            if path2file.endswith(".docx")\
            else self._read_csv_into_frame(path2file)  # path2file.endswith(".csv")

        if modifies_nlp:
            self.nlp.remove_pipe("sentencizer")

    def _read_docx_into_frame(self, path2file: str, couple_id_pat: str = re.compile(
            r"Paar_(\d+)")) -> None:  # TODO make generic

        # Read document
        document = docx.Document(path2file)
        couple_id = next(re.finditer(couple_id_pat, path2file)).group(1)
        new_document_id = len(self.frame[Parser.DOCUMENT_ID].unique())
        paragraph_w_speakers = [paragraph.text for paragraph in document.paragraphs]
        genders_lookup = _parse_speakers(paragraph_w_speakers)

        # remove legend (A,B,title)
        paragraph_w_speakers_wo_legend = _remove_legend(paragraph_w_speakers)

        # split speaker and paragrah (A: | "lorem ipson")
        paragraph_w_speakers_separated = [_split_speaker_text(
            p) for p in paragraph_w_speakers_wo_legend]  # TODO change as it might be variable 2/3/...

        # split speakers and paragraphs
        paragraph_split = map(list, zip(*paragraph_w_speakers_separated))
        paragraphs_speakers = next(paragraph_split)
        paragraphs_text = next(paragraph_split)

        # remove comments
        paragraphs_text, paragraphs_speakers = _clean_comments(
            paragraphs_text, paragraphs_speakers)

        # Split into sentences and assign ids to paragraphs and sentences
        speaker_pid_sentence = [
            (speaker, pid, sentence)
            for pid, (speaker, paragraph) in enumerate(zip(paragraphs_speakers, paragraphs_text))
            for sentence in _split_sentences(paragraph, self.nlp)
        ]

        speaker_pid_sentence = map(list, zip(*speaker_pid_sentence))
        sentence_speakers = next(speaker_pid_sentence)
        paragraph_ids = next(speaker_pid_sentence)
        sentences = next(speaker_pid_sentence)
        sentence_genders = [genders_lookup[speaker] for speaker in sentence_speakers]
        sentence_count = len(sentences)

        #couple_metadata = self.metadata.query(f"Paarnummer == {couple_id}")
        couple_metadata = self.metadata.loc[(
            self.metadata['Paarnummer'] == int(couple_id)) & (self.metadata['mzp'] == 1)]
        couple_metadata_f = self.metadata.loc[(self.metadata['Paarnummer'] == int(
            couple_id)) & (self.metadata['mzp'] == 1) & (self.metadata['Sex'] == 1)]
        couple_metadata_m = self.metadata.loc[(self.metadata['Paarnummer'] == int(
            couple_id)) & (self.metadata['mzp'] == 1) & (self.metadata['Sex'] == 0)]

        # NOTE: our dataset says that group id 1 means W is depressed
        #group_id = couple_metadata["Gruppe"].iloc[0]
        group_tel = couple_metadata["Gruppe_TEL"].iloc[0]  # TODO adapt
        is_depressed_group = bool(group_tel-1)  # TODO improve quickfix
        depressed_person = None if not is_depressed_group else "W"

        # Hamilton Score reading
        #male_hs, female_hs = couple_metadata["HDI.1"]
        female_hs = couple_metadata_f["HDI_SCORE"].iloc[0]
        male_hs = couple_metadata_m["HDI_SCORE"].iloc[0]

        hamilton_scores = [male_hs
                           if speaker == "M" else female_hs
                           for speaker in sentence_genders]

        collected = {
            Parser.DOCUMENT_ID: [new_document_id] * sentence_count,
            Parser.PARAGRAPH_ID: paragraph_ids,
            Parser.SENTENCE_ID: list(range(sentence_count)),

            Parser.COUPLE_ID: [couple_id] * sentence_count,
            Parser.SPEAKER: sentence_speakers,
            Parser.GENDER: sentence_genders,

            Parser.IS_DEPRESSED_GROUP: [is_depressed_group] * sentence_count,
            Parser.DEPRESSED_PERSON: [depressed_person] * sentence_count,

            Parser.HAMILTON_SCORE: hamilton_scores,

            Parser.SENTENCE: sentences
        }

        new_df = pd.DataFrame.from_dict(
            collected, orient="index").transpose().astype(
            Parser.SCHEMA)

        self.frame = self.frame.append(new_df)

    def _read_csv_into_frame(self, path2file: str) -> None:
        # Treat these all as new documents, which means offsetting by the amount of assigned document ids
        csv_df = pd.read_csv(path2file, index_col=0)
        csv_df[Parser.DOCUMENT_ID] += len(self.frame[Parser.DOCUMENT_ID].unique())
        self.frame = pd.concat(self.frame, csv_df)


def _parse_speakers(paragraph_w_speakers):

    # TODO

    # - Start mit Überschrift
    # - Start mit ''
    # - Start mit A,B
    # - Leere Anführungszeichen löschen
    counter = 0
    a_flag, b_flag = True, True
    for paragraph in paragraph_w_speakers:
        if paragraph.startswith(('A:', 'a:')) and a_flag:
            first_speaker = _extract_gender(paragraph)
            a_flag = False
        if paragraph.startswith(('B:', 'b:')) and b_flag:
            second_speaker = _extract_gender(paragraph)
            b_flag = False
        if not a_flag and not b_flag:
            return {"A": first_speaker, "B": second_speaker}
        counter += 1
        if counter > 5:
            print('Speakers cannot be identified.')
            system.exit(0)


def _extract_gender(text: str) -> str:
    without_prefix = re.sub("[AB]:\s?", "", text)
    without_prefix = re.sub("(Er|er)", "M", without_prefix)
    without_prefix = re.sub("(Sie|sie)", "W", without_prefix)
    return without_prefix[0]


def _split_speaker_text(text: str) -> Tuple[str, str]:
    matches = _SPEAKER_TEXT_SPLIT_REGEX.match(text)
    return (None, None) if not matches else matches.group(2, 3)


def _remove_legend(paragraphs: list) -> list:
    r1 = re.compile("[AB]:\s?(Sie|Er)")
    r2 = re.compile("^$")
    r3 = re.compile("^Paar\s?[0-9]+")
    # best guess: found AAH - 36 or AAI - Paar 37
    r4 = re.compile("^AA[A-Z]\s?[\–,\-]\s?(Paar)?\s?[0-9]+")
    filtered = [i for i in paragraphs if not (
        r1.match(i) or r2.match(i) or r3.match(i) or r4.match(i))]
    return filtered


def _clean_comments(paragraphs_text: list, paragraphs_speaker: list) -> list:
    """
    Remove Comments from Text
    :type text: str
    :param text: text to remove comments from
    :return: cleaned text
    """

    cleaned_paragraphs_text = list()

    index_counter = 0
    for paragraph in paragraphs_text:

        if not paragraph == None:

            # Normalise whitespace
            paragraph = re.sub(r"\s{2,}", "", paragraph)

            # Remove comments
            paragraph = re.sub(r"\(.*?\) ?", "", paragraph)
            paragraph = re.sub(r"\[.*?\] ?", "", paragraph)

            # Remove other extra symbols
            paragraph = re.sub(r"\°\s?", "", paragraph)

            # Append to new list
            if not paragraph == "":
                cleaned_paragraphs_text.append(paragraph)
                index_counter += 1
            else:
                del paragraphs_speaker[index_counter]
        else:
            del paragraphs_speaker[index_counter]

    assert len(cleaned_paragraphs_text) == len(paragraphs_speaker)

    return cleaned_paragraphs_text, paragraphs_speaker


def _split_sentences(text: str, nlp) -> list:
    if not text:
        return []

    #cleaned = _clean_comments(text)
    sentences = nlp(text).sents

    # Remove empty tokens from cleaned
    # strs = map(str, sentences)
    # return list(filterfalse(lambda x: not x, strs))
    return list(sentences)


class Queryable:
    """
    Helper class to treat the cumulative DataFrame as a dataset, such as
    enabling queries to be performed upon said dataset
    """

    def __init__(self, dataframe: pd.DataFrame, nlp):
        """
        @param dataframe: the dataset to query
        @param nlp: the NLP used to create the dataset, which can be used by Component implementations directly
        """
        self.df = dataframe.copy(deep=True)
        self._nlp = nlp
        self.query = list()

    def nlp(self):
        """
        @return: Access to the NLP used to create the dataset
        """
        return self._nlp

    def from_csv(path: str, nlp):
        """
        @param path: path to csv with all transcripts in parsed format
        @return: Queryable 
        """
        df = pd.read_csv(path)
        return Queryable(df, nlp)

    @staticmethod
    def from_parser(parser: Parser) -> "Queryable":
        """
        @param parser: Parser used to load transcripts
        @return: Queryable from the DataFrames collected by the Parser
        """
        return Queryable(parser.frame, parser.nlp)

    def is_depressed(self, d: bool = True) -> "Queryable":
        """
        @param d: Select from the depressed if True, else from the non-depressed group
        @return: The Queryable object
        """
        return self._extend_query_with(f"{Parser.IS_DEPRESSED_GROUP} == {d}")

    def by_couple_id(self, couple_id: int) -> "Queryable":
        """
        @param couple_id: Select data associated with couples with the given ID
        @return: The Queryable object
        """
        return self._extend_query_with(f"{Parser.COUPLE_ID} == {couple_id}")

    def by_speaker(self, speaker: str) -> "Queryable":
        """
        @param speaker:  Select data associated with speaker A or speaker B
        @return: The Queryable object
        """
        assert speaker in "AB"
        return self._extend_query_with(f"{Parser.SPEAKER} == {speaker}")

    def add_custom_query(self, query: str) -> "Queryable":
        return self._extend_query_with(query=query)

    def execute(self, level: TextBody, empty_query=True) -> pd.DataFrame:
        """
        Execute the query built with the is_* and by_* methods
        @param level: Aggregate the queried dataframe on the desired corpus level
        @param empty_query: Empty the query queue after building the DataFrame
        @return: Dataframe containing the requested information and the desired corpus level
        """
        # Join query together
        preced = map(lambda q: f"({q})", self.query)
        joined = " and ".join(preced)

        # print(f"=== {Queryable.__name__} is executing on {level} level, query = '{joined}' ===")

        # Copy, feel free to modify as you like
        df = self._get_agged_frame(level=level).copy()

        if joined:
            df.query(joined, inplace=True)

        # Reset query list
        if empty_query:
            self.reset_query()

        return df

    @functools.lru_cache(maxsize=None)
    def _get_agged_frame(self, level: TextBody) -> pd.DataFrame:
        """
        @param level:
        @return:
        """
        # Do not modify dataframe read from document!
        if level == TextBody.DOCUMENT:
            return self.df.groupby([Parser.DOCUMENT_ID], as_index=False).agg(
                {Parser.COUPLE_ID: "first", Parser.IS_DEPRESSED_GROUP: "first", Parser.
                    DEPRESSED_PERSON: "first", Parser.SENTENCE: ". ".join}
            )
        elif level == TextBody.PARAGRAPH:
            return self.df.groupby([Parser.DOCUMENT_ID, Parser.PARAGRAPH_ID], as_index=False).agg({
                Parser.COUPLE_ID: "first", Parser.SPEAKER: "first", Parser.GENDER: "first",
                Parser.IS_DEPRESSED_GROUP: "first", Parser.DEPRESSED_PERSON: "first",
                Parser.HAMILTON_SCORE: "first",
                Parser.SENTENCE: ". ".join,
            })
        else:  # level == TextBody.SENTENCE
            # self.df is always in sentence format, so just return df
            return self.df

    def reset_query(self):
        self.query.clear()

    def _extend_query_with(self, query: str) -> "Queryable":
        self.query.append(query)
        return self
