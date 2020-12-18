import re
import string
from copy import copy

import docx
import numpy as np
import pandas
import pandas as pd
import spacy
from prettytable import PrettyTable, prettytable
from sklearn.pipeline import Pipeline

nlp = spacy.load("de_core_news_md")
nlp.add_pipe(nlp.create_pipe('sentencizer'))


class DialogueParser:
    @staticmethod
    def split_speaker_text(text_line):
        pattern = re.compile(r"^(([AB])[:;] )(.*)")
        matches = pattern.match(text_line)

        if matches:
            return matches.group(2), matches.group(3)
        else:
            return None, None

    def __init__(
            self, doc_file, group, couple_id, female_label="B", depressed=None,
            remove_annotations=True):

        self.female_label = female_label
        self.group = group
        self.couple_id = couple_id
        self.docx = docx.Document(doc_file)
        self.is_depressed = depressed
        self.id = id

        super().__init__()

    # @ToDo Deal with single Characters and whitespaces interpreted as sentences
    def get_sentences(self, remove_annotations=True, drop_na=True):
        rows = []
        for i, p in enumerate(self.docx.paragraphs[3:]):
            speaker, text = self.split_speaker_text(p.text)

            gender = "W" if speaker == self.female_label else "M"
            depression_label = (gender == "W" and self.is_depressed)

            if speaker:
                if remove_annotations:
                    text = re.sub(r"\(.*\)\w?", "", text)
                    text = re.sub(r"\[.*\]\w?", "", text)
                doc = nlp(text)
                for j, sent in enumerate(doc.sents):
                    row = {
                        "par_pos": i,
                        "sent_pos": j,
                        "speaker": speaker,
                        "raw_text": sent.text,
                        "group": self.group,
                        "gender": gender,
                        "is_depressed": depression_label,
                        "couple_id": self.couple_id}
                    rows.append(row)

        sentence_df = pd.DataFrame(rows)

        if drop_na:
            sentence_df.dropna(inplace=True)

        return sentence_df

    def get_paragraphs(self, remove_annotations=True, drop_na=True):
        rows = []
        for i, p in enumerate(self.docx.paragraphs[3:]):
            speaker, text = self.split_speaker_text(p.text)
            gender = "W" if speaker == self.female_label else "M"
            depression_label = (gender == "W" and self.is_depressed)
            if speaker:
                if remove_annotations:
                    text = re.sub(r"\(.*\)\w?", "", text)
                    text = re.sub(r"\[.*\]\w?", "", text)
                row = {
                    "par_pos": i,
                    "speaker": speaker,
                    "raw_text": text,
                    "group": self.group,
                    "gender": gender,
                    "is_depressed": depression_label,
                    "couple_id": self.couple_id,
                }
                rows.append(row)

        paragraph_df = pd.DataFrame(rows)

        if drop_na:
            paragraph_df.dropna(inplace=True)

        return paragraph_df

    # @ToDo Implement get fulltext
    def get_fulltext(self):
        return self.get_paragraphs().groupby(
            ['gender', "is_depressed"],
            as_index=False).agg(
            {'raw_text': ' '.join})


def clear_annotations(raw_text):
    removed_annotations = re.sub(r"\(.*\)", "", raw_text)
    removed_annotations = ' '.join(removed_annotations.split())
    return removed_annotations


def preprocess(row):
    raw_text = clear_annotations(str.lower(row["raw_text"]))
    doc = nlp(raw_text)
    row["sent_count"] = len(list(doc.sents))
    doc = nlp(' '.join([token.text for token in doc if not token.is_punct]))
    row["word_count"] = len(list(doc))
    row["lemmatized"] = ' '.join(
        [token.lemma_ if not token.pos_ == "PRON" else token.text for token in doc])
    doc = nlp(row["lemmatized"])
    row["stopwords_removed"] = ' '.join(
        [token.text for token in doc if not token.is_stop])
    # Normalized without
    row["normalized_text"] = doc.text
    return row


if __name__ == '__main__':
    dp_1 = DialogueParser(
        r"C:\Users\juliusdaub\PycharmProjects\sigmund\data\Paar 27_T1_IM_FW.docx",
        "DEPR", 27, "B", True)
    sentences = dp_1.get_sentences(drop_na=False)
    print(sentences.to_markdown())
