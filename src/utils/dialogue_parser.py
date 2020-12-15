import string
from copy import copy

import pandas
import spacy
from sklearn.pipeline import Pipeline

nlp = spacy.load("de_core_news_md")
nlp.add_pipe(nlp.create_pipe('sentencizer'))

import pandas as pd
import numpy as np

import docx
import re


class DialogueParser:
    @staticmethod
    def split_speaker_text(text_line):
        pattern = re.compile(r"^(([AB]): )(.*)")
        matches = pattern.match(text_line)
        if matches:
            return matches.group(2), matches.group(3)
        else:
            return None, None

    def __init__(self, doc_file, group, couple_id, female_label="B", depressed=None):

        self.female_label = female_label
        self.group = group
        self.couple_id = couple_id
        self.docx = docx.Document(doc_file)
        self.is_depressed = depressed
        self.id = id

        rows = []
        for p in self.docx.paragraphs[3:]:
            speaker, text = self.split_speaker_text(p.text)
            gender = "W" if speaker == self.female_label else "M"
            depression_label = (gender == "W" and self.is_depressed)
            if speaker:
                row = {
                    "speaker": speaker,
                    "raw_text": text,
                    "group": self.group,
                    "gender": gender,
                    "is_depressed": depression_label,
                    "couple_id": self.couple_id,
                }
                rows.append(row)
            pass

        self.paragraph_df = pd.DataFrame(rows)

        super().__init__()

    def preprocess(self):
        pass

    def get_sentences(self):
        pass

    def get_paragraphs(self):
        return self.paragraph_df

    def get_fulltext(self):
        pass


def preprocess(row):
    doc = nlp(str.lower(row["raw_text"]))
    row["sent_count"] = len(list(doc.sents))
    doc = nlp(' '.join([token.text for token in doc if not token.is_punct]))
    row["word_count"] = len(list(doc))
    row["lemmatized"] = ' '.join([token.lemma_ if not token.pos_ == "PRON" else token.text for token in doc])
    doc = nlp(row["lemmatized"])
    row["stopwords_removed"] = ' '.join([token.text for token in doc if not token.is_stop])
    row["normalized_text"] = doc.text
    return row

