import re
import string
from collections import Counter
from copy import copy

import liwc
import pandas as pd
import pyphen as pyphen
import spacy


def text_cleaner(text):
    # Remove Annotations
    text = re.sub(r"\(.*\)", "", text)
    text = re.sub(r"\[.*\]", "", text)
    # Remove Digits and Symbols
    text = ''.join([symb for symb in text if not (symb.isdigit() or symb in string.punctuation)])
    # Remove whitespace
    text = re.sub(' +', ' ', text)

    return text.lower()


class DialogueCorpusManager:
    def __init__(self, corpus_file, nlp: callable, build_sentences=False, ) -> None:
        self.paragraph_frame = pd.read_csv(corpus_file)
        self.paragraph_frame.dropna(inplace=True)
        self.paragraph_frame.drop(["normalized_text"], axis=1, inplace=True)
        self.sentence_frame = pd.DataFrame()
        self.sentences_ready = build_sentences
        self.nlp = nlp

        if build_sentences:
            self.build_sentences()

        super().__init__()

    def build_sentences(self):
        sentence_rows = []
        for row_dict in self.paragraph_frame.to_dict(orient="records"):
            doc = self.nlp(row_dict["raw_text"])
            for index, sentence in enumerate(doc.sents):
                row = copy(row_dict)
                row["sent_pos"] = index
                row["raw_text"] = sentence.text
                sentence_rows.append(row)

        self.sentence_frame = pd.DataFrame(sentence_rows)
        self.sentence_frame.dropna(inplace=True)
        self.sentences_ready = True

    def get_depressive_sentences(self):
        assert self.sentences_ready, "Sentence Data-Frame has not been loaded yet"
        return self.sentence_frame.query("is_depressed")

    def get_depressive_paragraphs(self):
        return self.paragraph_frame.query("is_depressed")

    def get_non_depressive_paragraphs(self):
        return self.paragraph_frame.query("is_depressed == False")

    def get_couple_paragraphs(self, couple_id):
        return self.paragraph_frame.query(f"couple_id == {couple_id}")

    def get_couple_sentences(self, couple_id):
        assert self.sentences_ready, "Sentence Data-Frame has not been loaded yet"
        return self.sentence_frame.query(f"couple_id == {couple_id}")

    def get_sentences(self):
        assert self.sentences_ready, "Sentence Data-Frame has not been loaded yet"
        return self.sentence_frame

    def get_paragraphs(self):
        # assert self.sentences_ready, "Sentence Data-Frame has not been loaded yet"
        return self.paragraph_frame

    def get_couple_full_text(self, couple_id):
        self.paragraph_frame.query(f"couple_id == {couple_id}")


def word_count(text):
    return len(text.split(" "))


def sent_count(text):
    doc = nlp(text)
    return len(list(doc.sents))


def syll_count(text):
    syll_count_no = 0
    for token in text.split(" "):
        if not str(token) in string.punctuation:
            for syll in dic.inserted(str(token)).split("-"):
                syll_count_no += 1
    return syll_count_no


def flesh_reading_ease(syll_count, word_count, sent_count):
    if sent_count == 0 or syll_count == 0 or word_count == 0:
        return 0
    return 180 - (word_count / sent_count) - (58.5 * syll_count / word_count)


def normalize_and_count(dataframe):
    dataframe["normalized"] = dataframe["raw_text"].apply(text_cleaner)
    dataframe["word_count"] = dataframe["normalized"].apply(word_count)
    dataframe["sent_count"] = dataframe["normalized"].apply(sent_count)
    dataframe["syll_count"] = dataframe["normalized"].apply(syll_count)
    dataframe["reading_ease"] = dataframe.apply(
            lambda x: flesh_reading_ease(x.syll_count, x.word_count, x.sent_count), axis=1)

def calc_liwc(dataframe):
    row_dict = []
    for row in dataframe.itertuples():
        tokens = row.normalized.split(" ")
        liwc_counts = Counter(category for token in tokens for category in parse(token))
        liwc_scores = {
            token: score / len(tokens) for token, score in liwc_counts.items()
        }
        row_dict.append(liwc_scores)
    df_features = pd.DataFrame(row_dict)
    return df_features

if __name__ == '__main__':
    nlp = spacy.load("de_core_news_md")
    dic = pyphen.Pyphen(lang='de')
    dcm = DialogueCorpusManager("../data/all_preprocessed.csv", nlp)
    couple_paragraphs = dcm.get_paragraphs()
    parse, cat_names =liwc.load_token_parser("../data/German_LIWC2001_Dictionary.dic")
    normalize_and_count(couple_paragraphs)

    liwcs = calc_liwc(couple_paragraphs)
    couple_paragraphs = couple_paragraphs.join(liwcs)
    print(couple_paragraphs.groupby(by="is_depressed").mean().transpose().to_markdown())
    # couple_paragraphs.groupby(by="is_depressed").mean().transpose().to_excel("export.xlsx")


    pass
