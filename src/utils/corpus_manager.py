from copy import copy

import pandas as pd
import spacy


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
       