from copy import copy

import pandas as pd
import spacy


class DialogueCorpusManager:
    def __init__(self, corpus_file, build_sentences=False) -> None:
        self.paragraph_frame = pd.read_csv(corpus_file)
        self.paragraph_frame.dropna(inplace=True)
        self.paragraph_frame.drop(["normalized_text"], axis=1, inplace=True)
        self.sentence_frame = pd.DataFrame()
        self.sentences_ready = build_sentences

        if build_sentences:
            self.build_sentences()

        super().__init__()

    def build_sentences(self):
        sentence_rows = []
        for row_dict in self.paragraph_frame.to_dict(orient="records"):
            doc = nlp(row_dict["raw_text"])
            for index, sentence in enumerate(doc.sents):
                row = copy(row_dict)
                row["sent_pos"] = index
                row["raw_text"] = sentence.text
                sentence_rows.append(row)

        self.sentence_frame = pd.DataFrame(sentence_rows)
        self.sentence_frame.dropna(inplace=True)
        self.sentences_ready = True

    def get_random_sample(self):
        pass

    def get_depressive_sentences(self):
        assert self.sentences_ready, "Sentence Data-Frame has not been loaded yet"
        return self.sentence_frame.query("is_depressed")
        pass

    def get_depressive_paragraphs(self):
        return self.paragraph_frame.query("is_depressed")
        pass

    def get_couple_paragraphs(self, couple_id):
        return self.paragraph_frame.query(f"couple_id == {couple_id}")

    def get_couple_sentences(self, couple_id):
        assert self.sentences_ready, "Sentence Data-Frame has not been loaded yet"
        return self.sentence_frame.query(f"couple_id == {couple_id}")

    def get_sentences(self):
        assert self.sentences_ready, "Sentence Data-Frame has not been loaded yet"
        return self.sentence_frame

if __name__ == '__main__':
    nlp = spacy.load("de_core_news_sm")
    DCM = DialogueCorpusManager("main/all_preprocessed.csv")
    DCM.build_sentences()
    print(DCM.get_couple_sentences(105).to_markdown())
