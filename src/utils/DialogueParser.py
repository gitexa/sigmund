import copy

import pandas as pd
import docx
import re
import string

import spacy


class DialogueParser:

    @staticmethod
    def split_speaker_text(text_line):
        pattern = re.compile(r"^(([AB])[:;] )(.*)")
        matches = pattern.match(text_line)

        if matches:
            return matches.group(2), matches.group(3)
        else:
            return None, None

    @staticmethod
    def clean_comments(text):
        # Remove Annotations
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

    def split_sentences(self, text, using_punctuation=True):
        if not using_punctuation:
            text = ''.join([symb for symb in text if not (symb.isdigit() or symb in string.punctuation)])
        doc = self.nlp(text)

        sentences = [sent.text for sent in doc.sents]

        return sentences

    def __init__(self, file, nlp, clean_comments=True) -> None:
        self.document = docx.Document(file)
        self.nlp = nlp
        dictionary_rows = []
        sentence_dict_row = []
        for paragraph_position, paragraph in enumerate(list(self.document.paragraphs)[3:]):
            dictionary_row = {}
            dictionary_row["paragraph_position"] = paragraph_position
            speaker, text = self.split_speaker_text(paragraph.text)

            if clean_comments:
                text = DialogueParser.clean_comments(text)

            if speaker:
                dictionary_row["speaker"] = speaker
                dictionary_row["raw_text"] = text
                dictionary_row["is_paragraph"] = True
                sentences = self.split_sentences(text, True)
                for index, sentence in enumerate(sentences):
                    sentence_row = copy.copy(dictionary_row)
                    sentence_row["raw_text"] = sentence
                    sentence_row["sentence_index"] = index
                    sentence_dict_row.append(sentence_row)
            else:
                dictionary_row["raw_text"] = paragraph.text
                dictionary_row["is_paragraph"] = False
            dictionary_rows.append(dictionary_row)



        self.paragraph_frame = pd.DataFrame(dictionary_rows)
        self.sentence_frame = pd.DataFrame(sentence_dict_row)
        super().__init__()

        self.df_paragraphs = pd.DataFrame()

    def get_sentences(self, speaker=None):
        return self.sentence_frame
        pass

    def get_paragraphs(self, speaker=None):
        return self.paragraph_frame
        pass

    def get_fulltext(self, split_speakers=True):

        if split_speakers:
            return self.paragraph_frame.groupby(by="speaker").agg({"raw_text": ''.join})
        else:
            return self.paragraph_frame["raw_text"].str.cat(sep=" ")


if __name__ == '__main__':
    dp = DialogueParser(r"../data/Paar 105_T1_IM_FW.docx", spacy.load("de_core_news_md"), clean_comments=True)
    sentences = dp.get_sentences()
    fulltext = dp.get_fulltext(False)
    pass
