import string

import spacy
from sklearn.pipeline import Pipeline

nlp = spacy.load("de_core_news_sm")
nlp.add_pipe(nlp.create_pipe('sentencizer'))

import pandas as pd
import numpy as np

# st.title("Hello World")

import docx
import re


# stopwords = nltk.corpus.stopwords.words('german')

def is_dialog_line(line):
    pattern = re.compile(r"^([AB]: )(.*)")
    return pattern.match(line)


def preprocessing(text, lemmatize=True, lowercase=True, remove_stopwords=True, remove_punctuation=True):
    text = text.lower()
    doc = nlp(text)

    tokens = [token for token in doc if not token.text in string.punctuation]
    for token in tokens:
        print(token.pos_)
    pass


def dialogue_parser(file_object):
    doc = docx.Document(file_object)
    file_name = file_object.name
    couple = file_object.name
    listdict = []
    for index, para in enumerate(doc.paragraphs[3:]):
        dialog_line = is_dialog_line(para.text)
        if dialog_line:
            text = dialog_line.group(2).lower()
            doc = nlp(text)
            sent_count = len(list(doc.sents))
            tokens = " ".join([x.lemma_ for x in doc if x.lemma_])
            word_count = len(tokens.split(" "))
            # sentiments = []
            # for token in nlp(text):
            #     sentiments.append(token._.sentiws)
            dictrow = {
                "Couple": couple,
                "Pos in doc": index,
                "Speaker": dialog_line.group(1),
                "Sentences": sent_count,
                "tokens": tokens,
                "Text": text,
                # "sentiments": sentiments,
                "word_count": word_count
            }

            listdict.append(dictrow)

    df = pd.DataFrame(data=listdict)
    return df

if __name__ == '__main__':
    preprocessing("Hallo wir du welt!")
