import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

from utils.dialogue_parser import DialogueParser, preprocess
import pandas as pd

if __name__ == '__main__':
    nlp = spacy.load("de_core_news_md")
    init_corpus = False

    if init_corpus:
        # Paths must be adjust according to the location of the files!
        parsers = [
            DialogueParser(r"C:\Users\juliusdaub\PycharmProjects\sigmund\data\Paar 27_T1_IM_FW.docx", "DEPR", 27, "B",
                           True),
            DialogueParser(r"C:\Users\juliusdaub\PycharmProjects\sigmund\data\Paar 29_T1_IM_FW.docx", "DEPR", 29, "B",
                           False),
            DialogueParser(r"C:\Users\juliusdaub\PycharmProjects\sigmund\data\Paar 47_T1_IM_FW.docx", "DEPR", 47, "B",
                           True),
            DialogueParser(r"C:\Users\juliusdaub\PycharmProjects\sigmund\data\Paar 58_T1_IM_FW.docx", "DEPR", 58, "B",
                           False),
            DialogueParser(r"C:\Users\juliusdaub\PycharmProjects\sigmund\data\Paar 60_T1_IM_FW.docx", "DEPR", 60, "A",
                           True),
            DialogueParser(r"C:\Users\juliusdaub\PycharmProjects\sigmund\data\Paar 81_T1_IM_FW.docx", "DEPR", 81, "A",
                           False),
            DialogueParser(r"C:\Users\juliusdaub\PycharmProjects\sigmund\data\Paar 87_T1_IM_FW.docx", "DEPR", 87, "B",
                           False),
            DialogueParser(r"C:\Users\juliusdaub\PycharmProjects\sigmund\data\Paar 105_T1_IM_FW.docx", "DEPR", 105, "B",
                           True),
            DialogueParser(r"C:\Users\juliusdaub\PycharmProjects\sigmund\data\Paar 138_T1_IM_FW.docx", "DEPR", 138, "A",
                           True),
            DialogueParser(r"C:\Users\juliusdaub\PycharmProjects\sigmund\data\Paar 182_T1_IM_FW.docx", "DEPR", 182, "A",
                           False)
        ]

        preprocessed = [parser.get_paragraphs().apply(preprocess, axis=1) for parser in parsers]
        all_frames = pd.concat(preprocessed)
        all_frames.to_csv("all_preprocessed.csv")

    else:
        all_frames = pd.read_csv("all_preprocessed.csv")

    corpus = all_frames["stopwords_removed"]
    corpus = corpus.dropna()

    vectorizer = TfidfVectorizer()
    vectorizer.fit(corpus)
    x = vectorizer.transform(corpus).toarray()
    print(x.shape)
