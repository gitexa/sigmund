from os import getcwd

import numpy as np
import spacy

from utils.corpus_manager import DialogueCorpusManager
from utils.dialogue_parser import DialogueParser

from pipelinelib.pipeline import Pipeline
from sigmund.classification import qda
from sigmund.features import words as fwords
from sigmund.preprocessing import syllables as psyllables
from sigmund.preprocessing import words as pwords

if __name__ == "__main__":
    nlp = spacy.load("de_core_news_sm", disable=["ner", "parser"])

    path2file = r"C:\Users\juliusdaub\PycharmProjects\sigmund\data\Paar 47_T1_IM_FW.docx"
    liwc_dict_path = r"../data/German_LIWC2001_Dictionary.dic"
    dialogue = DialogueParser(
        doc_file=path2file, group="DEPR", couple_id=105, female_label="B",
        depressed=True, remove_annotations=True)

    paragraphs = dialogue.get_paragraphs()
    # For this example we use a randomly-generated Training-Set for QDA-Prediction, Later on this will be
    # Done on real Data

    corpus_file_path = r"C:\Users\juliusdaub\PycharmProjects\sigmund\data\all_preprocessed.csv"
    full_dataset = DialogueCorpusManager(corpus_file=corpus_file_path, nlp=nlp)
    ds = full_dataset.get_paragraphs()

    pipeline = Pipeline(model=nlp) \
        .add_component(psyllables.SyllableExtractor()) \
        .add_component(pwords.WordExtractor()) \
        .add_component(fwords.LiwcScores(liwc_dict_path))

    x_train = np.zeros((len(ds), 3))

    y_train = ds["is_depressed"]
    for index, doc in enumerate(ds.apply(lambda p: pipeline.execute(p.raw_text), axis=1)):
        x_train[index, :] = [doc._.liwc_scores.get("Posemo"), doc._.liwc_scores.get("Negemo"),
                             doc._.liwc_scores.get("Inhib")]

    x_train = np.nan_to_num(x_train, copy=False)

    pipeline = Pipeline(model=nlp) \
        .add_component(psyllables.SyllableExtractor()) \
        .add_component(pwords.Tokenizer()) \
        .add_component(pwords.StemmedAndLemmatized()) \
        .add_component(fwords.LiwcScores("./data/German_LIWC2001_Dictionary.dic")) \
        .add_component(qda.QDA_ON_LIWC(X_train=x_train, y_train=y_train))

    for doc in paragraphs.apply(lambda p: pipeline.execute(p.raw_text), axis=1):
        print(len(doc), doc._.liwc_scores.get("Inhib", 0.0),
              f"qda_prediction: {doc._.QDA_ON_LIWC}")
    # print("\n".join(map(str, [doc._.words, doc._.word_count, doc._.liwc_scores])))
