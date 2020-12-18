from os import getcwd

import spacy

from utils.corpus_manager import DialogueCorpusManager
from utils.dialogue_parser import DialogueParser

from pipelinelib.pipeline import Pipeline
from sigmund.classification import qda
from sigmund.features import words as fwords
from sigmund.preprocessing import syllables as psyllables
from sigmund.preprocessing import words as pwords
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

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

    pipeline = Pipeline(model=nlp, empty_pipeline=True) \
        .add_component(psyllables.SyllableExtractor()) \
        .add_component(pwords.WordExtractor()) \
        .add_component(fwords.LiwcScores(liwc_dict_path)) \
        .add_component(qda.QDA_ON_LIWC(X_train=x_train, y_train=y_train)) \


    dict_rows = []
    for doc in paragraphs.apply(lambda p: pipeline.execute(p.raw_text), axis=1):
        dict_row = {"len": len(doc), "inhib_score" : doc._.liwc_scores.get("Inhib", 0.0), "qda_prediction": doc._.QDA_ON_LIWC}
        dict_rows.append(dict_row)

    feature_frame = pd.DataFrame(dict_rows)
    feature_frame["ground_truth"] = paragraphs["is_depressed"]

    true_labels = np.array(feature_frame["ground_truth"])
    prediction = np.array(feature_frame["qda_prediction"])

    print(prediction.shape, true_labels.shape)
    accuracy = accuracy_score(prediction, true_labels)
    print(f"Total Accuracy {accuracy}")



    # print("\n".join(map(str, [doc._.words, doc._.word_count, doc._.liwc_scores])))
