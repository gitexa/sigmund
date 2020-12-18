from os import getcwd

import spacy
from utils.dialogue_parser import DialogueParser

from pipelinelib.pipeline import Pipeline
from sigmund.classification import qda
from sigmund.features import words as fwords
from sigmund.preprocessing import syllables as psyllables
from sigmund.preprocessing import words as pwords
import numpy as np

if __name__ == "__main__":
    nlp = spacy.load("de_core_news_sm", disable=["ner", "parser"])

    path2file = r"C:\Users\juliusdaub\PycharmProjects\sigmund\data\Paar 47_T1_IM_FW.docx"
    dialogue = DialogueParser(
        doc_file=path2file, group="DEPR", couple_id=105, female_label="B",
        depressed=True, remove_annotations=True)

    paragraphs = dialogue.get_paragraphs()
    # For this example we use a randomly-generated Training-Set for QDA-Prediction, Later on this will be
    # Done on real Data
    train_set_length = 100
    np.random.seed(0)
    x_train = np.random.uniform(0,0.2,[100,3])
    y_train = np.random.randint(2, size=train_set_length)


    pipeline = Pipeline(model=nlp) \
        .add_component(psyllables.SyllableExtractor()) \
        .add_component(pwords.WordExtractor()) \
        .add_component(fwords.LiwcScores("../data/German_LIWC2001_Dictionary.dic")) \
        .add_component(qda.QDA_ON_LIWC(X_train=x_train, y_train=y_train))

    for doc in paragraphs.apply(lambda p: pipeline.execute(p.raw_text), axis=1):
        print(len(doc), doc._.liwc_scores.get("Inhib", 0.0),f"qda_prediction: {doc._.QDA_ON_LIWC}")
    # print("\n".join(map(str, [doc._.words, doc._.word_count, doc._.liwc_scores])))
