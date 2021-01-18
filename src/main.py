import json
import os
from os import getcwd

import liwc
import numpy as np
import pandas as pd
import spacy
from utils.corpus_manager import DialogueCorpusManager
from utils.dialogue_parser import DialogueParser

from pipelinelib.pipeline import Pipeline
from pipelinelib.text_body import TextBody
from sigmund.classification import qda
from sigmund.features import agreement_score as fagree
from sigmund.features import talk_turn as ftalkturn
from sigmund.features import words as fwords
from sigmund.preprocessing import syllables as psyllables
from sigmund.preprocessing import words as pwords

<< << << < HEAD

== == == =

>>>>>> > 854a22e(Implementation of agreement_score now delivers final score per diolag using liwc categorys)


if __name__ == "__main__":
    nlp = spacy.load("de_core_news_sm", disable=["ner", "parser"])

    #path2file = r"/home/rise/Schreibtisch/Sigmund/Paardialog_text/Paar 47_T1_IM_FW.docx"
    path2file = r"/home/rise/Schreibtisch/Sigmund/Paardialog_text/Paar 81_T1_IM_FW.docx"
    liwc_dict_path = r"/home/rise/Schreibtisch/Sigmund_git/sigmund/data/German_LIWC2001_Dictionary.dic"

    #parse, category_names = liwc.load_token_parser(liwc_dict_path)

    dialogue = DialogueParser(
        doc_file=path2file, group="DEPR", couple_id=105, female_label="B",
        depressed=True, remove_annotations=True)
    # path2file = "/home/benji/Documents/Uni/heidelberg/01/text-analytics/sigmund/src/data/Paargespräche_text/Paar 47_T1_IM_FW.docx"
    # with open(os.path.join(os.getcwd(), 'config.json'), 'r') as configfile:
    #    config = json.load(configfile)
    #path2file = os.path.join(config['path_to_transcripts'], 'Paar 47_T1_IM_FW.docx'),

    # dialogue = DialogueParser(
    # doc_file=path2file, group="DEPR", couple_id=105, female_label="B",
    # depressed=True, remove_annotations=True)

    corpus_file_path = 'all_preprocessed.csv'
    full_dataset = DialogueCorpusManager(corpus_file=corpus_file_path, nlp=nlp)
    ds = full_dataset.get_paragraphs()

    paragraphs = dialogue.get_paragraphs()
    fulltext = dialogue.get_all_paragraphs()

    y_train = ds["is_depressed"]
    for index, doc in enumerate(
        ds.apply(lambda p: pipeline.execute(p.raw_text, body=TextBody.DOCUMENT),
                 axis=1)):
        x_train[index, :] = [doc._.liwc_scores.get("Posemo"), doc._.liwc_scores.get(
            "Negemo"), doc._.liwc_scores.get("Inhib")]

    x_train = np.nan_to_num(x_train, copy=False)

    pipeline = Pipeline(model=nlp, empty_pipeline=True) \
        .add_component(psyllables.SyllableExtractor()) \
        .add_component(pwords.Tokenizer()) \
        .add_component(pwords.StemmedAndLemmatized()) \
        .add_component(fwords.LiwcScores("./data/German_LIWC2001_Dictionary.dic")) \
        .add_component(qda.QDA_ON_LIWC(X_train=x_train, y_train=y_train))

    for doc in paragraphs.apply(
            lambda p: pipeline.execute(p.raw_text, body=TextBody.DOCUMENT),
            axis=1):
        print(len(doc), doc._.liwc_scores.get("Inhib", 0.0),
              f"qda_prediction: {doc._.QDA_ON_LIWC}")
    # print("\n".join(map(str, [doc._.words, doc._.word_count, doc._.liwc_scores])))
