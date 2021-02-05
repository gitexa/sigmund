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
from sigmund.features import flesch_reading_ease as fflesch
from sigmund.features import talk_turn as ftalkturn
from sigmund.features import words as fwords
from sigmund.preprocessing import syllables as psyllables
from sigmund.preprocessing import words as pwords

if __name__ == "__main__":
    nlp = spacy.load("de_core_news_sm", disable=["ner", "parser"])

    #path2file = r"/home/rise/Schreibtisch/Sigmund/Paardialog_text/Paar 47_T1_IM_FW.docx"
    path2file = r"./data/texts/Paar 81_T1_IM_FW.docx"
    liwc_dict_path = r"/data/German_LIWC2001_Dictionary.dic"

    #parse, category_names = liwc.load_token_parser(liwc_dict_path)

    dialogue = DialogueParser(
        file=path2file, nlp=nlp, clean_comments=True)

    paragraphs = dialogue.get_sentences()
    fulltext = dialogue.get_fulltext()

    pipeline_fulltext = Pipeline(model=nlp) \
        .add_component(fflesch.FleschExtractor())

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
