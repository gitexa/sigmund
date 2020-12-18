from os import getcwd

import spacy
from utils.dialogue_parser import DialogueParser

from pipelinelib.pipeline import Pipeline
from sigmund.classification import qda
from sigmund.features import words as fwords
from sigmund.preprocessing import syllables as psyllables
from sigmund.preprocessing import words as pwords

if __name__ == "__main__":
    nlp = spacy.load("de_core_news_sm", disable=["ner", "parser"])
    sentence = "Ich habe Hunger und bin glücklich aber auch traurig und ein Baum"

    path2file = "/home/benji/Documents/Uni/heidelberg/01/text-analytics/sigmund/src/data/Paargespräche_text/Paar 47_T1_IM_FW.docx"
    dialogue = DialogueParser(
        doc_file=path2file, group="DEPR", couple_id=105, female_label="B",
        depressed=True, remove_annotations=True)

    paragraphs = dialogue.get_fulltext()
    # print(paragraphs.to_markdown())

    pipeline = Pipeline(model=nlp) \
        .add_component(psyllables.SyllableExtractor()) \
        .add_component(pwords.WordExtractor()) \
        .add_component(fwords.LiwcScores("./data/German_LIWC2001_Dictionary.dic")) \
        # .add_component(qda.QDA_ON_LIWC())

    for doc in paragraphs.apply(lambda p: pipeline.execute(p.raw_text), axis=1):
        print(len(doc), doc._.liwc_scores.get("Inhib", 0.0))

    # print("\n".join(map(str, [doc._.words, doc._.word_count, doc._.liwc_scores])))
