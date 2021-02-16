import string
from itertools import filterfalse
from typing import Dict

import pandas as pd
import pyphen
import spacy
from spacy.tokens import Doc

from src.pipelinelib.component import Component
from src.pipelinelib.extension import Extension
from src.pipelinelib.querying import Queryable
from src.pipelinelib.text_body import TextBody

nlp = spacy.load("de_core_news_md")


class TalkTurnExtractor(Component):
    """
    Extracts Talk turn from text and stores these under doc._.talkturn
    """
    talkturn = Extension(name="talkturn")

    def __init__(self):
        super().__init__(name=TalkTurnExtractor.__name__, required_extensions=[],
                         creates_extensions=[TalkTurnExtractor.talkturn])
        self.dic = pyphen.Pyphen(lang='de')

    # def apply(self, doc: Doc) -> Doc:
    def apply(self, storage: Dict[Extension, pd.DataFrame],
              queryable: Queryable) -> Dict[Extension, pd.DataFrame]:

        docs = queryable.execute(level=TextBody.DOCUMENT)
        display(docs)
        #parapgraphs = pd.DataFrame([{'raw_text':doc.text}])

        #parapgraphs['raw_text'] = parapgraphs['raw_text'].split('$')
        #parapgraphs = parapgraphs.set_index().apply(pd.Series).stack()

        # pd.concat(row['raw_text'].split('$')
        #                                  for _, row in parapgraphs.iterrows()).reset_index()
        # print(parapgraphs)
        #tokens = map(str, doc)
        #print(list(filterfalse(string.punctuation.__contains__, tokens)))
        #print(' = ', doc.text)
        tmp = doc.text.split("$")
        tmp = [tmp[0].split("|"), tmp[1].split("|")]

        # doc = doc.concat([Series( row['raw_text'].split('|'))
        #                  for _, row in a.iterrows()]).reset_index()

        #print(' == ', len(tmp[0]), ', ', len(tmp[1]))

        talkturns = [0, 0]
        for x in range(len(tmp)):
            for y in range(len(tmp[x])):
                # nlp(tmp[x][y])
                #map(str, nlp(tmp[x][y]))
                #print(list(filterfalse(string.punctuation.__contains__, map(str, nlp(tmp[x][y])))))
                #print(len(list(filterfalse(string.punctuation.__contains__, map(str, nlp(tmp[x][y]))))))

                if len(
                    list(
                        filterfalse(
                            string.punctuation.__contains__,
                            map(str, nlp(tmp[x][y]))))) > 5:
                    talkturns[x] += 1

        doc._.talkturn = round(talkturns[0] / (talkturns[0]+talkturns[1]), 2)
        return doc
