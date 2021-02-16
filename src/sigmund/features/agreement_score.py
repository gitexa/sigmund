import re
import string
from collections import Counter
from itertools import filterfalse

import liwc
import pyphen
import spacy
from spacy.tokens import Doc

from src.pipelinelib.component import Component
from src.pipelinelib.extension import Extension

nlp = spacy.load("de_core_news_md")


class AgreementScoreExtractor(Component):
    """
    Extracts the Agreement-Score turn from text and stores these under doc._.agreementscore
    """
    agreementscore = Extension(name="agreementscore", default_type=list())

    def __init__(self, dictionary_path: str):
        super().__init__(name=AgreementScoreExtractor.__name__, required_extensions=list(),
                         creates_extensions=[AgreementScoreExtractor.agreementscore])
        self.dic = pyphen.Pyphen(lang='de')
        self._dictionary_path = dictionary_path

    def apply(self, doc: Doc) -> Doc:

        # Load LIWC Dictionary provided by path
        parse, category_names = liwc.load_token_parser(self._dictionary_path)
        # neg : negate(7) ,negemo(16),  23	Discrep,  45	Excl
        disagreement_cat = ['Negate', 'Negemo', 'Discrep', 'Excl']

        tmp = doc.text.split("$")
        tmp = [tmp[0].split("|"), tmp[1].split("|")]

        agr_score = [0, 0]
        agr_score[1] = len(tmp[0]) + len(tmp[1])
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
                    tokens = [
                        x.lower()
                        for x in list(
                            filterfalse(
                                string.punctuation.__contains__,
                                map(str, nlp(tmp[x][y]))))[: 5]]

                    flag = False
                    for category in disagreement_cat:
                        if Counter(category for token in tokens
                                   for category in parse(token))[category] >= 1:

                            print(x, y, category, tokens)
                            if flag == False:
                                agr_score[0] += 1
                            flag = True
                else:
                    tokens = [
                        x.lower()
                        for x in list(
                            filterfalse(
                                string.punctuation.__contains__,
                                map(str, nlp(tmp[x][y]))))]

                    flag = False
                    for category in disagreement_cat:
                        if Counter(category for token in tokens
                                   for category in parse(token))[category] >= 1:

                            print(x, y, category, tokens)
                            if flag == False:
                                agr_score[0] += 1
                            flag = True

        doc._.agreementscore = round(agr_score[0] / agr_score[1], 2)
        print(agr_score)
        return doc


def tokenize(text):
    # you may want to use a smarter tokenizer
    for match in re.finditer(r'\w+', text, re.UNICODE):
        yield match.group(0)
