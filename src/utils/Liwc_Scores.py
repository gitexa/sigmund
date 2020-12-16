import re
from collections import defaultdict, Counter

import spacy


class LiwcFeatures:
    def __init__(self, dict_file) -> None:
        liwc_dict = open(dict_file, "r")
        text_lines = liwc_dict.read().split("\n")
        text = text_lines[70:]

        self.kinds = text_lines[1:69]
        self.kinds = [line.split("\t") for line in self.kinds]
        self.pydict_2 = {cat[0]: cat[1] for cat in self.kinds}

        self.cats = [line.split("\t") for line in text]
        self.pydict = {cat[0]: cat[1:] for cat in self.cats}

        super().__init__()

    def get_scores(self, sentence):
        lower = str.lower(sentence)
        word_count = len(lower.split(" "))
        counts = defaultdict(int)
        for word in lower.split(" "):
            print("word", word)
            dict_entry = self.pydict.get(word)
            if dict_entry:
                for key in dict_entry:
                    counts[self.pydict_2[key]] += (1 / word_count)
            else:
                entries = []
                for dict_entry in self.pydict.keys():
                    pattern = dict_entry.replace("*", ".*")
                    match = re.match(f"{pattern}", word)
                    if match:
                        print(match.group())
                        entries.append(match.group(0))

        return dict(counts)


from nltk.stem.cistem import Cistem

import liwc


def tokenize(text):
    # you may want to use a smarter tokenizer
    for match in re.finditer(r'\w+', text, re.UNICODE):
        yield match.group(0)


if __name__ == '__main__':
    # lf = LiwcFeatures("../../data/German_LIWC2001_Dictionary.dic")
    # nlp = spacy.load('de_core_news_md')

    sentence = """ Da weiß ich jetzt gar nicht was ich darauf sein, (beide lachen) weil ich mir das auch sehr wünschen würde von dir, dass du offen bist. Davon hängt es auch ab ob ich nachher noch mit essen gehe oder nicht. Ja, Thema Ehrlichkeit.  Was ich bei dir schön finde, wenn es dir gut geht, dann bist du zärtlich, dann bist du, das mag ich einfach, das finde ich positiv. Aber es wird immer durch den Scheiß unterbrochen, das ist jetzt nicht positiv was ich sage, aber das kann ich mir nicht aus den Fingern saugen. Aber das positive an dir ist, dass du Phasen hast die werden allerdings immer schneller durchbrochen im Moment, wo du, gestern zum Beispiel, gut drauf warst, und dann denke ich „Boah, ist das alles schön!“. Dann bist du humorvoll, bist nett, bist zugewandt, und ja dann freue ich mich. Und wenn du irgendeinen Mist gebaut hast, dann kriegst du garantiert auch heute Nacht, wie auch immer, oder Mittag, gepennt, dann könntest du ja auch diese Offenheit beibehalten aber toppst du das noch in dem du sagt „ah und du“. So, jetzt bist du mal wieder dran, ist genug."""
    # print(lf.get_scores(sentence=sentence))
    gettysburg = sentence

    gettysburg_tokens = tokenize(gettysburg)
    parse, category_names = liwc.load_token_parser("../../data/German_LIWC2001_Dictionary.dic")
    # now flatmap over all the categories in all of the tokens using a generator:
    gettysburg_counts = Counter(category for token in gettysburg_tokens for category in parse(token))
    # and print the results:
    print(gettysburg_counts.most_common())
