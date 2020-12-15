import string

import pyphen

dic = pyphen.Pyphen(lang='de')
import spacy

nlp = spacy.load("de_core_news_md")



def preprocess(text):
    print(string.punctuation)


def add_features(data_frame):
    data_frame["flesh_score"] = data_frame["Text"].apply(reading_ease_german)
    return data_frame

def syllable_counter(text):
    doc = nlp(text)
    syllable_count = 0
    word_count = 0
    sent_count = len(list(doc.sents))


    for token in doc:
        if not str(token) in string.punctuation:
            word_count += 1
            syllable_count += len(dic.inserted(str(token)).split("-"))

    return word_count, syllable_count, sent_count


def reading_ease_german(text):
    word_count, syllable_count, sent_count = syllable_counter(text)
    if syllable_count == 0 or word_count == 0 or sent_count == 0:
        return 0
    score = 180 - (word_count / sent_count) - (58.5 * (syllable_count / word_count))
    return score


if __name__ == '__main__':
    # print(syllable_counter("Hallo Welt. Wie geht es dir heute?"))
    print(reading_ease_german(
        "Die Wiener Sachtextformel dient zur Berechnung der Lesbarkeit deutschsprachiger Texte. Wie geht es dir denn heute?"))
