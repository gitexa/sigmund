import string

import pandas as pd
import spacy
import pyphen


def get_word_sentence_ratio(n_sentences, n_words):
    return n_words / n_sentences


dic = pyphen.Pyphen(lang='de')

nlp = spacy.load("de_core_news_md")


def syllable_counter(text):
    doc = nlp(text)
    syllable_count = 0

    for token in doc:
        if not str(token) in string.punctuation:
            syllable_count += len(dic.inserted(str(token)).split("-"))

    return syllable_count


if __name__ == '__main__':
    df = pd.read_csv("all_preprocessed.csv")
    # df["syllables_counter"] = df["raw_text"].apply(syllable_counter)
    grouped_by_couple = df.groupby(by="couple_id").sum()
    grouped_by_depr = df.groupby(by="is_depressed").sum()
    grouped_by_gender = df.groupby(by=["gender", "is_depressed"]).sum()

    grouped_by_depr["ws_ratio"] = grouped_by_depr.apply(lambda x: get_word_sentence_ratio(x.word_count, x.sent_count),
                                                        axis=1)
    grouped_by_couple["ws_ratio"] = grouped_by_couple.apply(
        lambda x: get_word_sentence_ratio(x.word_count, x.sent_count),
        axis=1)

    grouped_by_gender["ws_ratio"] = grouped_by_gender.apply(
        lambda x: get_word_sentence_ratio(x.word_count, x.sent_count),
        axis=1)

    # word_count_total = df["raw_text"]

    # print(word_count_total)

    print(grouped_by_depr)
    print(grouped_by_couple.sort_values(by=["ws_ratio"]))
    print(grouped_by_gender)

