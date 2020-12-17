import json
import os

import matplotlib as plt
import pandas as pd
import spacy
from dialogue_parser import DialogueParser, preprocess
from sklearn.feature_extraction.text import TfidfVectorizer

if __name__ == '__main__':
    nlp = spacy.load("de_core_news_md")
    init_corpus = True

    with open(os.path.join(os.getcwd(), 'config.json'), 'r') as configfile:
        config = json.load(configfile)

    if init_corpus:
        # @todo Paths must be adjust according to the location of the files!

        parsers = []
        for transcript in config['transcrips']:
            parsers.append(DialogueParser(os.path.join(config['path_to_transcripts'], transcript['transcript_id']), transcript['group'], transcript['couple_id'], transcript['female_label'], transcript['depression']))
        
        preprocessed = [parser.get_paragraphs().apply(preprocess, axis=1)
                        for parser in parsers]
        all_frames = pd.concat(preprocessed)
        all_frames.to_csv("all_preprocessed.csv")

    else:
        all_frames = pd.read_csv("all_preprocessed.csv")

    corpus = all_frames["stopwords_removed"]
    corpus = corpus.dropna()

    vectorizer = TfidfVectorizer()
    vectorizer.fit(corpus)
    x = vectorizer.transform(corpus).toarray()

    df = pd.DataFrame(x, columns=vectorizer.get_feature_names())
    # df.to_excel("tf_idf.xlsx")

    # df_tfidf = pd.DataFrame(x.T.todense(), index=vectorizer.get_feature_names())
    # df_tfidf['mean'] = df_tfidf.mean(axis=1)
    # df_tfidf = df_tfidf.sort_values('mean', ascending=False)
    #
    # fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(30, 7))
    # titles = ['TFIDF - Homer Simpson\'s most important words', 'TFIDF - Marge Simpson\'s most important words',
    #           'TFIDF - Bart Simpson\'s most important words', 'TFIDF - Lisa Simpson\'s most important words']
    # for i in range(len(tfidf_lines_list)):
    #     tfidf_lines_list[i].head(10).plot(ax=axes[i], kind='bar', title=titles[i], xlabel='words',
    #                                       ylabel='mean tfidf weight over all ligns')
