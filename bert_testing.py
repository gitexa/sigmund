import gc
import os
import pickle
import re

import pandas as pd

from src.utils.germansentiment_modified import SentimentModel

# load preprocessed paragraphs
path_preprocessed = 'data/all_preprocessed_new_v3.csv'
df_preprocessed = pd.read_csv(path_preprocessed)
df_paragraphs = df_preprocessed.copy()

# get not normalized text


def get_lower_case(text):
    return str.lower(text)


def get_without_annotations(text):
    removed_annotations = re.sub(r"\(.*\)", "", text)
    removed_annotations = ' '.join(removed_annotations.split())
    return removed_annotations


df_paragraphs.rename(columns={'text': 'raw_text'}, inplace=True)
df_paragraphs['raw_text_without_annotations'] = df_paragraphs['raw_text'].apply(
    lambda row: get_without_annotations(get_lower_case(row)))

bert_predictions = dict()
all_cids = sorted(df_paragraphs.couple_id.unique())

print('Starting predictions!')

for c_id in all_cids:

    print(c_id)

    # get chunk
    chunk = df_paragraphs.loc[df_paragraphs['couple_id']
                              == c_id].raw_text_without_annotations

    # get sentiment
    model = SentimentModel()

    # result = model.predict_sentiment(['ich werde mich trotzdem nicht freuen können wegen.'])
    result = model.predict_sentiment(chunk.to_list())
    bert_predictions[c_id] = result

    # free memory
    print(result)
    del model
    del result
    del chunk
    gc.collect()

    print('---------')

with open(os.path.join(os.path.abspath(os.getcwd()), 'bert_predictions.pkl'), 'wb') as file:
    pickle.dump(bert_predictions, file)
