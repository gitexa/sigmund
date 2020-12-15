"""
app.py
========

The core module of this project, serves as *frontend*
"""
import spacy
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

nlp = spacy.load("de_core_news_md")

from utils.dialogue_parser import DialogueParser, preprocess
from docx import Document

from utils.feature_annotator import add_features, reading_ease_german

st.markdown("# Welcome to Sigmund")

uploaded_docx = st.file_uploader("Upload docx File here!")


def f(words, sentences):
    return words / sentences


if uploaded_docx:
    st.write("A file was uploaded!")
    group = st.text_input("Gruppe")
    is_depressed = st.checkbox("Depressiv?")
    female_label = st.selectbox("Sprecher W", ["A", "B"])
    couple_id = st.text_input("ID")
    dp = DialogueParser(uploaded_docx, group, couple_id=couple_id, female_label=female_label, depressed=is_depressed)
    preprocessed_df = dp.get_paragraphs().apply(preprocess, axis=1)
    st.write(dp.get_paragraphs())
    st.write(preprocessed_df)
    # add_features(parsed_dialogue)



    # concat_per_speaker = parsed_dialogue.groupby(['Speaker'], as_index=False).agg(
    #     {'Text': ' '.join, "Sentences": "sum", "word_count": "sum"})
    # concat_per_speaker["FleshScore"] = concat_per_speaker["Text"].apply(reading_ease_german)
    # concat_per_speaker['Avg Sentence Length'] = concat_per_speaker[['word_count', 'Sentences']].apply(lambda x: f(*x),
    #                                                                  axis=1)
    # st.write(concat_per_speaker)

    # corpus = concat_per_speaker["Text"]
    # vectorizer = TfidfVectorizer()
    # vectorizer.fit(corpus)
    #
    # x = vectorizer.transform(corpus)
    # st.write(vectorizer.transform(concat_per_speaker[concat_per_speaker["Speaker"] == "A: "]["Text"]).toarray())
    # st.write(vectorizer.transform(concat_per_speaker[concat_per_speaker["Speaker"] == "B: "]["Text"]).toarray())
    # st.write(x.toarray())

    # speaker_b_tfidf = vectorizer.transform(concat_per_speaker[concat_per_speaker["Speaker"] == "B: "])


    # st.write(speaker_b_tfidf.toarray().shape)

    # st.markdown("## Metrics")
    #
    # a_share = st.write("Speaker A Share")
    # st.progress(shares["A: "] / shares.sum())
    # st.write("Speaker B Share")
    # st.progress(shares["B: "] / shares.sum())
    #
    # st.markdown("## Dialogue as Data-Frame")
    # st.write(parsed_dialogue)
    # st.write(len(parsed_dialogue))
    # st.write(sent_df)
    # st.write(len(sent_df))
