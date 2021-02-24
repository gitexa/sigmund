"""
app.py
========

The core module of this project, serves as *frontend*
"""
import spacy
import streamlit as st
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from src.backwards_compatibility.dialogue_parser import DialogueParser, preprocess


nlp = spacy.load("de_core_news_md")

st.markdown("# Welcome to Sigmund")

uploaded_docx = st.file_uploader("Upload docx File here!")


def tf_idf_svd(corpus, components):
    svd = TruncatedSVD(n_components=components, algorithm="arpack")
    vectorizer = TfidfVectorizer()
    vectorizer.fit(corpus)
    x = vectorizer.transform(corpus).toarray()
    svd.fit(x)
    print(svd.singular_values_)


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
    st.markdown("## Before Preprocessing")
    st.write(dp.get_paragraphs())
    st.markdown("## After Preprocessing")
    st.write(preprocessed_df)

    corpus = preprocessed_df["stopwords_removed"]

    tf_idf_svd(corpus, 10)
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
