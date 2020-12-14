"""
app.py
========

The core module of this project, serves as *frontend*
"""

import streamlit as st
from utils.dialogue_parser import dialogue_parser
from docx import Document

from utils.feature_annotator import add_features, reading_ease_german

st.markdown("# Welcome to Sigmund")

uploaded_docx = st.file_uploader("Upload docx File here!")

if uploaded_docx:
    st.write("A file was uploaded!")
    parsed_dialogue = dialogue_parser(uploaded_docx)
    add_features(parsed_dialogue)
    """
    Something important to note!
    """
    shares = parsed_dialogue.groupby(["Speaker"]).sum()["word_count"]
    
    concat_per_speaker = parsed_dialogue.groupby(['Speaker'], as_index = False).agg({'Text': ' '.join})
    concat_per_speaker["FleshScore"] = concat_per_speaker["Text"].apply(reading_ease_german)
    st.write(concat_per_speaker)
    # for paragraph in paragraphs:
    #     st.write(paragraph.text)

    st.markdown("## Metrics")

    a_share = st.write("Speaker A Share")
    st.progress(shares["A: "] / shares.sum())
    st.write("Speaker B Share")
    st.progress(shares["B: "] / shares.sum())

    st.markdown("## Dialogue as Data-Frame")
    st.write(parsed_dialogue)