import os

import matplotlib.pyplot as plt
import spacy
import streamlit as st
from PIL import Image

from src.pipelinelib.pipeline import Pipeline
from src.pipelinelib.querying import Parser, Queryable
from src.pipelinelib.text_body import TextBody
from src.sigmund.classification.linear_discriminant_analysis import \
    LinearDiscriminantAnalysisClassifier
from src.sigmund.classification.merger import FeatureMerger
from src.sigmund.classification.naive_bayes import NaiveBayes
from src.sigmund.classification.pca import PCAReduction
from src.sigmund.extensions import *
from src.sigmund.features.basic_statistics import BasicStatistics
from src.sigmund.features.liwc import Liwc
from src.sigmund.features.pos import PartOfSpeech
from src.sigmund.features.tfidf import FeatureTFIDF
from src.sigmund.features.vocabulary_size import VocabularySize
from src.sigmund.preprocessing.words import Lemmatizer, Stemmer, Tokenizer

st.title('Welcome to Sigmund!')

folder = os.path.join(os.getcwd(), "data", "transcripts")
files = [os.path.join(root, f) for root, _, files in os.walk(folder)
         for f in files if f.endswith(".docx")]
nlp = spacy.load("de_core_news_sm", disable=["ner", "parser"])

# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')

parser = Parser(
    nlp=nlp, metadata_path="./data/transcripts/Kopie von Transkriptionspaare_Daten.xls")
parser.read_from_files(files)
queryable = Queryable.from_parser(parser)

# Notify the reader that the data was successfully loaded.
data_load_state.text('Loading data...done!')

# Show DataFrame
st.subheader('Raw data')
document_df = queryable.execute(level=TextBody.DOCUMENT)
st.write(document_df)

# Execute Pipeline
pipeline_execute_state = st.text('Executing pipeline ...')


pipeline = Pipeline(queryable=queryable)
pipeline.add_components([Tokenizer(), Stemmer(), Lemmatizer()])
pipeline.add_component(FeatureTFIDF(white_list=[
    'ja', 'auch', 'wenn', 'also', 'werden', 'schon', 'wir',  # high in depressed group
    'und', 'haben', 'du', 'sehr'])),  # 'so', 'wirkl  ich', 'ich', 'gerne', 'weil']))
pipeline.add_component(
    NaiveBayes(
        inputs=[TFIDF_DOCUMENT_MF],
        output=CLASSIFICATION_NAIVE_BAYES_TFIDF, voting=False))

pipeline.add_component(Liwc(white_list=[
    'Posemo', 'Past', 'Present', 'Future', 'Metaph',
    'Death', 'Affect', 'Incl', 'Achieve'
]))
pipeline.add_component(
    NaiveBayes(
        inputs=[LIWC_DOCUMENT_MF],
        output=CLASSIFICATION_NAIVE_BAYES_LIWC, voting=False))

pipeline.add_component(PartOfSpeech(white_list=["ADV", "PPER", "ADJD", "VAFIN", "KON"]))
pipeline.add_component(
    NaiveBayes(
        inputs=[POS_DOCUMENT_MF],
        output=CLASSIFICATION_NAIVE_BAYES_POS, voting=False))

pipeline.add_component(NaiveBayes(inputs=[
    CLASSIFICATION_NAIVE_BAYES_TFIDF,
    CLASSIFICATION_NAIVE_BAYES_LIWC,
    CLASSIFICATION_NAIVE_BAYES_POS,
], output=CLASSIFICATION_NAIVE_BAYES_VOTING, voting=True))

storage, plots = pipeline.execute(visualise=True)

features = [(ext, plot) for ext, plot in plots.items()
            if ext.kind == ExtensionKind.FEATURE]
classifications = [(ext, plot) for ext, plot in plots.items()
                   if ext.kind == ExtensionKind.CLASSIFIER]

st.write("Found", len(features), "features")
st.write("Found", len(classifications), "classifications")

# images = [
#     Image.open(os.path.join(pipeline._plot_output, ext.filename() + ".png"))
#     for ext, _ in features
# ]
# captions = [ext.name for ext in plots]

# st.image(images, caption=captions, width=None, clear_figure=False)


images = [
    Image.open(os.path.join(pipeline._plot_output, ext.filename() + ".png"))
    for ext, _ in classifications
]
captions = [ext.name for ext in classifications]

st.image(images, caption=captions, width=None, clear_figure=False)
