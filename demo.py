# Imports
import os

import spacy
from IPython.display import display
from src.pipelinelib.pipeline import Pipeline
from src.pipelinelib.querying import Parser, Queryable
from src.pipelinelib.text_body import TextBody
from src.sigmund.classification.linear_discriminant_analysis import \
    LinearDiscriminantAnalysisClassifier
from src.sigmund.classification.merger import FeatureMerger
from src.sigmund.classification.naive_bayes import NaiveBayes
from src.sigmund.classification.pca import PCAReduction
from src.sigmund.classification.random_forest import RandomForest
from src.sigmund.extensions import *
from src.sigmund.features.agreement_score import AgreementScoreExtractor
from src.sigmund.features.basic_statistics import BasicStatistics
from src.sigmund.features.flesch_reading_ease import FleschExtractor
from src.sigmund.features.liwc import Liwc
from src.sigmund.features.pos import PartOfSpeech
from src.sigmund.features.talk_turn import TalkTurnExtractor
from src.sigmund.features.tfidf import FeatureTFIDF
from src.sigmund.features.vocabulary_size import VocabularySize
from src.sigmund.preprocessing.words import Lemmatizer, Stemmer, Tokenizer

nlp = spacy.load("de_core_news_sm", disable=["ner", "parser"])

# Read data
folder = os.path.join(os.getcwd(), "data", "all_transcripts")
files = [os.path.join(root, f) for root, _, files in os.walk(folder)
         for f in files if f.endswith(".docx")]
parser = Parser(nlp=nlp, metadata_path="./data/all_transcripts/CBCT_corrected.xls")
parser.read_from_files(files)
queryable = Queryable.from_parser(parser)

# Initialize pipeline
pipeline = Pipeline(queryable=queryable)

# Add preprocessing components
pipeline.add_components([Tokenizer(), Stemmer(), Lemmatizer()])

# Add basic statistics
pipeline.add_component(BasicStatistics())
pipeline.add_component(
    RandomForest(
        inputs=[BASIC_STATISTICS_DOCUMENT_MF],
        output=CLASSIFICATION_NAIVE_BAYES_BASIC_STATISTICS,
        cross_validate=True, voting=False,
        number_cross_validations=4))

# Add Vocabulary size
pipeline.add_component(VocabularySize())
pipeline.add_component(
    RandomForest(
        inputs=[VOCABULARY_SIZE_DOCUMENT_MF],
        output=CLASSIFICATION_NAIVE_BAYES_VOCABULARY_SIZE,
        cross_validate=True, voting=False,
        number_cross_validations=4))

# Add Flesch extractor
pipeline.add_component(FleschExtractor())
pipeline.add_component(
    RandomForest(
        inputs=[FRE_DOCUMENT_MF],
        output=CLASSIFICATION_NAIVE_BAYES_FLESCH,
        cross_validate=True, voting=False,
        number_cross_validations=4))

# Add tfidf
pipeline.add_component(
    FeatureTFIDF(
        white_list=['ja', 'auch', 'wenn', 'also', 'werden', 'schon', 'wir', 'und',
                    'haben', 'du', 'sehr'])),

# Add liwc
pipeline.add_component(
    Liwc(
        white_list=['Posemo', 'Past', 'Present', 'Future', 'Metaph',
                    'Death', 'Affect', 'Incl', 'Achieve']))

# Add pos
pipeline.add_component(PartOfSpeech(white_list=["ADV", "PPER", "ADJD", "VAFIN", "KON"]))

# Aggregate features
pipeline.add_component(FeatureMerger())

# Classify
pipeline.add_component(RandomForest(cross_validate=True, number_cross_validations=4))

# Print results
storage = pipeline.execute(visualise=True)
