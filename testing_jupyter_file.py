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

'''
# Read data
folder = os.path.join(os.getcwd(), "data", "all_transcripts")
files = [os.path.join(root, f) for root, _, files in os.walk(folder)
         for f in files if f.endswith(".docx")]
parser = Parser(nlp=nlp, metadata_path="./data/all_transcripts/CBCT_corrected.xls")
parser.read_from_files(files)
queryable = Queryable.from_parser(parser)

path = 'data/all_new.csv'
queryable.df.to_csv(path)

# Read data
folder = os.path.join(os.getcwd(), "data", "all_transcripts")
files = [os.path.join(root, f) for root, _, files in os.walk(folder)
         for f in files if f.endswith(".docx")]

# files = [
#"data/transcripts/Paar 27_T1_IM_FW.docx",
#"data/transcripts/Paar 182_T1_IM_FW.docx",
#"data/transcripts/Paar 81_T1_IM_FW.docx",
#"data/transcripts/Paar 47_T1_IM_FW.docx",
#"data/transcripts/Paar 58_T1_IM_FW.docx",
#"data/transcripts/Paar 29_T1_IM_FW.docx",
#"data/transcripts/Paar 105_T1_IM_FW.docx",
#"data/transcripts/Paar 60_T1_IM_FW.docx",
#"data/transcripts/Paar 138_T1_IM_FW.docx",
# "data/transcripts/Paar 87_T1_IM_FW.docx"]

parser = Parser(
    nlp=nlp, metadata_path="./data/transcripts/Kopie von Transkriptionspaare_Daten.xls")
parser.read_from_files(files)
queryable = Queryable.from_parser(parser)
'''

path = 'data/all_new.csv'
queryable = Queryable.from_csv(path, nlp)


# Initialize pipeline
pipeline = Pipeline(queryable=queryable)

# Add preprocessing components
pipeline.add_components([Tokenizer(), Stemmer(), Lemmatizer()])

# Add talking turn
# pipeline.add_component(TalkTurnExtractor())
# pipeline.add_component(RandomForest(inputs=[TALKTURN], #output=CLASSIFICATION_NAIVE_BAYES_TALKING_TURNS, cross_validate=True, #voting=False, number_cross_validations=4))

# Add agreement score
# pipeline.add_component(AgreementScoreExtractor())
# pipeline.add_component(RandomForest(inputs=[AGREEMENTSCORE], #output=CLASSIFICATION_NAIVE_BAYES_AGREEMENT_SCORE, cross_validate=True, #voting=False, number_cross_validations=4))

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

# Add tfidf and tfidf-classifier
# white_list=['ja', 'auch', 'wenn', 'also', 'werden', 'schon', 'wir', 'und', 'haben', 'du', 'sehr']
pipeline.add_component(FeatureTFIDF())
pipeline.add_component(
    RandomForest(
        inputs=[TFIDF_DOCUMENT_MF],
        output=CLASSIFICATION_NAIVE_BAYES_TFIDF, cross_validate=True,
        voting=False, number_cross_validations=4))

# Add liwc and liwc-classifier
# white_list=['Posemo', 'Past', 'Present', 'Future', 'Metaph','Death', 'Affect', 'Incl', 'Achieve']
pipeline.add_component(Liwc())
pipeline.add_component(
    RandomForest(
        inputs=[LIWC_DOCUMENT_MF],
        output=CLASSIFICATION_NAIVE_BAYES_LIWC, voting=False,
        cross_validate=True, number_cross_validations=4))

# Add pos and pos-classifier
# white_list=["ADV", "PPER", "ADJD", "VAFIN", "KON"]
pipeline.add_component(PartOfSpeech())
pipeline.add_component(
    RandomForest(
        inputs=[POS_DOCUMENT_MF],
        output=CLASSIFICATION_NAIVE_BAYES_POS, voting=False,
        cross_validate=True, number_cross_validations=4))

# Classify with voting-classifier
pipeline.add_component(
    RandomForest(
        inputs=[CLASSIFICATION_NAIVE_BAYES_TFIDF,
                CLASSIFICATION_NAIVE_BAYES_LIWC,
                CLASSIFICATION_NAIVE_BAYES_POS, ],
        output=CLASSIFICATION_NAIVE_BAYES_VOTING, voting=True,
        cross_validate=True, number_cross_validations=4))

# Print results
storage = pipeline.execute(visualise=True)

# Initialize pipeline
pipeline = Pipeline(queryable=queryable)

# Add preprocessing components
pipeline.add_components([Tokenizer(), Stemmer(), Lemmatizer()])

# Add talking turn
# pipeline.add_component(TalkTurnExtractor())
#pipeline.add_component(NaiveBayes(inputs=[TALKTURN], output=CLASSIFICATION_NAIVE_BAYES_TALKING_TURNS, cross_validate=True, voting=False, number_cross_validations=4))

# Add agreement score
# pipeline.add_component(AgreementScoreExtractor())
#pipeline.add_component(NaiveBayes(inputs=[AGREEMENTSCORE], output=CLASSIFICATION_NAIVE_BAYES_AGREEMENT_SCORE, cross_validate=True, voting=False, number_cross_validations=4))

# Add basic statistics
pipeline.add_component(BasicStatistics())
pipeline.add_component(
    NaiveBayes(
        inputs=[BASIC_STATISTICS_DOCUMENT_MF],
        output=CLASSIFICATION_NAIVE_BAYES_BASIC_STATISTICS,
        cross_validate=True, voting=False,
        number_cross_validations=4))

# Add Vocabulary size
pipeline.add_component(VocabularySize())
pipeline.add_component(
    NaiveBayes(
        inputs=[VOCABULARY_SIZE_DOCUMENT_MF],
        output=CLASSIFICATION_NAIVE_BAYES_VOCABULARY_SIZE,
        cross_validate=True, voting=False,
        number_cross_validations=4))

# Add Flesch extractor
pipeline.add_component(FleschExtractor())
pipeline.add_component(
    NaiveBayes(
        inputs=[FRE_DOCUMENT_MF],
        output=CLASSIFICATION_NAIVE_BAYES_FLESCH,
        cross_validate=True, voting=False,
        number_cross_validations=4))

# Add tfidf and tfidf-classifier
# white_list=['ja', 'auch', 'wenn', 'also', 'werden', 'schon', 'wir', 'und', 'haben', 'du', 'sehr']
pipeline.add_component(FeatureTFIDF())
pipeline.add_component(
    NaiveBayes(
        inputs=[TFIDF_DOCUMENT_MF],
        output=CLASSIFICATION_NAIVE_BAYES_TFIDF, cross_validate=True,
        voting=False, number_cross_validations=4))

# Add liwc and liwc-classifier
# white_list=['Posemo', 'Past', 'Present', 'Future', 'Metaph','Death', 'Affect', 'Incl', 'Achieve']
pipeline.add_component(Liwc())
pipeline.add_component(
    NaiveBayes(
        inputs=[LIWC_DOCUMENT_MF],
        output=CLASSIFICATION_NAIVE_BAYES_LIWC, voting=False,
        cross_validate=True, number_cross_validations=4))

# Add pos and pos-classifier
# white_list=["ADV", "PPER", "ADJD", "VAFIN", "KON"]
pipeline.add_component(PartOfSpeech())
pipeline.add_component(
    NaiveBayes(
        inputs=[POS_DOCUMENT_MF],
        output=CLASSIFICATION_NAIVE_BAYES_POS, voting=False,
        cross_validate=True, number_cross_validations=4))

# Classify with voting-classifier
pipeline.add_component(
    NaiveBayes(
        inputs=[CLASSIFICATION_NAIVE_BAYES_TFIDF,
                CLASSIFICATION_NAIVE_BAYES_LIWC,
                CLASSIFICATION_NAIVE_BAYES_POS, ],
        output=CLASSIFICATION_NAIVE_BAYES_VOTING, voting=True,
        cross_validate=True, number_cross_validations=4))

# Print results
storage = pipeline.execute(visualise=True)


# Initialize pipeline
pipeline = Pipeline(queryable=queryable)

# Add preprocessing components
pipeline.add_components([Tokenizer(), Stemmer(), Lemmatizer()])

# Add liwc and liwc-classifier
pipeline.add_component(Liwc())
pipeline.add_component(
    RandomForest(
        inputs=[LIWC_DOCUMENT_MF],
        output=CLASSIFICATION_NAIVE_BAYES_LIWC, voting=False,
        cross_validate=True, number_cross_validations=4))

# Add pos and pos-classifier
pipeline.add_component(PartOfSpeech())
pipeline.add_component(
    RandomForest(
        inputs=[POS_DOCUMENT_MF],
        output=CLASSIFICATION_NAIVE_BAYES_POS, voting=False,
        cross_validate=True, number_cross_validations=4))

# Classify with voting-classifier
pipeline.add_component(
    RandomForest(
        inputs=[CLASSIFICATION_NAIVE_BAYES_LIWC, CLASSIFICATION_NAIVE_BAYES_POS, ],
        output=CLASSIFICATION_NAIVE_BAYES_VOTING, voting=True, cross_validate=True,
        number_cross_validations=4))

# Print results
storage = pipeline.execute(visualise=True)


# Initialize pipeline
pipeline = Pipeline(queryable=queryable)

# Add preprocessing components
pipeline.add_components([Tokenizer(), Stemmer(), Lemmatizer()])

# Add liwc and liwc-classifier
pipeline.add_component(
    Liwc(
        white_list=["We", "Swear", "Sexual", "Self", "Sad",
                    "Present", "Posfeel", "Posemo", "Past",
                    "Number", "Music", "Leisure"]))
pipeline.add_component(
    RandomForest(
        inputs=[LIWC_DOCUMENT_MF],
        output=CLASSIFICATION_NAIVE_BAYES_LIWC, voting=False,
        cross_validate=True, number_cross_validations=4))

# Classify with voting-classifier
pipeline.add_component(
    RandomForest(
        inputs=[CLASSIFICATION_NAIVE_BAYES_LIWC, ],
        output=CLASSIFICATION_NAIVE_BAYES_VOTING, voting=True,
        cross_validate=True, number_cross_validations=4))

# Print results
storage = pipeline.execute(visualise=True)


# Initialize pipeline
pipeline = Pipeline(queryable=queryable)

# Add preprocessing components
pipeline.add_components([Tokenizer(), Stemmer(), Lemmatizer()])

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
storage = pipeline.execute(visualise=False)
