import spacy

from blocks.pipeline import Pipeline
from components import syllables

if __name__ == "__main__":
    nlp = spacy.load("de_core_news_sm", disable=["ner", "parser"])
    sentence = "Ich habe Hunger und bin glücklich aber auch traurig und ein Baum"
    doc = Pipeline(model=nlp) \
        .add_component(syllables.SyllableExtractor()) \
        .execute(text=sentence)

    print(doc._.syllables)
