import spacy

from pipelinelib.pipeline import Pipeline
from sigmund.features import syllables, words

if __name__ == "__main__":
    nlp = spacy.load("de_core_news_sm", disable=["ner", "parser"])
    sentence = "Ich habe Hunger und bin glücklich aber auch traurig und ein Baum"
    doc = Pipeline(model=nlp) \
        .add_component(syllables.SyllableExtractor()) \
        .add_component(words.WordExtractor()) \
        .add_component(words.LiwcScores()) \
        .execute(text=sentence)

    print("\n".join(map(str, [doc._.words, doc._.word_count, doc._.syllables])))
