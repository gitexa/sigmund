import operator
from src.pipelinelib.text_body import TextBody
from typing import Dict

import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, cross_val_score
from spacy.tokens import Doc

from src.pipelinelib.component import Component
from src.pipelinelib.extension import Extension
from src.pipelinelib.querying import Queryable
from src.sigmund.extensions import CLASSIFICATION_NAIVE_BAYES, FEATURE_VECTOR


class NaiveBayes(Component):
    """
    Performs naive bayes on a feature dataframe 
    """

    def __init__(self):
        super().__init__(
            NaiveBayes.__name__,
            required_extensions=[FEATURE_VECTOR],
            creates_extensions=[CLASSIFICATION_NAIVE_BAYES])

    def apply(self, storage: Dict[Extension, pd.DataFrame],
              queryable: Queryable) -> Dict[Extension, pd.DataFrame]:

        # get features
        df_feature_vector = FEATURE_VECTOR.load_from(storage=storage)

        metadata = queryable.execute(level=TextBody.DOCUMENT)
        df_feature_vector = pd.merge(
            metadata[['couple_id', 'is_depressed_group']],
            df_feature_vector, on='couple_id', how='inner')

        display(df_feature_vector)

        couple_id = df_feature_vector["couple_id"]
        labels = df_feature_vector["is_depressed_group"].astype(int)
        features = df_feature_vector[df_feature_vector.columns.difference(
            ["couple_id", "is_depressed_group"], sort=False)]

        
        # Using "normal" validation
        features_train, features_test, label_train, label_test, indices_train, indices_test = train_test_split(
            features, labels, features.index.values, test_size=0.20, random_state=42)
        
        # fit classifier
        classifier = MultinomialNB()
        classifier.fit(features_train, label_train)

        # predict
        predicted = classifier.predict(features_test)
        display(label_test)
        display(predicted)

        # evaluate classifier
        accuracy = ((predicted == label_test).sum()) / len(label_test)
        display(accuracy)

        # Using cross validation
        classifier = MultinomialNB()    
        cv = StratifiedKFold(n_splits=5, random_state=42)
        scores = cross_val_score(classifier, features, labels, cv=cv)
        display(np.mean(scores))


        return {CLASSIFICATION_NAIVE_BAYES: predicted}


'''
class NaiveBayesOnTfIdf(Component):
    BAYES = Extension("tfidf_gaussian_nb", int)

    def __init__(self, x_train, y_train):
        super().__init__(name=NaiveBayesOnTfIdf.__name__, required_extensions=[
            TfIdf.TFIDF], creates_extensions=[NaiveBayesOnTfIdf.BAYES])
        self.naive_bayes = MultinomialNB()
        self.naive_bayes.fit(x_train, y_train)

    def apply(self, doc: Doc) -> Doc:
        feature_vector = operator.attrgetter(TfIdf.TFIDF.name)(doc._)
        prediction = self.naive_bayes.predict(feature_vector)

        doc._.tfidf_gaussian_nb = prediction[0]
        return doc
'''
