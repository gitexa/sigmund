import os
import pickle
from functools import reduce
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.core.display import display
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import (StratifiedKFold, cross_val_predict,
                                     cross_val_score, train_test_split)
from sklearn.naive_bayes import MultinomialNB

from src.pipelinelib.component import Component
from src.pipelinelib.extension import Extension, ExtensionKind
from src.pipelinelib.querying import Parser, Queryable
from src.pipelinelib.text_body import TextBody
from src.sigmund.extensions import CLASSIFICATION_NAIVE_BAYES, FEATURE_VECTOR


class NaiveBayes(Component):
    """
    Performs naive bayes on a feature vector and prints results

    @param inputs: feature extensions from the storage
    @param output: classification results
    @param voting: if used as voting classifiers based on the results of other classifiers
    @param hamilton: if classifier is to be used on the hamilton scale (4 classes) or binary (2 classes)
    @param save_model: if trained model should be saved (works only for normal validation, not for cross-validation)
    @param model_path: path to save and reload the model
    @param evaluate_model: if pretrained model is to be used on training data (model is loaded from storage)
    @param cross_validate: if crossvalidation or "normal" validation (random test and training set)
    @param number_cross_validations: number of crossvalidations to be performed (shuffles dataset and changes folds)
    """

    def __init__(self,
                 inputs: List[Extension] = None,
                 output: Extension = None,
                 voting: bool = False,
                 hamilton: bool = False,
                 save_model: bool = False,
                 model_path: str = './data/model/',
                 evaluate_model: bool = False,
                 cross_validate: bool = False,
                 number_cross_validations: int = 1):

        if evaluate_model is True and save_model is True:
            raise Exception(
                f"Invalid combination: evaluate_model={evaluate_model}, save_model={save_model}; only one can be True")

        self.inputs: List[Extension] = inputs or [FEATURE_VECTOR]
        self.output: Extension = output or CLASSIFICATION_NAIVE_BAYES
        self.voting = voting
        self.hamilton = hamilton
        self.save_model = save_model
        self.model_path = model_path
        self.evaluate_model = evaluate_model
        self.number_cross_validations = number_cross_validations
        self.cross_validate = cross_validate

        super().__init__(
            f"{NaiveBayes.__name__} for {self.inputs}",
            required_extensions=self.inputs,
            creates_extensions=[self.output])

    def apply(self, storage: Dict[Extension, pd.DataFrame],
              queryable: Queryable) -> Dict[Extension, pd.DataFrame]:

        # Read user input:
        # 1) from which features to construct feature vector
        # 2) key to store the results
        # if voting==True, feature vector consists of previous classifier outputs
        # if hamilton==True, 4-class prediction on Hamilton depression scale is performed, otherwise binary classification (depressed vs non-depressed is performed)

        # Binary classification (is depressed vs. not-depressed couple)
        if not self.hamilton:

            # Construct feature vector for non-voting-classifier
            if not self.voting:
                if not len(self.inputs):
                    return dict()

                elif len(self.inputs) == 1:
                    df_feature_vector = self.inputs[0].load_from(storage=storage)

                else:
                    loaded = map(lambda e: e.load_from(storage=storage), self.inputs)
                    df_feature_vector = reduce(lambda left, right: pd.merge(
                        left, right, on=Parser.COUPLE_ID, how="inner"), loaded)

                if "is_depressed_group" not in df_feature_vector.columns:
                    metadata = queryable.execute(level=TextBody.DOCUMENT)
                    df_feature_vector = pd.merge(
                        metadata[['couple_id', 'is_depressed_group']],
                        df_feature_vector, on='couple_id', how='inner')

            # Construct feature vector for voting-classifier
            else:
                if not len(self.inputs):
                    raise Exception(
                        'When voting, keys for the classifier results must be used.')

                elif len(self.inputs) == 1:
                    df_feature_vector = self.inputs[0].load_from(storage=storage)

                else:
                    loaded = [e.load_from(storage=storage) for e in self.inputs]

                    # rename predicted columns, join on couple id
                    acc = loaded[0]
                    for extension, df in zip(self.inputs[1:], loaded[1:]):
                        acc = self.__merge_frame(
                            left=acc, right_ext=extension, right=df)

                    acc.rename(columns={
                        "predicted": f"{self.inputs[0].name}_predicted"
                    }, inplace=True)
                    df_feature_vector = acc

                    metadata = queryable.execute(level=TextBody.DOCUMENT)
                    df_feature_vector = pd.merge(
                        metadata[['couple_id', 'is_depressed_group']],
                        df_feature_vector, on='couple_id', how='inner')

            # Display and partition the feature vector
            display(df_feature_vector)
            couple_id = df_feature_vector["couple_id"]
            labels = df_feature_vector["is_depressed_group"].astype(int)
            features = df_feature_vector[df_feature_vector.columns.difference(
                ["couple_id", "is_depressed_group"], sort=False)]

            # With "normal" validation = without cross-validation (with the possibility to store the model)
            if not self.cross_validate:

                # construct train and test set
                features_train, features_test, label_train, label_test, indices_train, indices_test = train_test_split(
                    features, labels, features.index.values, test_size=0.50)

                # if training_mode
                if not self.evaluate_model:

                    # fit classifier
                    classifier = MultinomialNB()
                    classifier.fit(features_train, label_train)

                    # if save model (CAVE: overwrites model for now)
                    if self.save_model:
                        if not os.path.exists(self.model_path):
                            os.mkdir(self.model_path)

                        pkl_filename = os.path.join(
                            self.model_path, "naive_bayes_binary.pkl")
                        with open(pkl_filename, 'wb') as file:
                            pickle.dump(classifier, file)

                # if evaluation mode
                if self.evaluate_model:

                    # load model
                    pkl_filename = os.path.join(
                        self.model_path, "naive_bayes_binary.pkl")
                    with open(pkl_filename, 'rb') as file:
                        classifier = pickle.load(file)

                # predict
                predicted_test = classifier.predict(features_test)
                df_prediction_test = pd.DataFrame(
                    data=predicted_test, columns=['predicted'],
                    index=label_test.index.copy())

                # evaluate classifier
                accuracy = ((predicted_test == label_test).sum()) / len(label_test)

                # aggregate results and build dataframe
                couple_id_test = df_feature_vector.iloc[indices_test, :]['couple_id']
                df_prediction_summary = pd.concat(
                    [couple_id_test, label_test, df_prediction_test], axis=1)

                # Print results
                display(
                    'Predictions on a random test set (without stratified sampling, unsuited for small sets)')
                display(df_prediction_summary)
                display(f'Accuracy on test set: {accuracy}')

            # Using cross validation
            else:
                gt = df_feature_vector['is_depressed_group']
                cv = StratifiedKFold(n_splits=5, random_state=42)

                classifier = MultinomialNB()

                prediction_test_cv = cross_val_predict(
                    classifier, features, labels, cv=cv)
                df_prediction_test_cv = pd.DataFrame(
                    data=prediction_test_cv, columns=['predicted'],
                    index=labels.index.copy())

                df_prediction_summary = pd.concat(
                    [couple_id, labels, df_prediction_test_cv], axis=1)

                # multiple times cross-validation with different splits for more accurate estimation
                accuracy_cv_list = []
                accuracy_cv_mean_list = []
                accuracy_cv_variance_list = []
                accuracy_cv_max = 0
                accuracy_cv_mean_average = 0
                accuracy_cv_variance_average = 0
                f1_cv_list = []
                f1_cv_mean_list = []
                f1_cv_variance_list = []
                f1_cv_max = 0
                f1_cv_mean_average = 0
                f1_cv_variance_average = 0

                for i in range(self.number_cross_validations):
                    # shuffle features and labels for different folds
                    idx = np.random.permutation(features.index)
                    features = features.reindex(idx)
                    labels = labels.reindex(idx)

                    # calculate values
                    accuracy_cv = cross_val_score(classifier, features, labels, cv=cv)
                    f1_cv = f1_score(y_true=gt, y_pred=prediction_test_cv)

                    accuracy_cv_list.append(accuracy_cv)
                    accuracy_cv_mean_list.append(np.mean(accuracy_cv))
                    accuracy_cv_variance_list.append(np.var(accuracy_cv))

                    f1_cv_list.append(f1_cv)
                    f1_cv_mean_list.append(np.mean(f1_cv))
                    f1_cv_variance_list.append(np.var(f1_cv))

                accuracy_cv_max = np.max(accuracy_cv_mean_list)
                accuracy_cv_mean_average = np.mean(accuracy_cv_mean_list)
                accuracy_cv_variance_average = np.mean(accuracy_cv_variance_list)

                f1_cv_max = np.max(f1_cv_mean_list)
                f1_cv_mean_average = np.mean(f1_cv_mean_list)
                f1_cv_variance_average = np.mean(f1_cv_variance_list)

                display('--' * 20)
                display('Prediction from one cross-validation')
                display(df_prediction_summary)
                display(
                    f'Scores from {self.number_cross_validations} cross-validation(s)')
                display('--' * 20)
                display(
                    f'Accuracy: {accuracy_cv_list}')
                display(
                    f'Mean of each: {accuracy_cv_mean_list} | Var of each: {accuracy_cv_variance_list}')
                display(
                    f'Overall max: {accuracy_cv_max} | Overall mean: {accuracy_cv_mean_average} | Overall variance: {accuracy_cv_variance_average}')
                display('--' * 20)
                display(f'F1-score: {f1_cv_list}')
                display(
                    f'Mean of each: {f1_cv_mean_list} | Var of each: {f1_cv_variance_list}')
                display(
                    f'Overall max: {f1_cv_max} | Overall mean: {f1_cv_mean_average} | Overall variance: {f1_cv_variance_average}')
                display('--' * 20)

            return {self.output: df_prediction_summary}

        # Hamilton Depression score classification
        else:
            # Construct feature vector for non-voting-classifier
            if not self.voting:
                if not len(self.inputs):
                    return dict()

                elif len(self.inputs) == 1:
                    df_feature_vector = self.inputs[0].load_from(storage=storage)

                else:
                    loaded = map(lambda e: e.load_from(storage=storage), self.inputs)
                    df_feature_vector = reduce(lambda left, right: pd.merge(
                        left, right, on=Parser.COUPLE_ID, how="inner"), loaded)

                if "hamilton" not in df_feature_vector.columns:
                    metadata_hrs = queryable.execute(level=TextBody.PARAGRAPH)
                    metadata_hrs = metadata_hrs.loc[metadata_hrs['gender'] == 'W'][[
                        'couple_id', 'hamilton']].groupby('couple_id').first().reset_index()
                    metadata_hrs['hamilton_score'] = metadata_hrs['hamilton'].apply(
                        lambda row: self.get_hamilton_class(row))
                    df_feature_vector = pd.merge(
                        metadata_hrs[['couple_id', 'hamilton_score']],
                        df_feature_vector, on='couple_id', how='inner')

            # Construct feature vector for voting-classifier
            else:
                if not len(self.inputs):
                    raise Exception(
                        'When voting, keys for the classifier results must be used.')

                elif len(self.inputs) == 1:
                    df_feature_vector = self.inputs[0].load_from(storage=storage)

                else:
                    loaded = [e.load_from(storage=storage) for e in self.inputs]

                    # rename predicted columns, join on couple id
                    acc = loaded[0]
                    for extension, df in zip(self.inputs[1:], loaded[1:]):
                        acc = self.__merge_frame(
                            left=acc, right_ext=extension, right=df)

                    acc.rename(columns={
                        "predicted": f"{self.inputs[0].name}_predicted"
                    }, inplace=True)
                    df_feature_vector = acc

                    if "hamilton" not in df_feature_vector.columns:
                        metadata_hrs = queryable.execute(level=TextBody.PARAGRAPH)
                        metadata_hrs = metadata_hrs.loc[metadata_hrs['gender'] == 'W'][[
                            'couple_id', 'hamilton']].groupby('couple_id').first().reset_index()
                        metadata_hrs['hamilton_score'] = metadata_hrs['hamilton'].apply(
                            lambda row: self.get_hamilton_class(row))
                        df_feature_vector = pd.merge(
                            metadata_hrs[['couple_id', 'hamilton_score']],
                            df_feature_vector, on='couple_id', how='inner')

            # Display and partition the feature vector
            display(df_feature_vector)
            couple_id = df_feature_vector["couple_id"]
            labels = df_feature_vector["hamilton_score"].astype(int)
            features = df_feature_vector[df_feature_vector.columns.difference(
                ["couple_id", "hamilton_score"], sort=False)]

            # With "normal" validation = without cross-validation (with the possibility to store the model)
            if not self.cross_validate:

                # construct the train and test set for hamilton
                features_train, features_test, label_train, label_test, indices_train, indices_test = train_test_split(
                    features, labels, features.index.values, test_size=0.50)

                # if training_mode
                if not self.evaluate_model:

                    # fit classifier
                    classifier = MultinomialNB()
                    classifier.fit(features_train, label_train)

                    # if save model (CAVE: overwrites model for now)
                    if self.save_model:
                        if not os.path.exists(self.model_path):
                            os.mkdir(self.model_path)

                        pkl_filename = os.path.join(
                            self.model_path, "naive_bayes_hamilton.pkl")
                        with open(pkl_filename, 'wb') as file:
                            pickle.dump(classifier, file)

                # if evaluation mode
                if self.evaluate_model:

                    # load model
                    pkl_filename = os.path.join(
                        self.model_path, "naive_bayes_hamilton.pkl")
                    with open(pkl_filename, 'rb') as file:
                        classifier = pickle.load(file)

                # predict
                predicted_test = classifier.predict(features_test)
                df_prediction_test = pd.DataFrame(
                    data=predicted_test, columns=['predicted'],
                    index=label_test.index.copy())

                # evaluate classifier
                accuracy = ((predicted_test == label_test).sum()) / len(label_test)

                # aggregate results and build dataframe
                couple_id_test = df_feature_vector.iloc[indices_test, :]['couple_id']
                df_prediction_summary = pd.concat(
                    [couple_id_test, label_test, df_prediction_test], axis=1)

                # Print results
                display(
                    'Predictions on a random test set (without stratified sampling, unsuited for small sets)')
                display(df_prediction_summary)
                display(f'Accuracy on test set: {accuracy}')

            # Using cross validation
            else:

                gt = df_feature_vector['hamilton_score']
                cv = StratifiedKFold(n_splits=5, random_state=42)

                classifier = MultinomialNB()

                prediction_test_cv = cross_val_predict(
                    classifier, features, labels, cv=cv)
                df_prediction_test_cv = pd.DataFrame(
                    data=prediction_test_cv, columns=['predicted'],
                    index=labels.index.copy())
                df_prediction_summary = pd.concat(
                    [couple_id, labels, df_prediction_test_cv], axis=1)

                # multiple times cross-validation with different splits for more accurate estimation
                accuracy_cv_list = []
                accuracy_cv_mean_list = []
                accuracy_cv_variance_list = []
                accuracy_cv_max = 0
                accuracy_cv_mean_average = 0
                accuracy_cv_variance_average = 0
                f1_cv_list = []
                f1_cv_mean_list = []
                f1_cv_variance_list = []
                f1_cv_max = 0
                f1_cv_mean_average = 0
                f1_cv_variance_average = 0

                for i in range(self.number_cross_validations):
                    # shuffle features and labels for different folds
                    idx = np.random.permutation(features.index)
                    features = features.reindex(idx)
                    labels = labels.reindex(idx)

                    # calculate values
                    accuracy_cv = cross_val_score(classifier, features, labels, cv=cv)
                    f1_cv = f1_score(y_true=gt, y_pred=prediction_test_cv)

                    accuracy_cv_list.append(accuracy_cv)
                    accuracy_cv_mean_list.append(np.mean(accuracy_cv))
                    accuracy_cv_variance_list.append(np.var(accuracy_cv))

                    f1_cv_list.append(f1_cv)
                    f1_cv_mean_list.append(np.mean(f1_cv))
                    f1_cv_variance_list.append(np.var(f1_cv))

                accuracy_cv_max = np.max(accuracy_cv_mean_list)
                accuracy_cv_mean_average = np.mean(accuracy_cv_mean_list)
                accuracy_cv_variance_average = np.mean(accuracy_cv_variance_list)

                f1_cv_max = np.max(f1_cv_mean_list)
                f1_cv_mean_average = np.mean(f1_cv_mean_list)
                f1_cv_variance_average = np.mean(f1_cv_variance_list)

                display('--' * 20)
                display('Prediction from one cross-validation')
                display(df_prediction_summary)
                display(
                    f'Scores from {self.number_cross_validations} cross-validation(s)')
                display('--' * 20)
                display(
                    f'Accuracy: {accuracy_cv_list}')
                display(
                    f'Mean of each: {accuracy_cv_mean_list} | Var of each: {accuracy_cv_variance_list}')
                display(
                    f'Overall max: {accuracy_cv_max} | Overall mean: {accuracy_cv_mean_average} | Overall variance: {accuracy_cv_variance_average}')
                display('--' * 20)
                display(f'F1-score: {f1_cv_list}')
                display(
                    f'Mean of each: {f1_cv_mean_list} | Var of each: {f1_cv_variance_list}')
                display(
                    f'Overall max: {f1_cv_max} | Overall mean: {f1_cv_mean_average} | Overall variance: {f1_cv_variance_average}')
                display('--' * 20)

                return {self.output: df_prediction_summary}

    def __merge_frame(
            self, left: pd.DataFrame, right_ext: Extension, right: pd.DataFrame):
        left_predicted_cols = [col for col in left.columns if col.endswith("predicted")]
        left = left[["couple_id"] + left_predicted_cols]

        right = right[["couple_id", "predicted"]]
        right.rename(columns={"predicted": f"{right_ext.name}_predicted"}, inplace=True)

        return pd.merge(left, right, on=Parser.COUPLE_ID, how="inner")

    def get_hamilton_class(self, value):

        if value >= 0 and value <= 9:
            depression_class = 1
        if value >= 10 and value <= 20:
            depression_class = 2
        if value >= 21 and value <= 30:
            depression_class = 3
        if value > 30:
            depression_class = 4

        return depression_class

    def visualise(self, created: Dict[Extension, pd.DataFrame],
                  queryable: Queryable):

        df_embedded = self.output.load_from(storage=created)
        display(df_embedded)

        plt.figure()
        conf = confusion_matrix(
            y_pred=df_embedded['predicted'],
            y_true=df_embedded['is_depressed_group'],
            normalize='true')
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(conf, annot=True)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

        return {Extension(name=f"{NaiveBayes.__name__} for {self.inputs}", kind=ExtensionKind.CLASSIFIER): fig}
