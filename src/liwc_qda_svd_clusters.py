from sklearn.decomposition import TruncatedSVD
from scipy.sparse import random as sparse_random
from sklearn.model_selection import train_test_split
from sklearn.random_projection import sparse_random_matrix
from sklearn.manifold import TSNE

import spacy
import numpy as np
from spacy.tokens import Doc

import matplotlib.pyplot as plt
X = sparse_random(100, 100, density=0.01, format='csr',
                  random_state=42)

import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.dummy import DummyClassifier

df = pd.read_excel("liwc_exported.xlsx", engine='openpyxl', sheet_name="Sheet0")
labels = df["Source (F)"]
features = df.loc[:, "ppron":"swear"]
label_array = labels.to_numpy()
feature_array = features.to_numpy()
print(feature_array.shape)
max_n_components = 50

iterations = 100

scores = np.zeros((max_n_components, iterations))
scores_dummy = np.zeros((max_n_components, iterations))

if False:
    for j in range(1, max_n_components):
        for i in range(iterations):
            svd = TruncatedSVD(n_components=j, algorithm="arpack")
            features_dense = pd.DataFrame(svd.fit_transform(feature_array))
            X_train, X_test, y_train, y_test = train_test_split(
                features_dense, label_array, test_size=0.2)
            dummy_clf = DummyClassifier(strategy="uniform")
            dummy_clf.fit(X_train, y_train)

            clf = QuadraticDiscriminantAnalysis()
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            score_dummy = dummy_clf.score(X_test, y_test)

            scores[j, i] = score
            scores_dummy[j, i] = score_dummy

    print(scores.mean(axis=1))
    print(scores_dummy.mean(axis=1))


    plt.plot(range(1, max_n_components), scores.mean(axis=1)[1:])
    plt.plot(range(1, max_n_components), scores_dummy.mean(axis=1)[1:])
    plt.ylim((0, 1))
    plt.show()

if __name__ == '__main__':
    # Clustered
    svd = TruncatedSVD(n_components=2, algorithm="arpack")
    tsne = TSNE(n_components=2, random_state=42)
    # features_dense = pd.DataFrame(tsne.fit_transform(feature_array))
    features_dense = pd.DataFrame(svd.fit_transform(feature_array))
    X_train, X_test, y_train, y_test = train_test_split(
        features_dense, label_array, test_size=0.2)

    print(label_array)
    depressed = features_dense[label_array]
    non_depressed = features_dense[np.logical_not(label_array)]



    plt.scatter(non_depressed[0], non_depressed[1])
    plt.scatter(depressed[0], depressed[1])
    plt.show()

    # clf = QuadraticDiscriminantAnalysis()
    # clf.fit(X_train, y_train)
