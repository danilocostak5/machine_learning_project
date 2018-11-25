from pandas import Index

import pandas as pd
import numpy as np
import csv
import pickle
import warnings
import random
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from classifiers.BayesClassifier import *
from classifiers.KnnClassifier import *
from classifiers.MaxRuleClassifier import *
from classifiers.BayesClassifier2 import NaiveBayes

def computar_acertos(posterioris, y_test):
    y_pred = np.argmax(posterioris, axis=1)
    accuracy = float(len(y_pred[y_pred == y_test])) / len(y_pred)
    return accuracy
    
class ClassifierTester:

    @staticmethod
    def make_n_fold_test(dataframe, n):
        X = dataframe.iloc[:, 1:].values
        y = dataframe.iloc[:, 0].values

        scaler = preprocessing.MinMaxScaler()
        # X = scaler.fit_transform(X)
  
        classes = sorted(set(y.ravel()))

        y = np.asarray(map(lambda i:classes.index(i), y))

        skf = StratifiedKFold(n_splits=n, shuffle=True)  # 10 rodadas

        accuracies_bayes_complete = []
        accuracies_bayes_shape = []
        accuracies_bayes_rgb = []
        accuracies_knn_shape = []
        accuracies_knn_rgb = []
        accuracies_knn_complete = []
        accuracies_max_rule = []

        for idx, (train_index, test_index) in enumerate(skf.split(X, y)):
            print 'Fold', idx+1 , 'of', n

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # ajuste dos dados da visao
            complete_view_train = X_train
            shape_view_train = X_train[:, :9]
            rgb_view_train = X_train[:, 9:]

            # X_test, y_test = X_test[:10], y_test[:10] # sanity check

            complete_view_test = X_test  # visao 1
            shape_view_test = X_test[:, :9]  # visao 2
            rgb_view_test = X_test[:, 9:]  # visao 3

            # Bayes classifier - working solution 1
            classifier_bayes_complete = NaiveBayes()
            classifier_bayes_complete.fit(complete_view_train, y_train)
            classifier_bayes_shape = NaiveBayes()
            classifier_bayes_shape.fit(shape_view_train, y_train)
            classifier_bayes_rgb = NaiveBayes()
            classifier_bayes_rgb.fit(rgb_view_train, y_train)

            priori = classifier_bayes_complete.priors

            classifier_knn_complete = KnnClassifier(complete_view_train, y_train, 3, classes)
            classifier_knn_shape = KnnClassifier(shape_view_train, y_train, 3, classes)
            classifier_knn_rgb = KnnClassifier(rgb_view_train, y_train, 3, classes)
        
            posteriori_knn_complete = classifier_knn_complete.compute_posteriors(complete_view_test, y_test)
            posteriori_knn_shape = classifier_knn_shape.compute_posteriors(shape_view_test, y_test)
            posteriori_knn_rgb = classifier_knn_rgb.compute_posteriors(rgb_view_test, y_test)

            posteriori_bayes_complete = classifier_bayes_complete.predict(complete_view_test)
            posteriori_bayes_shape = classifier_bayes_shape.predict(shape_view_test)
            posteriori_bayes_rgb = classifier_bayes_rgb.predict(rgb_view_test)

            classifier_max_rule_rgb = MaxRuleClassifier()
            posteriori_regra_max_complete = classifier_max_rule_rgb.calcular_regra_max(posteriori_bayes_complete, posteriori_bayes_shape, posteriori_bayes_rgb, posteriori_knn_complete, posteriori_knn_shape, posteriori_knn_rgb, priori)
            
            accuracies_bayes_complete.append(computar_acertos(posteriori_bayes_complete, y_test))
            accuracies_bayes_shape.append(computar_acertos(posteriori_bayes_shape, y_test))
            accuracies_bayes_rgb.append(computar_acertos(posteriori_bayes_rgb, y_test))

            accuracies_knn_complete.append(computar_acertos(posteriori_knn_complete, y_test))
            accuracies_knn_shape.append(computar_acertos(posteriori_knn_shape, y_test))
            accuracies_knn_rgb.append(computar_acertos(posteriori_knn_rgb, y_test))
            
            accuracies_max_rule.append(computar_acertos(posteriori_regra_max_complete, y_test))
   
        accuracies = {
            'accuracies_bayes_complete' : accuracies_bayes_complete,
            'accuracies_bayes_shape': accuracies_bayes_shape,
            'accuracies_bayes_rgb': accuracies_bayes_rgb,
            'accuracies_knn_complete': accuracies_knn_complete,
            'accuracies_knn_shape': accuracies_knn_shape,
            'accuracies_knn_rgb': accuracies_knn_rgb,
            'accuracies_max_rule': accuracies_max_rule}
        
        return accuracies