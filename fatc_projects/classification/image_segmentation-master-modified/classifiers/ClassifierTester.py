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

        skf = StratifiedKFold(n_splits=n, random_state=42, shuffle=True)  # 10 rodadas

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

            # # Bayes classifier - solution 1 (not working properly right now)
            # # classifier_bayes_complete = BayesClassifier(complete_view_train, y_train, classes)
            # # classifier_bayes_shape = BayesClassifier(shape_view_train, y_train, classes)
            # # classifier_bayes_rgb = BayesClassifier(rgb_view_train, y_train, classes)
            
            # Bayes classifier - working solution 1
            classifier_bayes_complete = NaiveBayes()
            classifier_bayes_complete.fit(rgb_view_train, y_train)
            classifier_bayes_shape = NaiveBayes()
            classifier_bayes_shape.fit(rgb_view_train, y_train)
            classifier_bayes_rgb = NaiveBayes()
            classifier_bayes_rgb.fit(rgb_view_train, y_train)

            priori = classifier_bayes_complete.priors

            classifier_knn_complete = KnnClassifier(complete_view_train, y_train, 3, classes)
            classifier_knn_shape = KnnClassifier(shape_view_train, y_train, 3, classes)
            classifier_knn_rgb = KnnClassifier(rgb_view_train, y_train, 3, classes)
        
            posteriori_knn_complete = classifier_knn_complete.compute_posteriors(complete_view_test, y_test)
            posteriori_knn_shape = classifier_knn_shape.compute_posteriors(shape_view_test, y_test)
            posteriori_knn_rgb = classifier_knn_rgb.compute_posteriors(rgb_view_test, y_test)

            # # posteriori_bayes_rgb = classifier_bayes_rgb.compute_posteriors(rgb_view_test, y_test)
            posteriori_bayes_complete = classifier_bayes_complete.predict(rgb_view_test)
            posteriori_bayes_shape = classifier_bayes_shape.predict(rgb_view_test)
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
        
        print(accuracies['accuracies_max_rule'])
        print("-" * 50)
        print(accuracies['accuracies_knn_complete'])
    
        return accuracies

        # X = dataframe.X
        # Y = dataframe.Y
        # classes = dataframe.classes
        # num_classes = len(classes)
        # num_elements = np.shape(X)[0]
        # num_elements_for_class = (num_elements/n)/num_classes
        
        # X_for_class = []
        # X_index = [(index, x) for (index, x) in enumerate(X)]

        # size_view_shape = 9
        # size_view_rgb = 10
        
        # for i in xrange(num_classes):
        #     separated_X = [[x,i] for (index, x) in X_index if Y[index]==i]
        #     random.shuffle(separated_X)
        #     X_for_class.append(separated_X)

        # folds = []

        # for k in xrange(n):
        #     low_index = k*num_elements_for_class
        #     high_index = low_index + num_elements_for_class
        #     new_fold = []
        #     for i in xrange(num_classes):
        #         new_fold += X_for_class[i][low_index:high_index]
        #     random.shuffle(new_fold)
        #     folds.append(new_fold)

        # print 'Folds created'
        # accuracies_bayes_shape = []
        # accuracies_bayes_rgb = []
        # accuracies_knn_shape = []
        # accuracies_knn_rgb = []
        # accuracies_majority = []
        
        # for k in xrange(n):
        #     print 'Iteration', k , 'of', n
        #     test = folds[k]
        #     train = []
        #     for i in xrange(n):
        #         if i != k:
        #             train += folds[i]

        #     X_train = np.array([t[0] for t in train])
        #     Y_train = np.array([t[1] for t in train]).ravel()
        #     X_test = np.array([t[0] for t in test])
        #     Y_test = np.array([t[1] for t in test]).ravel()

        #     len_train = np.shape(X_train)[0]
        #     len_test = np.shape(X_test)[0]
            
        #     shape_mapping_train = np.ix_(range(len_train), range(size_view_shape))
        #     shape_mapping_test = np.ix_(range(len_test), range(size_view_shape))
        #     rgb_mapping_train = np.ix_(range(len_train), range(size_view_shape, size_view_shape + size_view_rgb))
        #     rgb_mapping_test = np.ix_(range(len_test), range(size_view_shape, size_view_shape + size_view_rgb)) 

        #     X_train_shape = X_train[shape_mapping_train]
        #     X_test_shape = X_test[shape_mapping_test]
        #     X_train_rgb = X_train[rgb_mapping_train]
        #     X_test_rgb = X_test[rgb_mapping_test]
            
        #     classifier_bayes_shape = BayesClassifier(X_train_shape, Y_train, classes)
        #     classifier_bayes_rgb = BayesClassifier(X_train_rgb, Y_train, classes)
        #     classifier_knn_shape = KnnClassifier(X_train_shape, Y_train, 3)
        #     classifier_knn_rgb = KnnClassifier(X_train_rgb, Y_train, 3)
        #     classifier_majority = MajorityVoteClassifier(X_train, Y_train, classes)

        #     accuracies_bayes_shape.append(classifier_bayes_shape.evaluate(X_test_shape, Y_test))
        #     accuracies_bayes_rgb.append(classifier_bayes_rgb.evaluate(X_test_rgb, Y_test))
        #     accuracies_knn_shape.append(classifier_knn_shape.evaluate(X_test_shape, Y_test))
        #     accuracies_knn_rgb.append(classifier_knn_rgb.evaluate(X_test_rgb, Y_test))
        #     accuracies_majority.append(classifier_majority.evaluate(X_test, Y_test))

        # accuracies = {'accuracies_bayes_shape': accuracies_bayes_shape,
        #               'accuracies_bayes_rgb': accuracies_bayes_rgb,
        #               'accuracies_knn_shape': accuracies_knn_shape,
        #               'accuracies_knn_rgb': accuracies_knn_rgb,
        #               'accuracies_majority': accuracies_majority}
        
        # return accuracies

