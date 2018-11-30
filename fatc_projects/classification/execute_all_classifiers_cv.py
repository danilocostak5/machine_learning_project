# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import csv
import pickle
from classificadores.KnnClassifier import *
from classificadores.MaxRuleClassifier import *
from classificadores.NaiveBayes import NaiveBayes
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold

def main():
    print 'Testando todas as views'
    df = pd.read_csv('data/segmentation.test.txt', sep=',')
    acuracias_completas = []
    for i in xrange(1, 30 + 1):
        print 'Repetição', i, 'de 30'
        new_acuracias = execute_10fold_test(df, 10, apply_scale=True, apply_standarization=False)
        acuracias_completas.append(new_acuracias)

    out_completas = open('resultados_modelos/acuracias_all_with_scale.pickle', 'wb')
    pickle.dump(acuracias_completas, out_completas)
    out_completas.close()

def execute_10fold_test(dataframe, n, apply_scale=True, apply_standarization=False):
        X = dataframe.iloc[:, 1:].values
        y = dataframe.iloc[:, 0].values

        if apply_scale:
            scaler = preprocessing.MinMaxScaler()
            X = scaler.fit_transform(X)
        elif apply_standarization:
            normalizer = preprocessing.StandardScaler()
            X = normalizer.fit_transform(X)

        classes = sorted(set(y.ravel()))

        y = np.asarray(map(lambda i:classes.index(i), y))

        skf = StratifiedKFold(n_splits=n, shuffle=True)  # 10 rodadas

        acuracias_bayes_complete = []
        acuracias_bayes_shape = []
        acuracias_bayes_rgb = []
        acuracias_knn_complete = []
        acuracias_knn_shape = []
        acuracias_knn_rgb = []
        acuracias_max_rule = []

        for idx, (train_index, test_index) in enumerate(skf.split(X, y)):
            print 'Fold', idx+1 , 'de', n

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # ajuste dos dados da visao
            complete_view_train = X_train
            shape_view_train = X_train[:, :9]
            rgb_view_train = X_train[:, 9:]

            complete_view_test = X_test  # visao 1
            shape_view_test = X_test[:, :9]  # visao 2
            rgb_view_test = X_test[:, 9:]  # visao 3

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

            acuracias_bayes_complete.append(computar_acertos(posteriori_bayes_complete, y_test))
            acuracias_bayes_shape.append(computar_acertos(posteriori_bayes_shape, y_test))
            acuracias_bayes_rgb.append(computar_acertos(posteriori_bayes_rgb, y_test))

            acuracias_knn_complete.append(computar_acertos(posteriori_knn_complete, y_test))
            acuracias_knn_shape.append(computar_acertos(posteriori_knn_shape, y_test))
            acuracias_knn_rgb.append(computar_acertos(posteriori_knn_rgb, y_test))

            acuracias_max_rule.append(computar_acertos(posteriori_regra_max_complete, y_test))
   
        acuracias = {
            'acuracias_bayes_complete' : acuracias_bayes_complete,
            'acuracias_bayes_shape': acuracias_bayes_shape,
            'acuracias_bayes_rgb': acuracias_bayes_rgb,
            'acuracias_knn_complete': acuracias_knn_complete,
            'acuracias_knn_shape': acuracias_knn_shape,
            'acuracias_knn_rgb': acuracias_knn_rgb,
            'acuracias_max_rule': acuracias_max_rule
        }

        print('acuracias_bayes_complete: {}'.format(np.mean(acuracias_bayes_complete)))
        print('acuracias_bayes_shape: {}'.format(np.mean(acuracias_bayes_shape)))
        print('acuracias_bayes_rgb: {}'.format(np.mean(acuracias_bayes_rgb)))
        print('acuracias_knn_complete {}'.format(np.mean(acuracias_knn_complete)))
        print('acuracias_knn_shape {}'.format(np.mean(acuracias_knn_shape)))
        print('acuracias_knn_rgb {}'.format(np.mean(acuracias_knn_rgb)))
        print('acuracias_max_rule {}\n'.format(np.mean(acuracias_max_rule)))
        
        return acuracias

def computar_acertos(posterioris, y_test):
    y_pred = np.argmax(posterioris, axis=1)
    accuracy = float(len(y_pred[y_pred == y_test])) / len(y_pred)
    return accuracy

if __name__ == "__main__":
    main()