# -*- coding: utf-8 -*-
import operator
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from classificadores.KnnClassifier import *
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold

print('Otimizando o hiperparametro K')
df = pd.read_csv('data/segmentation.test.txt', sep=',')

n = 10 # We are using 10 fold

accuracies = []

X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values

scaler = preprocessing.MinMaxScaler()
X = scaler.fit_transform(X)

classes = sorted(set(y.ravel()))

y = np.asarray(map(lambda i:classes.index(i), y))

accuracies_bayes_complete = []
accuracies_bayes_shape = []
accuracies_bayes_rgb = []
accuracies_knn_complete = []
accuracies_knn_shape = []
accuracies_knn_rgb = []
accuracies_max_rule = []

def computar_acertos(posterioris, y_test):
    y_pred = np.argmax(posterioris, axis=1)
    accuracy = float(len(y_pred[y_pred == y_test])) / len(y_pred)
    return accuracy

for i in range(10):
    skf = StratifiedKFold(n_splits=n, shuffle=True)  # 10 rodadas
    mean_accuracies = 0
    K = 2 * i + 1
    print('Testando K: {}'.format(K))

    for idx, (train_index, valid_index) in enumerate(skf.split(X, y)):
        print 'Fold', idx+1 , 'of', n

        X_train, X_valid = X[train_index], X[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]

        # ajuste dos dados da visao
        complete_view_train = X_train
        shape_view_train = X_train[:, :9]
        rgb_view_train = X_train[:, 9:]

        # X_valid, y_valid = X_valid[:10], y_valid[:10] # sanity check

        complete_view_valid = X_valid  # visao 1
        shape_view_valid = X_valid[:, :9]  # visao 2
        rgb_view_valid = X_valid[:, 9:]  # visao 3

        knn_complete = KnnClassifier(complete_view_train, y_train, K, classes)
        knn_rgb = KnnClassifier(rgb_view_train, y_train, K, classes)
        knn_shape = KnnClassifier(shape_view_train, y_train, K, classes)

        new_accuracy_complete = knn_complete.evaluate_proba(complete_view_valid, y_valid, weighted=True)
        new_accuracy_rgb = knn_rgb.evaluate_proba(rgb_view_valid, y_valid, weighted=True)
        new_accuracy_shape = knn_shape.evaluate_proba(shape_view_valid, y_valid, weighted=True)

        # new_accuracy_complete = knn_complete.compute_posteriors(complete_view_valid, y_valid, weighted=True)
        # new_accuracy_rgb = knn_rgb.compute_posteriors(rgb_view_valid, y_valid, weighted=True)
        # new_accuracy_shape = knn_shape.compute_posteriors(shape_view_valid, y_valid, weighted=True)

        # new_accuracy_complete = computar_acertos(new_accuracy_complete, y_valid)
        # new_accuracy_rgb = computar_acertos(new_accuracy_rgb, y_valid)
        # new_accuracy_shape = computar_acertos(new_accuracy_shape, y_valid)

        mean_accuracies += sum([new_accuracy_rgb, new_accuracy_shape]) / 2.

    mean_accuracies /= 10.
    print('Acurácia: {}'.format(mean_accuracies))
    accuracies.append([K, mean_accuracies])

out = open('resultados_modelos/knn_accuracies_for_k.pickle', 'wb')
pickle.dump(accuracies, out)
out.close()

with open('resultados_modelos/knn_accuracies_for_k.pickle', "rb") as input_file:
    accuracies = pickle.load(input_file)

plt.xlabel('Valor de K')
plt.ylabel('Acuracia')
x = [a for [a,b] in accuracies]
y = [b for [a,b] in accuracies]
plt.plot(x,y)
plt.savefig("resultados_imagens/otimizacao_do_parametro_k.png")
plt.show()