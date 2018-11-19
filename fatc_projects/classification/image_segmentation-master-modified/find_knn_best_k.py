import operator
import random
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from classifiers.DataVectorizer import *
from classifiers.KnnClassifier import *

dv = DataVectorizer(filename='result_pickles/full_view.pickle')

n = 10 # We are using 10 fold

X = dv.X
Y = dv.Y
classes = dv.classes
num_classes = len(classes)
num_elements = np.shape(X)[0]
num_elements_for_class = (num_elements/n)/num_classes

X_for_class = []
X_index = [(index, x) for (index, x) in enumerate(X)]

size_view_shape = 9
size_view_rgb = 10

accuracies = []

for i in xrange(10): #All Ks in the form 2*i + 1
    K = 2*i + 1
    print 'Testing K', K
    mean_accuracies = 0

    for j in xrange(10): #Make 10 fold 10 times
        print 'Making', n, 'fold', j + 1, 'of 10'
        for c in xrange(num_classes):
            separated_X = [[x,c] for (index, x) in X_index if Y[index]==c]
            random.shuffle(separated_X)
            X_for_class.append(separated_X)

        folds = []

        for k in xrange(n):
            low_index = k*num_elements_for_class
            high_index = low_index + num_elements_for_class
            new_fold = []
            for c in xrange(num_classes):
                new_fold += X_for_class[c][low_index:high_index]
            random.shuffle(new_fold)
            folds.append(new_fold)


        test = folds[1]
        train = []
        for f in xrange(n):
            if f != 1:
                train += folds[f]

        len_train = len(train)
        size_train = int(float(5.0*len_train/6.0))
        
        X_train = np.array([t[0] for t in train[:size_train]])
        Y_train = np.array([t[1] for t in train[:size_train]]).ravel()
        X_validation = np.array([t[0] for t in train[size_train:]])
        Y_validation = np.array([t[1] for t in train[size_train:]]).ravel()

        len_train = np.shape(X_train)[0]
        len_validation = np.shape(X_validation)[0]
            
        shape_mapping_train = np.ix_(range(len_train), range(size_view_shape))
        shape_mapping_validation = np.ix_(range(len_validation), range(size_view_shape))
        rgb_mapping_train = np.ix_(range(len_train), range(size_view_shape, size_view_shape + size_view_rgb))
        rgb_mapping_validation = np.ix_(range(len_validation), range(size_view_shape, size_view_shape + size_view_rgb)) 

        knn_rgb = KnnClassifier(X_train[rgb_mapping_train], Y_train, K)
        knn_shape = KnnClassifier(X_train[shape_mapping_train], Y_train, K)
        new_accuracy_rgb = knn_rgb.evaluate(X_validation[rgb_mapping_validation], Y_validation)
        new_accuracy_shape = knn_shape.evaluate(X_validation[shape_mapping_validation], Y_validation)
        mean_accuracies += 0.5*(new_accuracy_rgb + new_accuracy_shape)
        print 'New accuracy_rgb', new_accuracy_rgb
        print 'New accuracy_shape', new_accuracy_shape

    mean_accuracies/=10
    print 'New mean Accuracy', mean_accuracies
    accuracies.append([K, mean_accuracies])

out = open('result_pickles/knn_accuracies_for_k.pickle', 'wb')
pickle.dump(accuracies, out)
out.close()

plt.xlabel('valor de K')
plt.ylabel('acuracia media')
x = [a for [a,b] in accuracies]
y = [b for [a,b] in accuracies]
plt.plot(x,y)
plt.show()
