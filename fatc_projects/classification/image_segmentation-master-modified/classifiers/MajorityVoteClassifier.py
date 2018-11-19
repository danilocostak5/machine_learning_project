import pandas as pd
import numpy as np
import math as m
import pickle
import warnings
import random

#ignore warnings
warnings.filterwarnings('ignore')

from classifiers.KnnClassifier import *
from classifiers.BayesClassifier import *
from classifiers.DataVectorizer import *

class MajorityVoteClassifier:
    def __init__(self, X, Y, classes):
        [X_rgb, X_shape] = MajorityVoteClassifier.separate_views(X)
        self.X_rgb = X_rgb
        self.X_shape = X_shape
        self.classes = classes
        self.Y = Y

        self.bayes_rgb = BayesClassifier(self.X_rgb, self.Y, self.classes)
        self.bayes_shape = BayesClassifier(self.X_shape, self.Y, self.classes)
        self.knn_rgb = KnnClassifier(self.X_rgb, self.Y, 1)
        self.knn_shape = KnnClassifier(self.X_shape, self.Y, 1)

    def predict(self, x):
        [x_rgb, x_shape] = MajorityVoteClassifier.separate_views(x)
        predict_b_rgb = self.bayes_rgb.predict(x_rgb)
        predict_b_shape = self.bayes_shape.predict(x_shape)
        predict_knn_rgb = self.knn_rgb.predict(x_rgb)
        predict_knn_shape = self.knn_shape.predict(x_shape)

        result_dic = {}
        for (index, c) in enumerate(self.classes):
            result_dic[index] = 0

        result_dic[predict_b_rgb] += 1
        result_dic[predict_b_shape] += 1
        result_dic[predict_knn_rgb] += 1
        result_dic[predict_knn_shape] += 1
        votes = sorted(result_dic.items(), key=operator.itemgetter(1), reverse=True)
        return votes[0][0]

    def evaluate(self, X, Y):
        num_total = 0.0
        num_right = 0.0
        num_wrong = 0.0

        for (index, row) in enumerate(X):
            current_index = Y[index]
            
            predicted_index = self.predict(row)
            if predicted_index == current_index:
                num_right += 1.0
            else:
                num_wrong += 1.0
            num_total += 1.0

        return num_right/num_total

    @staticmethod
    def separate_views(X):
        size_view_1 = 9
        size_view_2 = 10
        num_rows = 0
        
        if len(np.shape(X)) != 2:
            num_rows = 1
            view_1 = X[:size_view_1]
            view_2 = X[size_view_1:size_view_1 + size_view_2]
        else:
            num_rows = np.shape(X)[0]
            view_1 = X[np.ix_(range(num_rows),range(size_view_1))]
            view_2 = X[np.ix_(range(num_rows),range(size_view_1, size_view_1 + size_view_2))]
        
        return [view_1, view_2]
