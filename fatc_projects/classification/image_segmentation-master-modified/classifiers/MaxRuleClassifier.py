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

class MaxRuleClassifier:
    # def __init__(self, X, Y, classes):
    #     [X_rgb, X_shape] = MaxRuleClassifier.separate_views(X)
    #     self.X_rgb = X_rgb
    #     self.X_shape = X_shape
    #     self.classes = classes
    #     self.Y = Y

    #     self.bayes_rgb = BayesClassifier(self.X_rgb, self.Y, self.classes)
    #     self.bayes_shape = BayesClassifier(self.X_shape, self.Y, self.classes)
    #     self.knn_rgb = KnnClassifier(self.X_rgb, self.Y, 1)
    #     self.knn_shape = KnnClassifier(self.X_shape, self.Y, 1)

    # recebe uma lista com as probabilidades posteriores dos classificadores guassiano e knn para cada visao
    def calcular_regra_max(self, p_gauss_v1, p_gauss_v2, p_gauss_v3, p_knn_v1, p_knn_v2, p_knn_v3, priori):
        posteriori_max = []
        for i in range(len(p_gauss_v1)):
            p_max_per_class = []
            for j in range(len(p_gauss_v1[i])):
                p_max_per_class.append(((1-3)*priori[j]) + 3 * max([p_gauss_v1[i][j], p_gauss_v2[i][j], p_gauss_v3[i][j], p_knn_v1[i][j], p_knn_v2[i][j], p_knn_v3[i][j]]))
            posteriori_max.append(p_max_per_class)
        return np.asarray(posteriori_max)

    def calcular_regra_max_visao(self, p_gauss, p_knn):
        posteriori_max = []
        for i in range(len(p_gauss)):
            p_max = []
            for j in range(len(p_gauss[i])):
                p_max.append(max(p_gauss[i][j],p_knn[i][j]))
            posteriori_max.append(p_max)
        return np.asarray(posteriori_max)

    def predict(self, x):
        [x_rgb, x_shape] = MaxRuleClassifier.separate_views(x)
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
