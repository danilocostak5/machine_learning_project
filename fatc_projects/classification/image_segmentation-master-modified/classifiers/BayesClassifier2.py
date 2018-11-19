# https://deeplearningcourses.com/c/data-science-supervised-machine-learning-in-python
# https://www.udemy.com/data-science-supervised-machine-learning-in-python
# This is an example of a Bayes classifier on MNIST data.
from __future__ import print_function, division
from future.utils import iteritems
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future


import numpy as np
import matplotlib.pyplot as plt
# from util import get_data
from datetime import datetime
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvn

class NaiveBayes:
    def fit(self, X, Y, smoothing=1e-2):
        N, D = X.shape
        self.gaussians = dict()
        self.priors = dict()
        labels = set(Y)
        for c in labels:
            current_x = X[Y == c]
            self.gaussians[c] = {
                'mean': current_x.mean(axis=0),
                'var': current_x.var(axis=0) + smoothing,
            }
            self.priors[c] = float(len(Y[Y == c])) / len(Y)

    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)

    def predict(self, X):
        N, D = X.shape
        K = len(self.gaussians)
        P = np.zeros((N, K))
        normalization_term = []
        for c, g in iteritems(self.gaussians):
            mean, cov = g['mean'], g['var']
            P[:,c] = (mvn.pdf(X, mean=mean, cov=cov) *  self.priors[c])
            normalization_term.append(P[:,c])
        
        for i in range(N):
            P[i,:] = P[i,:] / np.sum(P[i])

        return P
