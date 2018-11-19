import numpy as np
import pandas as pd
import pickle
import warnings

# ignore warnings
warnings.filterwarnings('ignore')

class BayesClassifier:
    def __init__(self, X, Y, classes):
        self.classes = classes
        self.X = X
        self.Y = Y

        self.get_w_frequency()
        self.calculate_prob_diss_classes()


    @staticmethod
    def cvt_np_array(matrix):
        if type(matrix) != np.ndarray:
            matrix = np.array(matrix)
        return matrix

    #checks if the named file exists
    def gets_w_classes(self,file_w_class_name):
        file = open(file_w_class_name,'wb')
        if file is not None:
            return True
        else:
            return False

    def get_w_frequency(self):
        my_classes = self.classes
        num_classes = len(my_classes)
        num_elems = np.shape(self.X)[0]

        self.apriori = np.zeros((num_classes, 1))

        for index, row in enumerate(self.X):
            current_index = self.Y[index]
            self.apriori[current_index] += 1

        self.apriori /= num_elems

    def calculate_prob_diss_classes(self):
        my_classes = self.classes
        num_classes = len(my_classes)

        element_size = np.shape(self.X)[1]

        self.centers = np.zeros((num_classes, element_size))
        self.num_for_class = np.zeros((num_classes, 1))
        self.diags = np.zeros((num_classes, element_size))

        #Compute centers
        for index, row in enumerate(self.X):
            current_index = self.Y[index]
            self.centers[current_index] += row
            self.num_for_class[current_index] += 1
        self.centers /= self.num_for_class

        #Compute variance matrix diagonals
        for index, row in enumerate(self.X):
            current_index = self.Y[index]
            current_center = self.centers[current_index]
            self.diags[current_index] += pow(row - current_center, 2)

        self.diags /= self.num_for_class

        for c in xrange(num_classes):
            for d in xrange(element_size):
                if self.diags[c][d] == 0:
                    self.diags[c][d] = 0.0000000001 #Avoid zero divisions

    def p_x_w(self, x, index):
        d = len(x)
        center = self.centers[index]
        variance_diag = self.diags[index]
        prod = np.prod(variance_diag)
        diff_2 = pow(x - center,2)
        diff_2_div = diff_2/variance_diag
        internal_sum = np.sum(diff_2_div)
        return pow(2*np.pi, -0.5*d)*pow(prod, -0.5)*np.exp((-0.5)*internal_sum)

    def p_w_x(self, x, index):
        probs = np.array([self.p_x_w(x, i) for (i, c) in enumerate(self.classes)])
        return probs[index]*self.apriori[index]/np.sum(probs)

    def predict(self, x):
        probs = np.array([self.p_w_x(x, index) for (index, c) in enumerate(self.classes)])
        max_index = np.argmax(probs)
        return max_index
    
    def predict_proba(self, x):
        probs = np.array([self.p_w_x(x, index) for (index, c) in enumerate(self.classes)])
        return probs

    def evaluate(self, X, Y):
        num_total = 0.0
        num_right = 0.0
        num_wrong = 0.0

        for (index, row) in enumerate(X):
            current_index = Y[index]
            row_values = row
            predicted_index = self.predict(row_values)
            if predicted_index == current_index:
                num_right += 1.0
            else:
                num_wrong += 1.0
            num_total += 1.0

        return num_right/num_total
    
    def compute_posteriors(self, X, Y):
        
        posteriors = np.zeros((X.shape[0], len(self.classes)))

        for (index, row) in enumerate(X):
            current_index = Y[index]
            row_values = row
            posteriors[index] = self.predict_proba(row_values)[:,0]
            print(sum(posteriors[index]))
            exit()
        return posteriors