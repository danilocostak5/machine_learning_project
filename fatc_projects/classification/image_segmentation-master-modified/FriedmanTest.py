# By Felipe {fnba}, Paulo {pan2} e Debora {dca2}@cin.ufpe.br
#Chi_square table used
# https://www.medcalc.org/manual/chi-square-table.php

import pandas as pd
import numpy as np
import warnings

# ignore warnings
warnings.filterwarnings('ignore')


class FriedmanTest:

    def __init__(self, bayes_result=None, knn_result=None, majority_result=None):
        self.data_frame = None
        self.bayes_result = bayes_result
        self.knn_result = knn_result
        self.majority_result = majority_result
        self.n_rows = None

        if bayes_result is None:
            raise ValueError('No result for Bayes Classifier defined')
        if knn_result is None:
            raise ValueError('No result for KNN Classifier defined')
        if majority_result is None:
            raise ValueError('No result for Majority Vote Classifier defined')

        self.read_from_csv('chi_square.csv')

    def read_from_csv(self, filename):
        print 'Reading Chi-Square table...'
        self.data_frame = pd.read_csv(filename, sep=",", header=0)

    @staticmethod
    def get_xr_pow2(rank_tmp):
        # n = number_of_rows
        n = rank_tmp.__len__()

        # k = number_of_columns
        k = rank_tmp[0].__len__()

        RjPow2 = 0
        xr_pow_2 = 0

        # Rj = sum of the results per columns,
        # in this case we've only one value for each group/column
        for r in xrange(n):
            for j in xrange(k):
                value = rank_tmp[r - 1][j]
                RjPow2 += pow(value, 2)

        # xr_pow_2 = (12/n * k*(k+1)*[RjPow2-3*n(k+1)]
        calc_part_1 = n * k * (k + 1)
        calc_part_1 = 12 / calc_part_1
        calc_part_2 = (3 * n) * (k + 1)
        calc_part_2 = RjPow2 - calc_part_2
        xr_pow_2 = calc_part_1 * calc_part_2

        if xr_pow_2 == 0:
            xr_pow_2 = 0.000000000001

        xr_pow_2 = xr_pow_2/100

        return xr_pow_2

    def convert_to_decimal(self, value):
        if value == 0:
            value = 0.00000000001
        else:
            value = value / 100

        return value

    def evaluate_classifiers(self, n_rows, x_pow_2):
        df = self.data_frame
        arr = np.array(df)
        if n_rows == 1:
            print 'Defining search for the default value of N'
            n_rows -= 1

        #aproves if the classifiers have used the same data source
        row_counter = 0
        for row in arr:
            if row_counter == n_rows:
                max_value = np.max(row)
                min_value = np.min(row)
                if min_value <= x_pow_2 <= max_value:
                    is_approved = True
                    return is_approved
            row_counter += 1


    def friedman_test(self):
        bayes = self.bayes_result
        knn = self.knn_result
        majority = self.majority_result

        classifiers = ['Bayes', 'Knn', 'Majority']
        table_of_values = np.zeros(shape=(1, classifiers.__len__()))
        for x in xrange(table_of_values.__len__()):
            table_of_values[x][0] = bayes
            table_of_values[x][1] = knn
            table_of_values[x][2] = majority

        rank_tmp = np.zeros(shape=(1, 3))

        for c in table_of_values:
            max_value = np.max(c)
            min_value = np.min(c)
            for value in c:
                for i in xrange(rank_tmp.__len__()):
                    if value == max_value:
                        if rank_tmp[0][i] == 0:
                            rank_tmp[0][i] = classifiers.__len__()
                        else:
                            rank_tmp[0][i + 1] = classifiers.__len__()
                    if value == min_value:
                        if rank_tmp[0][i] == 0:
                            rank_tmp[0][i] = classifiers.__len__() - 2
                        else:
                            rank_tmp[0][i + 1] = classifiers.__len__() - 2
                    else:
                        if rank_tmp[0][i] == 0:
                            rank_tmp[0][i] = classifiers.__len__() - 1
                        else:
                            if rank_tmp[0][i + 1] == 0:
                                rank_tmp[0][i + 1] = classifiers.__len__() - 1
                            else:
                                rank_tmp[0][i + 2] = classifiers.__len__() - 1

        sum_weights = 0

        for lk in xrange(rank_tmp.__len__()):
            for c in xrange(rank_tmp[lk].__len__()):
                sum_weights += rank_tmp[lk][c]
                rank_tmp[lk][c] = sum_weights
                sum_weights = 0

        x_r_pow_2 = self.get_xr_pow2(rank_tmp)

        #get number of rows
        n = rank_tmp.__len__()

        is_approved = self.evaluate_classifiers(n, x_r_pow_2)

        print 'Defining Null Hyphotesis...'
        print 'H_0: The classifiers are equivalent'
        print 'H_1: The classifiers are not equivalent'
        print 'The chi-square from xr_pow_2 = (12/n * k*(k+1)*[RjPow2-3*n(k+1)]  is ', x_r_pow_2
        print 'The number of groups is ', n
        print 'The number of tests is ', classifiers.__len__()
        print 'For n = ', n, 'then H_0 is acceptance is ', is_approved

        print 'Done'

# ---------------

# Begin

ft = FriedmanTest(bayes_result=0.79, knn_result=0.89, majority_result=0.88)
ft.friedman_test()

