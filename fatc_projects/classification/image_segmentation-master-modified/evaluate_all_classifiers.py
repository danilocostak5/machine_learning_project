# By Felipe {fnba}, Paulo {pan2} e Debora {dca2}@cin.ufpe.br
from pandas import Index

import pandas as pd
import numpy as np
import csv
import pickle
import warnings
import random

from classifiers.DataVectorizer import *
from classifiers.ClassifierTester import *

print 'Testing full and separated views'
df = pd.read_csv('../segmentation_2.test', sep=',')

accuracies_full = []
for i in xrange(1, 30 + 1):
    print 'Repetition', i, 'of 30-------------'
    new_accuracies = ClassifierTester.make_n_fold_test(df, 10)
    accuracies_full.append(new_accuracies)

out_full = open('result_pickles/accuracies_all.pickle', 'wb')

pickle.dump(accuracies_full, out_full)

out_full.close()



# print 'Testing full view'
# dv = DataVectorizer(filename='result_pickles/full_view.pickle')
# accuracies_full = []
# for i in xrange(30):
#     print 'iteration', i, 'of 30-------------'
#     new_accuracies = ClassifierTester.make_n_fold_test(dv, 10)
#     accuracies_full.append(new_accuracies)

# out_full = open('result_pickles/accuracies_all.pickle', 'wb')

# pickle.dump(accuracies_full, out_full)

# out_full.close()
