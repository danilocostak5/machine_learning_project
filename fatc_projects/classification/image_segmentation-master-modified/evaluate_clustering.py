import pandas as pd
import numpy as np
import math as m
import pickle
import warnings

from copy import deepcopy
from classifiers.ClusterMaker import *

#ignore warnings
warnings.filterwarnings('ignore')

#Begin


cm = ClusterMaker('data/segmentation.test.txt', 7, 3, 1.6, 1, 100, read_files=False)
best_cost = 10000000000000
best_cm = []
rand_for_best = 0
all_rands = []
all_costs = []

other_U = pickle.load(open('result_pickles/apriori_U.pickle', 'rb'))

for i in xrange(100):
    print 'Running clustering for the ', i+1, 'time -----------------------'
    cm.run_clustering()
    #Compute rand index
    new_rand = cm.adjusted_rand_index(other_U)
    print 'Rand',i, new_rand
    all_rands.append(new_rand)
    new_cost = cm.cost_evolution[-1]
    all_costs.append(new_cost)
    if new_cost < best_cost:
        best_cost = new_cost
        best_cm = deepcopy(cm)
        rand_for_best = new_rand
    print 'Best so far Cost:', best_cost,'Rand:', rand_for_best
        
print 'Auto saving best data'
out_cm = open('result_pickles/best_cm.pickle', 'wb')
out_costs = open('result_pickles/all_costs.pickle', 'wb')
out_rands = open('result_pickles/all_rands.pickle', 'wb')

pickle.dump(best_cm, out_cm)
pickle.dump(all_costs, out_costs)
pickle.dump(all_rands, out_rands)

out_cm.close()
out_costs.close()
out_rands.close()
