import pickle
from classifiers.ClusterMaker import *
import matplotlib.pyplot as plt
import numpy as np

best_cm = pickle.load(open('result_pickles/best_cm.pickle', 'rb'))

len_cost = len(best_cm.cost_evolution)
derivatives = []
const_0 = []
for i in xrange(len_cost - 1):
    diff = best_cm.cost_evolution[i+1] - best_cm.cost_evolution[i]
    derivatives.append(diff)
    const_0.append(0)
    
plt.xlabel('iteracoes')
plt.ylabel('funcao de custo')
plt.plot(best_cm.cost_evolution)
plt.savefig('result_images/evolucao_custo.png')

plt.gcf().clear()

plt.xlabel('iteracoes')
plt.ylabel('derivada da funcao de custo')
plt.plot(derivatives)
plt.plot(const_0, 'r--')
plt.savefig('result_images/derivada_custo.png')

print 'prototypes'
print best_cm.G
print 'view relevancy'
print best_cm.Lambda

reprs = []
for i in xrange(7):
    reprs.append([])

hard_U = np.array(ClusterMaker.soft_to_hard_cluster(best_cm.U))
(num_elems, dims) = np.shape(hard_U)
for i in xrange(num_elems):
    maxindex = np.argmax(hard_U[i])
    reprs[maxindex].append(i)

for i in xrange(7):
    print '----cluster', i, '------'
    print reprs[i]
