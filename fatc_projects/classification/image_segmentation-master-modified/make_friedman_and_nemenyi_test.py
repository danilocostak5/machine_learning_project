import pickle
import numpy as np

all_accuracies = pickle.load(open('result_pickles/accuracies_all.pickle', 'rb'))
accuracies_together = []

tam_test = len(all_accuracies)*len(all_accuracies[0]['accuracies_bayes_rgb'])
num_classifiers = len(all_accuracies[0])
ranks_matrix = []


def find_index(array_var, title):
    for i, e in enumerate(array_var):
        if e[0] == title:
            return i + 1
    return -1

for a in all_accuracies:
    size_k_fold = len(a['accuracies_bayes_rgb'])
    for i in xrange(size_k_fold):
        bayes_rgb = a['accuracies_bayes_rgb'][i]
        bayes_shape = a['accuracies_bayes_shape'][i]
        knn_rgb = a['accuracies_knn_rgb'][i]
        knn_shape = a['accuracies_knn_shape'][i]
        majority = a['accuracies_majority'][i]
        results_coupled = [['b_rgb', bayes_rgb], ['b_shape', bayes_shape],
                           ['knn_rgb', knn_rgb], ['knn_shape', knn_shape],
                           ['majority', majority]]
        rs = sorted(results_coupled, key= lambda x: x[1], reverse=True)
        new_line = [find_index(rs, 'b_rgb'), find_index(rs, 'b_shape'),
                    find_index(rs, 'knn_rgb'), find_index(rs, 'knn_shape'),
                    find_index(rs, 'majority')]
        ranks_matrix.append(new_line)

ranks_np = np.array(ranks_matrix)
mean_ranks = np.mean(ranks_np, axis=0)

k = float(num_classifiers)

sum_squared_ranks = float(np.sum(pow(mean_ranks, 2)))

chi_square = ((12.0*tam_test)/(k*(k+1)))*(sum_squared_ranks - k*pow(k+1,2)/4.0)
statistic = ((tam_test - 1)*chi_square)/(tam_test*(k-1) - chi_square)
print 'tam_test', tam_test
print 'k', k
print 'chi_square', chi_square
print 'Statistic', statistic

compare_val = 2.7858

if statistic > compare_val:
    print 'The classifiers are different'
else:
    print 'The classifiers are equal'

#Nemenyi test
q0 = 2.728
critical_val = q0*np.sqrt(k*(k+1)/(6*float(tam_test)))
print 'Critical values for Nemenyi test', critical_val

name_of_classifiers = ['bayes rgb', 'bayes shape', 'knn rgb', 'knn shape', 'majority']
for i, n1 in enumerate(name_of_classifiers):
    for j , n2 in enumerate(name_of_classifiers):
        if i != j:
            print 'Is', n1, 'different from', n2, '?', abs(mean_ranks[i] - mean_ranks[j]) > critical_val, 'diff', abs(mean_ranks[i] - mean_ranks[j])
