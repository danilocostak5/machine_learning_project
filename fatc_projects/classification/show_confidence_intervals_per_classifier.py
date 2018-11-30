# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['axes.labelsize'] = 'large'
plt.rcParams["axes.labelweight"] = "bold"

import pickle
import numpy as np

# ESCOLHA NA PASTA resultados_modelos O ARQUIVO DE ACURACIAS QUE DESEJA AVALIAR (COM OU SEM NORMALIZACAO)
todas_acuracias = pickle.load(open('resultados_modelos/accuracies_all_with_scale.pickle', 'rb'))
bayes_complete = np.array([a['accuracies_bayes_complete'] for a in todas_acuracias]).ravel()
bayes_rgb = np.array([a['accuracies_bayes_rgb'] for a in todas_acuracias]).ravel()
bayes_shape = np.array([a['accuracies_bayes_shape'] for a in todas_acuracias]).ravel()
knn_complete = np.array([a['accuracies_knn_complete'] for a in todas_acuracias]).ravel()
knn_rgb = np.array([a['accuracies_knn_rgb'] for a in todas_acuracias]).ravel()
knn_shape = np.array([a['accuracies_knn_shape'] for a in todas_acuracias]).ravel()
max_rule = np.array([a['accuracies_max_rule'] for a in todas_acuracias]).ravel()

def create_IC_interval(np_array):
    t_val = 1.96
    len_array = len(np_array)
    mean_array = np.mean(np_array)
    std_array = np.std(np_array, ddof=1)
    int_lower_array = mean_array - t_val*std_array/np.sqrt(len_array)
    int_high_array = mean_array + t_val*std_array/np.sqrt(len_array)
    return [mean_array, std_array, int_lower_array, int_high_array]

print ("Bayes Complete")
ic_bayes_complete = create_IC_interval(bayes_complete)
print 'Mean: ', ic_bayes_complete[0]
print 'Std: ', ic_bayes_complete[1]
print 'IC 95: [', ic_bayes_complete[2], ',' , ic_bayes_complete[3], ']\n'

print ("Bayes RGB")
ic_bayes_rgb = create_IC_interval(bayes_rgb)
print 'Mean: ', ic_bayes_rgb[0]
print 'Std: ', ic_bayes_rgb[1]
print 'IC 95: [', ic_bayes_rgb[2], ',' , ic_bayes_rgb[3], ']\n'

print ("Bayes Shape")
ic_bayes_shape = create_IC_interval(bayes_shape)
print 'SHAPE: '
print 'Mean: ', ic_bayes_shape[0]
print 'Std', ic_bayes_shape[1]
print 'IC 95 [', ic_bayes_shape[2], ',' , ic_bayes_shape[3], ']\n'

print("KNN complete")
ic_knn_complete = create_IC_interval(knn_complete)
print 'Mean: ', ic_knn_complete[0]
print 'Std: ', ic_knn_complete[1]
print 'IC 95 [', ic_knn_complete[2], ',' , ic_knn_complete[3], ']\n'

print("KNN RGB")
ic_knn_rgb = create_IC_interval(knn_rgb)
print 'Mean: ', ic_knn_rgb[0]
print 'Std: ', ic_knn_rgb[1]
print 'IC 95: [', ic_knn_rgb[2], ',' , ic_knn_rgb[3], ']\n'

print("KNN shape")
ic_knn_shape = create_IC_interval(knn_shape)
print 'Mean: ', ic_knn_shape[0]
print 'Std: ', ic_knn_shape[1]
print 'IC 95: [', ic_knn_shape[2], ',' , ic_knn_shape[3], ']\n'

print 'MAX RULE CLASSIFIERS -----------'
ic_max_rule = create_IC_interval(max_rule)
print 'Mean: ', ic_max_rule[0]
print 'Std: ', ic_max_rule[1]
print 'IC 95: [', ic_max_rule[2], ',' , ic_max_rule[3], ']\n'


print 'Salvando histogramas'
plt.xlabel('Naive Bayes View: Complete')
plt.ylabel('Frequencia Amostral')
plt.hist(bayes_complete)
plt.savefig('resultados_imagens/bayes_accuracies_complete.png')
plt.gcf().clear()

plt.xlabel('Naive Bayes View: RGB')
plt.ylabel('Frequencia Amostral')
plt.hist(bayes_rgb)
plt.savefig('resultados_imagens/bayes_accuracies_rgb.png')
plt.gcf().clear()

plt.xlabel('Naive Bayes View: SHAPE')
plt.ylabel('Frequencia Amostral')
plt.hist(bayes_shape)
plt.savefig('resultados_imagens/bayes_accuracies_shape.png')
plt.gcf().clear()

plt.xlabel('KNN View: Complete')
plt.ylabel('Frequencia Amostral')
plt.hist(knn_complete)
plt.savefig('resultados_imagens/knn_accuracies_complete.png')
plt.gcf().clear()

plt.xlabel('KNN View:  RGB')
plt.ylabel('Frequencia Amostral')
plt.hist(knn_rgb)
plt.savefig('resultados_imagens/knn_accuracies_rgb.png')
plt.gcf().clear()

plt.xlabel('KNN View:  SHAPE', fontname="DejaVu Sans Mono")
plt.ylabel('Frequencia Amostral')
plt.hist(knn_shape)
plt.savefig('resultados_imagens/knn_accuracies_shape.png')
plt.gcf().clear()

plt.xlabel('Regra do Maximo')
plt.ylabel('Fequencia Amostral')
plt.hist(max_rule)
plt.savefig('resultados_imagens/max_rule.png')
plt.gcf().clear()