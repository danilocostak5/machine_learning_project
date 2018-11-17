# -*- coding:utf-8 -*-

import time
import numpy as np
import pandas as pd
from kcm import KCM_F_GH
from load_data import Dataset
from sklearn.metrics.cluster import adjusted_rand_score


# calcula o indice de Rand ajustado
# gera um vetor com os índices do cluster e um vetor com os rótulos de classe
def rand_score(sef, X, y, clusters):
    labels_pred = []
    for k in X.index:
        for i in range(len(clusters)):
            if (k in clusters[i]) == True:
                labels_pred.append(i)
                break
    return adjusted_rand_score(y, labels_pred)

if __name__ == '__main__':
    datadir='segmentation_2.test'

    df = pd.read_csv(datadir, sep=',')
    mydata = Dataset()
    mydata.load(df, 'all')
    kcm = KCM_F_GH(c=7, p=mydata.X.values.shape[1], data=mydata)

    # Variáveis para armazenar resultado da execução do algoritmo
    res_obj_function = []
    res_cluster = []
    res_hp = []

    # executar o algoritmo 100x
    for epoch in range(5):
        start_total_time = time.time()
        # inicialização do algoritmo
        kcm.initialization()
        # parte iterativa do algoritmo
        iteration = 0  # controlar a quantidade de iterações dentro do loop interno
        test = 1  # variável para controlar a mudança de cluster
        while (test != 0):
            iteration += 1
            # step 2
            kcm.update_hiperparams()
            # step 1
            kcm.calc_kernel_dist_matrix()
            # step 3
            test = kcm.allocation()
            # calcula a função objetivo para o cluster gerado
            obj_function = kcm.objective_function()
            # calcula a quantidade de objetos em cada cluster
            len_clusters = []
            for k in range(len(kcm.clusters)):
                len_clusters.append(len(kcm.clusters[k]))
            print('Epoch: ', epoch, ' | Iteration: ', iteration, '| Changes: ', test, '| : ', len_clusters,
                  ' | J-KCM_F_GH : ', obj_function, ' | ', (time.time() - start_total_time), 'seconds')
        # ao sair do loop, adiciona no vetor o valor da função objetivo, cluster, hyperarametros e o protótipo
        res_obj_function.append(obj_function)
        res_cluster.append(kcm.clusters)
        res_hp.append(kcm.hp)

    print('####### Result #######')
    index = np.argmin(res_obj_function)
    print(index)
    print('Objective Fuction > ', res_obj_function[index])
    print('Clusters > ', res_cluster[index])
    for i in range(len(res_cluster[index])):
        print('Cluster[{}] has {} elements\n'.format(i, len(res_cluster[index][i])))
    print('Hiperparams > ', res_hp[index])
    score = rand_score(kcm.X, kcm.y, res_cluster[index])
    print('Adjusted Rand Score > ', score)
