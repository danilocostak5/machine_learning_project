# -*- coding:utf-8 -*-

import time
import numpy as np
import pandas as pd
from kcm import KCM_F_GH
from load_data import Dataset
from sklearn.metrics.cluster import adjusted_rand_score


if __name__ == '__main__':
    datadir='segmentation_2.test'

    df = pd.read_csv(datadir, sep=',')
    mydata = Dataset()
    mydata.load(df, 'rgb')

    # Variáveis para armazenar resultado da execução do algoritmo
    res_obj_function = []
    res_cluster = []
    res_obj_to_cluster = []
    res_hp = []
    # executar o algoritmo 100x
    for epoch in range(1):
        start_total_time = time.time()
        # inicialização do algoritmo
        kcm = KCM_F_GH(c=7, p=mydata.X.shape[1], data=mydata)
        kcm.initialization()
        # parte iterativa do algoritmo
        iteration = 0  # controlar a quantidade de iterações dentro do loop interno
        test = 1  # variável para controlar a mudança de cluster
        obj_function_ant = np.float('inf')
        obj_function = kcm.objective_function()
        print("J-KCM_F_GH : {:.4f}".format(obj_function))
        while (test != 0 and np.abs(obj_function_ant - obj_function) > 0.25):
            iteration += 1
            obj_function_ant = obj_function
            # step 2
            kcm.update_hiperparams()
            # step 1
            # kcm.calc_kernel_dist_matrix()
            # step 3
            test = kcm.allocation()
            # calcula a função objetivo para o cluster gerado
            obj_function = kcm.objective_function()
            # calcula a quantidade de objetos em cada cluster
            len_clusters = []
            for k in range(len(kcm.clusters)):
                len_clusters.append(len(kcm.clusters[k]))
            print('Epoch: {:3d} | Iteration: {:2d}'
                  ' | Changes: {:4d}| : {} | J-KCM_F_GH : {:.4f} | {} seconds'.
                  format(epoch, iteration, test, len_clusters, obj_function,
                        (time.time() - start_total_time)))
        # ao sair do loop, adiciona no vetor o valor da função objetivo, cluster, hyperarametros e o protótipo
        res_obj_function.append(obj_function)
        res_cluster.append(kcm.clusters)
        res_obj_to_cluster.append(kcm.obj_to_cluster)
        res_hp.append(kcm.hp)

    print('####### Result #######')
    index = np.argmin(res_obj_function)
    print(index)
    print('Objective Fuction > ', res_obj_function[index])
    # print('Clusters > ', res_cluster[index])
    for i in range(len(res_cluster[index])):
        print('Cluster[{}] has {} elements\n'.format(i, len(res_cluster[index][i])))
    print('Hiperparams ==> ', res_hp[index])
    # score = rand_score(kcm.X, kcm.y, res_cluster[index])
    score = adjusted_rand_score(mydata.y, list(res_obj_to_cluster[index].values()))
    print('Adjusted Rand Score > ', score)
