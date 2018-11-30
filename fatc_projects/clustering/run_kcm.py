# -*- coding:utf-8 -*-

import os
import time
import pickle
import numpy as np
import pandas as pd
from kcm import KCM_F_GH
from load_data import Dataset
from sklearn.metrics.cluster import adjusted_rand_score


if __name__ == '__main__':

    datadir='../../data/segmentation_2.test'
    result_dir = '../../results/clustering'
    result_file = 'results'
    bresult_file = 'best_result'
    view = 'rgb'
    norm = True

    result_file = '{}{}_{}'.format(view,
                                    '_norm' if norm else '',
                                    result_file)
    bresult_file = '{}{}_{}'.format(view,
                                    '_norm' if norm else '',
                                    bresult_file)

    df = pd.read_csv(datadir, sep=',')
    mydata = Dataset()
    mydata.load(df, view)

    # Variáveis para armazenar resultado da execução do algoritmo
    res_obj_function = []
    res_cluster = []
    res_obj_to_cluster = []
    res_hp = []
    res_ari = [] # adjusted rand indexes list
    res_J = [] # uma lista com a melhor serie de convergência de J
    # executar o algoritmo 100x
    for epoch in range(5):
        start_total_time = time.time()
        # inicialização do algoritmo
        kcm = KCM_F_GH(c=7, p=mydata.X.shape[1], data=mydata, norm=norm)
        kcm.initialization()
        # parte iterativa do algoritmo
        iteration = 0  # controlar a quantidade de iterações dentro do loop interno
        test = 1  # variável para controlar a mudança de cluster
        obj_function_ant = np.float('inf')
        obj_function = kcm.objective_function()
        objectives_aux = []
        objectives_aux.append(obj_function)
        print("J-KCM_F_GH : {:.4f}".format(obj_function))
        while (test != 0 and np.abs(obj_function_ant - obj_function) > 0.25 and obj_function_ant > obj_function):
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
            objectives_aux.append(obj_function)
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
        res_J.append(objectives_aux)
        score = adjusted_rand_score(mydata.y, list(kcm.obj_to_cluster.values()))
        res_ari.append(score)

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    if os.path.exists(os.path.join(result_dir, result_file)):
        fid = os.popen("ls {} | grep {} | wc -l".format(result_dir, result_file)).read()[0]
        result_file = '{}{}.pkl'.format(result_file, fid)
        bresult_file = '{}{}.pkl'.format(bresult_file, fid)
    else:
        result_file = '{}.pkl'.format(result_file)
        bresult_file = '{}.pkl'.format(bresult_file)

    with open(os.path.join(result_dir, result_file), 'wb') as fpick:
        res_dict = {'res_obj_function':res_obj_function,
                    'res_cluster':res_cluster,
                    'res_obj_to_cluster':res_obj_to_cluster,
                    'res_hp':res_hp,
                    'res_J': res_J,
                    'res_ari':res_ari}
        pickle.dump(res_dict, fpick)

    with open(os.path.join(result_dir, bresult_file), 'wb') as fpick:
        index = np.argmin(res_obj_function)
        res_dict = {'res_obj_function':res_obj_function[index],
                    'res_cluster':res_cluster[index],
                    'res_obj_to_cluster':res_obj_to_cluster[index],
                    'res_hp':res_hp[index],
                    'res_J':res_J[index],
                    'ARI':res_ari[index]}
        pickle.dump(res_dict, fpick)

    print('####### Result #######')
    # index = np.argmin(res_obj_function)
    index = np.argmax(res_ari)
    print(index)
    print('Objective Fuction ==> ', res_obj_function[index])
    # print('Clusters > ', res_cluster[index])
    for i in range(len(res_cluster[index])):
        print('Cluster[{}] has {} elements\n'.format(i, len(res_cluster[index][i])))
    print('Hiperparams ==> ', res_hp[index])
    print('Adjusted Rand Score ==> ', res_ari[index])
