# -*- coding:utf-8 -*-

import numpy as np
from copy import deepcopy
from sklearn.metrics.pairwise import euclidean_distances

# Espera receber dados = dados.iloc[:].values
# Calcula o valor de gama com base na distância euclidiana dos dados

class KCM_F_GH(object):
    def __init__(self, c, p, data):
        self.c = c
        self.p = p
        self.X, self.y = data.X.values, data.y
        self.gamma = 0.
        self.hp = None # variances vector
        self.clusters = None
        self.K_dist_mat = np.zeros((len(self.X), len(self.X))).astype(float) # all against all in the feature kernel space util matrix for a many computations
        self.obj_to_cluster = {} # has the id => cluster format
        self.part2 = None # For optimization stores the values of the second somatory in the objective function.

    def normalize(self):
        self.X = (self.X - self.X.min(axis=0)) / (self.X.max(axis=0) - self.X.min(axis=0))

    def calc_gamma(self):
        """ Calcula  o valor de gama de acordo com
        """
        dists = euclidean_distances(self.X, self.X, squared=True)  # calculando distância euclidiana ao quadrado
        list_dists = np.sort(dists[np.triu_indices(len(dists), 1)]) # transformando a matriz em uma lista e ordena os valores

        idx_quantil_1 = int((len(list_dists) * 0.1) - 1)  # calculando o índice do quantil 0.1
        idx_quantil_9 = int((len(list_dists) * 0.9) - 1)  # calculando o índice do quantil 0.9

        sigma = (list_dists[idx_quantil_1] + list_dists[idx_quantil_9]) / 2 # média do quantil 0.1 e 0.9
        self.gamma = (1/sigma)

    # inicializa o vetor de hiperparametros com o valor de gama
    def init_hiperparams(self): # the gaussian kernel's global vector of variances
        self.hp = np.array([self.gamma] * self.p)

    # inicializa c clusters vazios
    def init_clusters(self):
        self.clusters = []
        for i in range(self.c):
            self.clusters.append([])

    # calcula a variante KCM-F-GH para um objeto x, dado um protótipo p e um vetor de hiperparametros
    def calc_kernel(self, x, y): # Equação 9
        dist = (self.hp * np.power(x - y, 2)).sum()
        result = np.exp(-0.5 * dist)
        return result

    # calcula uma matriz com a distancia de todos contra todos usando o kernel
    def calc_kernel_dist_matrix(self):
        print("Updating kernel dists reference matriz...")
        for l, vl in enumerate(self.X):
            # if l % 100 == 0: print("l: {}".format(l))
            for r, vr in enumerate(self.X):
                if l != r:
                    self.K_dist_mat[l, r] = self.calc_kernel(vl, vr)

    # atribuição inicial dos objetos ao cluster conforme
    def initialization(self):
        print("Initializing...")
        self.normalize()
        self.calc_gamma()
        self.init_hiperparams()
        self.init_clusters()
        self.calc_kernel_dist_matrix()
        # print(np.where(self.K_dist_mat == 0 ))
        # exit()
        # inicializa o vetor de protótipos distintos randomicamente
        prot_indexes = np.random.choice(np.arange(len(self.X)), self.c, replace=False)

        for k in range(len(self.X)):
            results = []
            for i in prot_indexes:
                results.append(self.K_dist_mat[k, i])
            index_min_dist = np.argmin(results) # o objeto será atribuido ao cluster que possuir a menor distância
            # print(self.clusters)
            self.clusters[index_min_dist].append(k)
            self.obj_to_cluster[k] = index_min_dist

    # def __object_against_cluster_wsum(self, k, c):
    #     """Executa o somatorio poderado do objeto k contra todos em um cluster c da equação 24 com relação às features. Resultado é ponderado pela cardinalidade do cluster.
    #     Para atualizar a variância.
    #     """
    #     res = np.zeros(self.p)
    #     for l in self.clusters[c]:
    #         res += self.K_dist_mat[k][l] * np.power(self.X.iloc[k].values - self.X.iloc[l].values, 2)
    #     return (-2 * res) / len(self.clusters[c])

    def __all_against_all_cluster_wsum(self, c):
        """Executa o somatorio ponderado todos contra todos em um cluster c da equação 24 com relação às features.
        Resultado é ponderado pela cardinalidade do cluster.
        Para atualizar a variância.
        """
        res = np.zeros(self.p)
        for l in self.clusters[c]:
            for r in self.clusters[c]:
                res += self.K_dist_mat[l][r] * np.power(self.X[l] - self.X[r], 2)
        # return res / np.power(len(self.clusters[c]), 2)
        return res / len(self.clusters[c])

    # calcula o vetor de hyperparametros
    def update_hiperparams(self):
        """Calcula a equação 24 do artigo com a ajuda de duas funções auxiliares
        """
        print("updating sigmas...")
        pi_h_list = np.zeros(self.p)
        for i in range(self.c):
            pi_h_list += self.__all_against_all_cluster_wsum(i)
        num = np.power(pi_h_list.prod(axis=0), (1/self.p)) * self.gamma
        self.hp = num / pi_h_list

    def __object_against_cluster_sum(self, k, c):
        """Executa o somatorio do objeto k contra todos em um cluster c da equação 24 com relação às features.
        Resultado é ponderado pela cardinalidade do cluster.
        Para atualizar a variância.
        """
        res = 0.
        for l in self.clusters[c]:
            res += self.K_dist_mat[k][l]
        return (2 * res) / len(self.clusters[c])

    def __all_against_all_cluster_sum(self):
        """Executa o somatorio todos contra todos em um cluster c da equação 24 com relação às features. Resultado é ponderado pela cardinalidade do cluster ao quadrado.
        Para atualizar a variância.
        """
        res = np.zeros(self.c)
        for i in range(self.c):
            for l in self.clusters[i]:
                for r in self.clusters[i]:
                    res[i] += self.K_dist_mat[l][r]
            res[i] /= np.power(len(self.clusters[i]), 2)
        return res

    # Not used because is too low
    def __curr_cluster(self, k):
        for i in range(self.c):
            if k in self.clusters[i]:
                return i

    def ___min_index(self, k):
        """Retorna o índice do cluster com menor distância para o objeto k.
        Necessário para a definição do cluster na etapa de alocaçao.
        """
        dists = []
        for i in range(self.c):
            dists.append(1 - self.__object_against_cluster_sum(k, i) + self.part2[i])
        # print("Len dists: {}\nDists: {}".format(len(dists), dists))
        return np.argmin(dists)

    def allocation(self):
        """Atribui um objeto k a um cluster c dado o argumento mínimo.
        Retorna a quantidade de mudanças. Se igual a zero, o algoritmo convergiu.
        """
        print("allocating...")
        test = 0
        # Id to Cluster Mapping auxliar
        obj_to_cluster_aux = deepcopy(self.obj_to_cluster)
        # Cluster auxiliar
        aux_clusters = deepcopy(self.clusters)

        # Calulando a segunda parte do somatório que é fixo para toda a etapa de alocação
        # Só calcula na primeira vez, nas demais já tem calculado da função objetivo
        if self.part2 is None:
            self.part2 = self.__all_against_all_cluster_sum()

        for k, curr_indx in obj_to_cluster_aux.items():
            # if k % 100 == 0: print("K = {}".format(k))
            pred_idx = self.___min_index(k)
            # print('Argmin: {}'.format(pred_idx))
            # print('Clusters type: {} - Num Clusters: {}'.format(type(self.clusters), len(self.clusters)))
            # curr_indx = self.__curr_cluster(k) # Demora muito,
            if pred_idx != curr_indx:
                test += 1
                aux_clusters[curr_indx].remove(k)
                aux_clusters[pred_idx].append(k)
                self.obj_to_cluster[k] = pred_idx
        self.clusters = aux_clusters
        return test

    def objective_function(self):
        print("calculating J_kcm_f_gh...")
        result = 0.
        self.part2 = self.__all_against_all_cluster_sum()
        for i in range(self.c):
            for k in self.clusters[i]:
                result += 1 - self.__object_against_cluster_sum(k, i) + self.part2[i]
        return result

if __name__ == "__main__":
    import pandas as pd
    from load_data import Dataset


    datadir='/home/dcp2/Documents/AM/segmentation_2.test'
    df = pd.read_csv(datadir, sep=',')
    mydata = Dataset()
    mydata.load(df, 'rgb')
    kcm = KCM_F_GH(c=7, p=mydata.X.values.shape[1], data=mydata)
