# -*- coding: utf-8 -*-
import pickle
import numpy as np

def main():
    execute_friedman_test()

def execute_friedman_test():
    ########################  TESTE DE FRIEDMAN #########################
    print("Iniciando teste de Friedman")

    # ESCOLHA O ARQUIVO COM AS ACURACIAS DENTRO DA PASTA "resultados_modelos" - COM OU SEM NORMALIZACAO
    todas_acuracias = pickle.load(open('resultados_modelos/accuracies_all_with_scale.pickle', 'rb'))
    acuracias_juntas = []

    tam_test = len(todas_acuracias)*len(todas_acuracias[0]['accuracies_bayes_rgb'])
    nb_classificadores = len(todas_acuracias[0])

    ranks_matrix = []
    for a in todas_acuracias:
        size_k_fold = len(a['accuracies_bayes_rgb'])
        for i in xrange(size_k_fold):
            bayes_complete = a['accuracies_bayes_complete'][i]
            bayes_rgb = a['accuracies_bayes_rgb'][i]
            bayes_shape = a['accuracies_bayes_shape'][i]
            knn_complete = a['accuracies_knn_complete'][i]
            knn_rgb = a['accuracies_knn_rgb'][i]
            knn_shape = a['accuracies_knn_shape'][i]
            max_rule = a['accuracies_max_rule'][i]
            resultados_conjunto = [
                ['b_complete', bayes_complete], ['b_rgb', bayes_rgb], ['b_shape', bayes_shape],
                ['knn_complete', knn_complete], ['knn_rgb', knn_rgb], ['knn_shape', knn_shape],
                ['max_rule', max_rule]
                ]
            rs = sorted(resultados_conjunto, key= lambda x: x[1], reverse=True)
            new_line = [
                obter_indice(rs, 'b_complete'), obter_indice(rs, 'b_rgb'), obter_indice(rs, 'b_shape'),
                obter_indice(rs, 'knn_complete'), obter_indice(rs, 'knn_rgb'), obter_indice(rs, 'knn_shape'),
                obter_indice(rs, 'max_rule')
            ]
            ranks_matrix.append(new_line)

    ranks_np = np.array(ranks_matrix)
    mean_ranks = np.mean(ranks_np, axis=0)
    k = float(nb_classificadores)

    sum_squared_ranks = float(np.sum(pow(mean_ranks, 2)))

    chi_square = ((12.0*tam_test)/(k*(k+1)))*(sum_squared_ranks - k*pow(k+1,2)/4.0)
    estatistica = ((tam_test - 1)*chi_square)/(tam_test*(k-1) - chi_square)

    print ('tam_test: {}'.format(tam_test))
    print ('k (nº de classificadores): {}'.format(k))
    print ('Estatística computada: {}'.format(estatistica))

    compare_val = 2.4082 # http://www.socr.ucla.edu/applets.dir/f_table.html (valor critico para (7-1) e (2100-1)*(7-1) graus de liberdade)

    if estatistica > compare_val:
        print('Resultado do Teste de Friedman: os classificadores são diferentes')
        execute_nemenyi_test(mean_ranks, k, tam_test)
    else:
        print('Resultado do Teste de Friedman: os classificadores são iguais')

def execute_nemenyi_test(mean_ranks, k, tam_test):
    ########################  TESTE DE NEMENYI #########################

    print("\n\nIniciando teste de Nemenyi")
    q0 = 2.948 # sete modelos a serem comparados
    critical_val = q0*np.sqrt(k*(k+1)/(6*float(tam_test)))
    print ('Diferença crítica para o teste de Nemenyi: {}\n'.format(critical_val))

    name_of_classifiers = ['bayes_complete', 'bayes rgb', 'bayes shape', 'knn_complete', 'knn rgb', 'knn shape', 'max_rule']
    for i, n1 in enumerate(name_of_classifiers):
        for j , n2 in enumerate(name_of_classifiers):
            if i != j:
                diferenca_computada = abs(mean_ranks[i] - mean_ranks[j])
                diferenca_significativa = "Sim" if diferenca_computada > critical_val else "Não"
                print("Comparando os modelos '{}' e '{}'... Diferença computada: {}  |  Diferença Significativa? {}".format(n1, n2, np.round(diferenca_computada, 2), diferenca_significativa))

def obter_indice(array_var, title):
    for i, e in enumerate(array_var):
        if e[0] == title:
            return i + 1
    return -1

if __name__ == "__main__":
    main()