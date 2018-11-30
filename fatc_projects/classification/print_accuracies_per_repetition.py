# -*- coding: utf-8 -*-
import pickle
import numpy as np

# ESCOLHA NA PASTA resultados_modelos O ARQUIVO DE ACURACIAS QUE DESEJA AVALIAR (COM OU SEM NORMALIZACAO)
all_accuracies = pickle.load(open('resultados_modelos/accuracies_all_with_scale.pickle', 'rb'))

# lista de resultados por classificador
bayes_complete = ("Bayes Complete", np.array([a['accuracies_bayes_complete'] for a in all_accuracies]))
bayes_rgb = ("Bayes RGB", np.array([a['accuracies_bayes_rgb'] for a in all_accuracies]))
bayes_shape = ("Bayes SHAPE", np.array([a['accuracies_bayes_shape'] for a in all_accuracies]))
knn_complete = ("KNN Complete", np.array([a['accuracies_knn_complete'] for a in all_accuracies]))
knn_rgb = ("KNN RGB", np.array([a['accuracies_knn_rgb'] for a in all_accuracies]))
knn_shape = ("KNN Shape", np.array([a['accuracies_knn_shape'] for a in all_accuracies]))
max_rule = ("Regra do Máximo", np.array([a['accuracies_max_rule'] for a in all_accuracies]))

# ESCOLHA O CLASSIFICADOR QUE DESEJA AVALIAR DAS OPCOES ACIMA
classificador_selecionado = max_rule

print("Resultados (30 repetições)")
print("Classificador: {}".format(classificador_selecionado[0]))
classificador_selecionado_resultados = list(classificador_selecionado[1].mean(axis=1))
classificador_selecionado_resultados = map(lambda i:"{:0.2f}".format(i*100), classificador_selecionado_resultados)
for i, acc in enumerate(classificador_selecionado_resultados):
    print("{},{}".format(i+1, acc))