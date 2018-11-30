# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':

    pathdir = '../../results/clustering'
    fname = 'all_norm_best_result2.pkl'

    imdir = '../../images/clustering'
    imname = 'convergence2.png'

    resdict = None
    with open(os.path.join(pathdir, fname), 'rb') as fpick:
        resdict = pickle.load(fpick)

    vJ = resdict['res_J']
    title = 'Convergência do KCM-F-GH'
    ylabel = 'Função Objetivo'
    xlabel = 'Iterações'

    print(vJ)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.plot(vJ[:6])

    if not os.path.exists(imdir):
        os.makedirs(imdir)
    plt.savefig(os.path.join(imdir, imname), format='png')
