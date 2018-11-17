# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import csv

TARGET = {'GRASS':0,
          'PATH':1,
          'WINDOW':2,
          'CEMENT':3,
          'FOLIAGE':4,
          'SKY':5,
          'BRICKFACE':6}

class Dataset(object):
    def __init__(self):
        # self.samples = []
        self.X = None
        self.y = []

    def load(self, df, view='all'):
        if view == 'all':
            self.X = df.iloc[:, 1:]
        elif view == 'shape':
            self.X = df.iloc[:, 1:9]
        elif view == 'rgb':
            self.X = df.iloc[:, 9:]

        for label in df.iloc[:, 0:1].values:
            self.y.append(TARGET[label[0]])

        # for i in range(len(X)):
        #     self.samples.append(Sample(index=i, features=X[i], target=y[i]))

# class Sample(object):
#      def __init__(self, index, features, target):
#          self.index = index
#          self.values = features
#          self.target = target
#          self.cluster = -1


if __name__ == '__main__':

    datadir='segmentation_2.test'
    X, y = [], []

    with open(datadir) as csvfile:
        reader = csv.reader(csvfile, delimiter=',', skipinitialspace=True)
        next(reader) # Skip first row
        for line in reader:
            X.append(line[1:])
            y.append(line[0])
    X = np.asarray(X).astype(np.float)
    # X_norm = (X - X.min(0)) / (X.max(0) - X.min(0))
    y = np.asarray(y)

    df = pd.read_csv(datadir, sep=',')
    mydata = Dataset()
    mydata.load(df, 'rgb')
    print(np.random.choice(np.arange(10), 7, replace=False))
    # for s in mydata.samples:
    #     print("{:d} : {} -> {} \n".format(s.index, s.target, s.features))
    #     break
