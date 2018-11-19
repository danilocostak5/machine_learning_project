import pandas as pd
import numpy as np
import warnings

# ignore warnings
warnings.filterwarnings('ignore')

class DataVectorizer:

    def __init__(self, pandas_data=None, filename=None):
        self.data_frame = None

        if not filename is None:
            self.get_from_pickle_pandas(filename)
        elif not pandas_data is None:
            self.data_frame = pandas_data

        if self.data_frame is None:
            raise ValueError('No input data defined')

        self.get_classes_from_data_frame()
        self.create_vectors()

    def get_from_pickle_pandas(self, filename):
        self.data_frame = pd.read_pickle(filename)

    def read_from_csv(self, filename):
        self.data_frame = pd.read_csv(filename, sep=",", header=2)

    def get_classes_from_data_frame(self):
        my_data = self.data_frame
        classes=[]
        for index, row in my_data.iterrows():
            classes.append(index)
        self.classes = sorted(set(classes))

    def create_vectors(self):
        X = []
        Y = []
        for (i, row) in self.data_frame.iterrows():
            X.append(row.values)
            current_class = self.classes.index(str(i))
            Y.append(current_class)
        self.X = np.array(X)
        self.Y = np.array(Y).ravel()
