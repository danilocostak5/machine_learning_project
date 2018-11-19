import pandas as pd
import pickle

def create_views():
    shape_columns=["REGION-CENTROID-COL", "REGION-CENTROID-ROW", "REGION-PIXEL-COUNT", "SHORT-LINE-DENSITY-5", "SHORT-LINE-DENSITY-2", "VEDGE-MEAN", "VEDGE-SD", "HEDGE-MEAN", "HEDGE-SD"]
    rgb_columns=["INTENSITY-MEAN", "RAWRED-MEAN", "RAWBLUE-MEAN", "RAWGREEN-MEAN", "EXRED-MEAN", "EXBLUE-MEAN", "EXGREEN-MEAN", "VALUE-MEAN", "SATURATION-MEAN","HUE-MEAN"]
    df =pd.read_csv("data/segmentation.test.txt", sep=",", header=2)
    print 'Saving shape view'
    shape_view = df[shape_columns].copy()
    persist(shape_view, "result_pickles/shape_view.pickle")
    print 'Saving rgb view'
    rgb_view = df[rgb_columns].copy()
    persist(rgb_view,"result_pickles/rgb_view.pickle")
    print 'Saving full view'
    full_view = df.copy()
    persist(full_view,"result_pickles/full_view.pickle")

def persist(to_persist, file_name):
    file = open(file_name,'wb')
    pickle.dump(to_persist,file)
    file.close()


create_views()
