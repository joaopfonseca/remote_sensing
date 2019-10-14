import pandas as pd
import numpy as np
from sklearn.metrics import calinski_harabasz_score
from minisom import MiniSom



def SOM_clustering(X, grid_shape, random_state=None):
    # setup SOM
    n_nodes = grid_shape[0]*grid_shape[1]
    som = MiniSom(grid_shape[0],grid_shape[1],14,sigma=0.8,learning_rate=0.6,random_seed=random_state)
    # fit SOM
    som.train_random(X, 2000)
    # assign labels to node
    labels = np.apply_along_axis(som.winner, 1, X).astype(str)
    return som, np.char.add(labels[:,0], labels[:,1])


def find_optimal_architecture_and_cluster(X, nodes, random_state=None):
    label_list = []
    CH_score = []
    for architecture in nodes:
        labels = SOM_clustering(X,architecture,random_state=random_state)[-1]
        # Paris et al. 2019 uses the Calinski Harabasz index to identify the number of clusters to use
        score = calinski_harabasz_score(X, labels)
        label_list.append(labels)
        CH_score.append(score)
    return label_list[np.argmax(CH_score)]

def get_keep_discard_pixels(labels):
    pd.Series(labels)
