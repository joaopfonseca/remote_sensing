# core
import pandas as pd
import numpy as np
import pickle

# configs
random_state = 0
DATA_PATH = 'T29SNC/data/preprocessed/2020_01_RS_1.csv'
MODELS_PATH = 'T29SNC/models/'
RESULTS_PATH = 'T29SNC/results/'

df = pd.read_csv(DATA_PATH)

# split by feature type
df_meta = df[['x','y','Megaclasse', 'class18']]
df_bands = df.drop(columns=df_meta.columns)




# util functions
def make_data_cubes(X, order):
    """order is the indices of the ordered bands/months"""
    X_reshaped = np.reshape(
        X[:,order],
        (X.shape[0], 4, 3, int(X.shape[-1]/(12)))
    )
    return X_reshaped

def make_data_arrays(X, order):
    """order is the indices of the ordered bands/months"""
    X_reshaped = np.reshape(
        X[:,order],
        (X.shape[0], 12, int(X.shape[-1]/(12)))
    )
    return X_reshaped
