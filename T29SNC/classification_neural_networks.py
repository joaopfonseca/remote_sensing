# core
import numpy as np
import pandas as pd

# utilities
from sklearn.model_selection import train_test_split

# data normalization
from sklearn.preprocessing import StandardScaler

# classifiers
from src.models.HybridSpectralNet import PixelBasedHybridSpectralNet
from src.models.resnet import ResNet50

# configs
random_state = 0
DATA_DIR = 'T29SNC/data/preprocessed/2019_02.csv'
MODELS_DIR = 'T29SNC/models/'

# read data
df = pd.read_csv(DATA_DIR)
df = df.dropna()

# split by feature type
# split by feature type
df_meta = df[['x','y','Megaclasse']]
df_bands = df.drop(columns=df_meta.columns)

# normalize
znorm = StandardScaler()
df_bands = pd.DataFrame(znorm.fit_transform(df_bands.values), columns=df_bands.columns, index=df_bands.index)

# get data in simple format
X = df_bands.values
y = df_meta.Megaclasse.values.astype(int)

months = np.array([c.split('_')[1]  for c in df_bands.columns])
months[months=='12'] = '00'
bands  = np.array([c.split('_')[-1] for c in df_bands.columns])
order = np.argsort(np.array([f'{m}_{b}' for m,b in zip(months, bands)]))

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


# prepare data
X_cubes = make_data_cubes(X, order)
X_arrays = make_data_arrays(X, order)
X_train_arrays, X_test_arrays, X_train_cubes, X_test_cubes, y_train, y_test = train_test_split(
    X_arrays, X_cubes, y, test_size=.20, random_state=random_state
)

# set up experiments
## HybridSN
cnn = PixelBasedHybridSpectralNet(X_train_cubes[0].shape, y.max()+1, MODELS_DIR+'hybridsn.hdf5')
cnn.fit(X_train_cubes, y_train, epochs=150)

## ResNet50
resnet = ResNet50(X_train_cubes[0].shape, y.max()+1, MODELS_DIR+'resnet50.hdf5')
resnet.fit(X_train_cubes, y_train)

## LSTM
