# core
import numpy as np
import pandas as pd

# utilities
import pickle
from sklearn.model_selection import train_test_split
from src.reporting.reports import reports

# data normalization
from sklearn.preprocessing import StandardScaler

# classifiers
from sklearn.ensemble import RandomForestClassifier
from src.models.HybridSpectralNet import PixelBasedHybridSpectralNet
from src.models.resnet import ResNet50
from src.models.recurrent import LSTMNet

# configs
random_state = 0
DATA_PATH = 'T29SNC/data/preprocessed/2019_02_RS_0.csv'
MODELS_PATH = 'T29SNC/models/'
RESULTS_PATH = 'T29SNC/results/'

# read data
df = pd.read_csv(DATA_PATH)
df = df.dropna()

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
X_train, X_test, X_train_arrays, X_test_arrays, X_train_cubes, X_test_cubes, y_train, y_test = train_test_split(
    X, X_arrays, X_cubes, y, test_size=.20, random_state=random_state
)

# set up experiments
## Random Forest
rfc = RandomForestClassifier(n_estimators=100, random_state=random_state)
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
rfc_results = reports(y_test, y_pred)
pickle.dump(rfc_results, open(RESULTS_PATH+'rfc_results.pkl','wb'))
pickle.dump(rfc, open(MODELS_PATH+'random_forest.pkl','wb'))

## HybridSN
cnn = PixelBasedHybridSpectralNet(X_train_cubes[0].shape, y.max()+1, MODELS_PATH+'pixelbased_hybridsn.hdf5')
cnn.fit(X_train_cubes, y_train, epochs=200)
y_pred = cnn.predict(X_test_cubes)
cnn_results = reports(y_test, y_pred)
pickle.dump(cnn_results, open(RESULTS_PATH+'HybridSN_results.pkl','wb'))

## ResNet50
resnet = PixelBasedResNet50(X_train_arrays[0].shape, y.max()+1, MODELS_PATH+'pixelbased_resnet50.hdf5')
resnet.fit(X_train_arrays, y_train, epochs=200)


## LSTM
lstm = LSTMNet(X_train_arrays[0].shape, y.max()+1, MODELS_PATH+'lstm.hdf5')
lstm.fit(X_train_arrays, y_train, epochs=100)
y_pred = lstm.predict(X_test_arrays)
lstm_results = reports(y_test, y_pred)
pickle.dump(lstm_results, open(RESULTS_PATH+'lstm_results.pkl','wb'))
