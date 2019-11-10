import sys
import os
PROJ_PATH = '.'

import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from src.preprocess.readers import SentinelProductReader
from src.models.HybridSpectralNet import HybridSpectralNet
from src.preprocess.utils import (
    ZScoreNormalization,
    pad_X,
    applyPCA,
    get_patches,
    get_2Dcoordinates_matrix
)
from src.reporting.reports import reports # make this structure more proper
from src.preprocess.data_selection import KMeans_filtering


################################################################################
# CONFIGS
################################################################################
## data path configs
DATA_PATH = PROJ_PATH+'/data/sentinel_coimbra/raw/'
PRODUCT_PATH = DATA_PATH+'S2A_MSIL1C_20150725T112046_N0204_R037_T29TNE_20150725T112540.SAFE/'
BANDS_PATH = PRODUCT_PATH+'/GRANULE/L1C_T29TNE_A000463_20150725T112540/IMG_DATA/'
COS_PATH = DATA_PATH+'COS2015-V1-PT16E_Regiao_Coimbra/COS2015-V1-PT16E_Regiao_Coimbra.shx'
INTERIM_PATH = PROJ_PATH+'/data/sentinel_coimbra/interim/'
PROCESSED_PATH = DATA_PATH+'../processed/'
CSV_PATH = PROCESSED_PATH+'picture_data.csv'
## preprocessing configs
random_state = 0            # Random state for everything. Generally left unchanged.
debuffer_cos = True         # Removes labels of pixels close to the border of each polygon
read_pickle  = True         # Read pickled SentinelProductReader object
read_csv     = True         # Read saved csv with band values
K            = 10           # Number of components for PCA
center_pixel = (5500,5500)  # Center for the study area
width_height = 1000         # Height and width of study (test) area
## pixel selection configs
n_splits_ps  = 7            # Numb of splits. total filters = len(filters)*n_splits_ps
granularity  = 10           # Hyperparam used to compute the number of clusters. More granularity = more clusters
keep_rate    = 0.7          # Rough % of observations to keep for each class
## Hybrid Spectral Net configs
n_epochs     = 2            # Numb of epochs
n_splits_cnn = 100          # Numb of splits on the training data. This is done to lower memory usage
window_size  = 25           # Size of the window to pass to the CNN
output_units = 9            # Number of labels


print('Reading image...')
coimbrinhas = SentinelProductReader(
    bands_path=BANDS_PATH,
    labels_shapefile=COS_PATH,
    label_col='Megaclasse'
)
coimbrinhas.add_indices(['NDVI', 'NDBI', 'NDMI', 'NDWI'])

labels = coimbrinhas.y_labels
X = coimbrinhas.X_array
y = coimbrinhas.y_array
del coimbrinhas

## Normalize and dimensionality reduction
print('Preprocessing data: Z-Score Normalization')
X_norm, zscore = ZScoreNormalization(X, axes=(0,1))
del X
print('Preprocessing data: PCA Dimensionality Reduction')
X_pca, pca = applyPCA(X_norm, numComponents=K)
del X_norm
print('Preprocessing data: Padding X')
X_lookup = pad_X(X_pca, window_size)
del X_pca

## select area subset
#print('Preprocessing data: Selecting area subset...')
x_center, y_center = center_pixel
margin = int((width_height/2))
x_lowlim  = x_center-margin
x_highlim = x_center+margin
y_lowlim  = y_center-margin
y_highlim = y_center+margin

print('Reading df')
df = pd.read_csv(CSV_PATH)

df_test = df[
    ((df['x']>=margin)&(df['x']<3*margin))
    &
    ((df['y']>=margin)&(df['y']<3*margin))
]

print('Reading df_pre_train')
df_pre_train = pd.read_csv(PROCESSED_PATH+'df_pre_train.csv')
df_selected = df_pre_train[df_pre_train['status']&(df_pre_train['Megaclasse']!=-1)]



print('Loading CNN')
ConvNet = HybridSpectralNet(input_shape=(window_size, window_size, K), output_units=output_units)
ConvNet.load_weights('models/ps_best_model.hdf5')
