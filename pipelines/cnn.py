
import pickle
from src.preprocess.readers import SentinelProductReader
from src.models.HybridSpectralNet import HybridSpectralNet
from src.preprocess.utils import (
    ZScoreNormalization,
    pad_X,
    applyPCA,
    get_2Dcoordinates_matrix,
    get_patches
)
from src.reporting.reports import reports # make this structure more proper

import numpy as np
from sklearn.model_selection import StratifiedKFold

## configs
DATA_PATH = '../data/sentinel_coimbra/raw/'
PRODUCT_PATH = DATA_PATH+'S2A_MSIL1C_20150725T112046_N0204_R037_T29TNE_20150725T112540.SAFE/'
BANDS_PATH = PRODUCT_PATH+'/GRANULE/L1C_T29TNE_A000463_20150725T112540/IMG_DATA/'
COS_PATH = DATA_PATH+'COS2015-V1-PT16E_Regiao_Coimbra/COS2015-V1-PT16E_Regiao_Coimbra.shx'
INTERIM_PATH = '../data/sentinel_coimbra/interim/'

n_splits = 100
window_size = 25
read_pickle = True
random_state = 0
K = 10
output_units = 41

## read data
print('Reading data...')
if not read_pickle:
    coimbrinhas = SentinelProductReader(
        bands_path=BANDS_PATH,
        labels_shapefile=COS_PATH,
        label_col='COS2015_Le' # Much more complex labels, originally was 'Megaclasse'
    )
    coimbrinhas.add_indices(['NDVI', 'NDBI', 'NDMI', 'NDWI'])
    coimbrinhas.dump(INTERIM_PATH+'coimbra.pkl')
    coimbrinhas.plot(alpha=0.5)
else:
    coimbrinhas = pickle.load(open(INTERIM_PATH+'coimbra.pkl', 'rb'))
    coimbrinhas.get_X_array()
    coimbrinhas.get_y_array()


labels = coimbrinhas.y_labels
X = coimbrinhas.X_array
y = coimbrinhas.y_array
del coimbrinhas

# comment out this section when not testing
X = X[5000:6000, 5000:6000]
y = y[5000:6000, 5000:6000]

## preprocessing
print('Preprocessing data: Z-Score Normalization')
X_norm, zscore = ZScoreNormalization(X, axes=(0,1))
del X
print('Preprocessing data: PCA Dimensionality Reduction')
X_pca, pca = applyPCA(X_norm, numComponents=K)
del X_norm
print('Preprocessing data: Padding X')
X_lookup = pad_X(X_pca, window_size)

## get coordinates used for training the model
print('Getting coordinates matrix...')
X_coords = get_2Dcoordinates_matrix(X_lookup.shape, window_size).reshape((2,X_pca.shape[1]**2)).T
y_coords = y.reshape((X_pca.shape[1]**2)).T
del y, X_pca
## model setup
print('Setting model...')
ConvNet = HybridSpectralNet(input_shape=(window_size, window_size, K), output_units=output_units)

## train model
i = 0
print(f'Stratified Splitting: {n_splits} splits, {int(X_coords.shape[0]/n_splits)} pixels per split')
skf = StratifiedKFold(n_splits = n_splits, shuffle=True, random_state=None)
for _, split_indices in skf.split(X_coords, y_coords):
    i+=1
    print(f'Split #{i}')
    X_split, y_split = X_coords[split_indices], y_coords[split_indices]
    X_patches = get_patches(X_split, X_lookup, window_size)
    ConvNet.fit(X_patches, y_split, batch_size=256, epochs=1)


#y_true = y_test[:1000]
#y_pred = ConvNet.predict(X_test[:1000], filepath='best_model.hdf5')
#reports(y_true, y_pred, labels)
#plot_image([X_test, y_true, y_pred])
