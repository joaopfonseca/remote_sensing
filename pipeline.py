
import pickle
from src.preprocess.product_reader import SentinelProductReader
from src.models.HybridSpectralNet import HybridSpectralNetCustom
from src.preprocess.utils import (
    ZScoreNormalization,
    pad_X,
    applyPCA,
    get_2Dcoordinates_matrix,
    get_patches
)
from src.reporting.reports import reports # make this structure more proper

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

## configs
DATA_PATH = 'data/sentinel_coimbra/raw/'
PRODUCT_PATH = DATA_PATH+'S2A_MSIL1C_20150725T112046_N0204_R037_T29TNE_20150725T112540.SAFE/'
BANDS_PATH = PRODUCT_PATH+'/GRANULE/L1C_T29TNE_A000463_20150725T112540/IMG_DATA/'
COS_PATH = DATA_PATH+'COS2015-V1-PT16E_Regiao_Coimbra/COS2015-V1-PT16E_Regiao_Coimbra.shx'
INTERIM_PATH = 'data/sentinel_coimbra/interim/'

n_splits = 100
window_size = 25
read_pickle = True
random_state = 0
K=10

## read data
if not read_pickle:
    coimbrinhas = SentinelProductReader(
        bands_path=BANDS_PATH,
        labels_shapefile=COS_PATH,
        label_col='COS2015_Le' # Much more complex labels, originally was 'Megaclasse'
    )
    coimbrinhas.add_indices(['NDVI', 'NDBI', 'NDMI', 'NDWI'])
    coimbrinhas.dump(INTERIM_PATH+'coimbra.pkl')
    coimbrinhas.get_X_array()
    coimbrinhas.get_y_array()
    coimbrinhas.plot(alpha=0.5)
else:
    coimbrinhas = pickle.load(open(INTERIM_PATH+'coimbra.pkl', 'rb'))
    coimbrinhas.get_X_array()
    coimbrinhas.get_y_array()


labels = coimbrinhas.y_labels
X = coimbrinhas.X_array
y = coimbrinhas.y_array
del coimbrinhas

## preprocessing
X_norm, zscore = ZScoreNormalization(X, axes=(0,1))
X_pca, pca = applyPCA(X_norm, numComponents=K)
X_lookup = pad_X(X_pca, window_size)

## get lookup array X and coordinates used for training the model
X_coords = get_2Dcoordinates_matrix(X_lookup.shape, window_size).reshape((2,X_pca.shape[1]**2)).T
y_coords = y.reshape((2,X_pca.shape[1]**2)).T

## Model setup
ConvNet = HybridSpectralNet(input_shape=(window_size, window_size, K), output_units=40)

## train model
sss = StratifiedShuffleSplit(n_splits = n_splits)
for X_split, y_split in sss.split(X_coords, y_coords):

    X_patches, y_patches = get_patches(X_split, X_lookup, y_split, window_size).T
    ConvNet.fit(X_patches, y_patches, batch_size=256, epochs=1, filepath='best_model.hdf5')


#y_true = y_test[:1000]
#y_pred = ConvNet.predict(X_test[:1000], filepath='best_model.hdf5')
#reports(y_true, y_pred, labels)
#plot_image([X_test, y_true, y_pred])
