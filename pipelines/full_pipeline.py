import sys
import os
PROJ_PATH = os.path.realpath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(PROJ_PATH)
print(PROJ_PATH)

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
random_state = 0        # Random state for everything. Generally left unchanged.
debuffer_cos = True     # Removes labels of pixels close to the border of each polygon
read_pickle  = False    # Read pickled SentinelProductReader object
read_csv     = False    # Read saved csv with band values
K            = 10       # Number of components for PCA
## Hybrid Spectral Net configs
n_splits = 100          # Numb of splits on the training data. Each split = one epoch
window_size = 25        # Size of the window to pass to the CNN
output_units = 41       # Number of labels


################################################################################
# DATA READING AND PREPROCESSING
################################################################################
## preprocess shapefile
if debuffer_cos:
    pass # rewrite cos and update COS_PATH
else:
    pass

## read data
if not read_pickle:
    print('Reading image...')
    coimbrinhas = SentinelProductReader(
        bands_path=BANDS_PATH,
        labels_shapefile=COS_PATH,
        label_col='Megaclasse'
    )
    coimbrinhas.add_indices(['NDVI', 'NDBI', 'NDMI', 'NDWI'])
    coimbrinhas.dump(INTERIM_PATH+'coimbra.pkl')
    coimbrinhas.plot(alpha=0.5)
else:
    print('Loading pickle object...')
    coimbrinhas = pickle.load(open(INTERIM_PATH+'coimbra.pkl', 'rb'))
    coimbrinhas.get_X_array()
    coimbrinhas.get_y_array()

# UNFINISHED PART - PROBABLY GOING TO BE REMOVED AND ADAPTED TO WORK ON THE AREA SUBSET ONLY
if read_csv:
    print('Reading csv...')
    df = pd.read_csv(CSV_PATH)
else:
    print('Generating csv...')
    df = coimbrinhas.to_pandas()
    df.to_csv(CSV_PATH, index=False)

labels = coimbrinhas.y_labels
X = coimbrinhas.X_array
y = coimbrinhas.y_array
del coimbrinhas

## select area subset
# UNFINISHED PART
X = X[5000:6000, 5000:6000]
y = y[5000:6000, 5000:6000]

## sepparate into train and test areas
# UNFINISHED PART
pass

################################################################################
# PIXEL SELECTION
################################################################################





################################################################################
# SETUP AND TRAINING - Hybrid Spectral Net
################################################################################
## data preprocessing
print('Preprocessing data: Z-Score Normalization')
X_norm, zscore = ZScoreNormalization(X, axes=(0,1))
del X
print('Preprocessing data: PCA Dimensionality Reduction')
X_pca, pca = applyPCA(X_norm, numComponents=K)
del X_norm
print('Preprocessing data: Padding X')
X_lookup = pad_X(X_pca, window_size)



################################################################################
# REPORTING AND IMAGE GENERATION
################################################################################
