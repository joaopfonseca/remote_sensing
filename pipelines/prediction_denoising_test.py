import sys
import os
#PROJ_PATH = os.path.realpath(os.path.join(os.path.dirname(__file__), '../../'))
#sys.path.append(PROJ_PATH)
#print(PROJ_PATH)
PROJ_PATH = '.'
## data manipulation and transformation
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from src.models.denoiser import DenoiserAE
from src.preprocess.utils import get_patches, get_2Dcoordinates_matrix

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
CSV_PATH = PROCESSED_PATH+'classification_results_resnet.csv'

## denoiser configs
window_size  = 33
n_splits_denoiser = 50
n_epochs = 3

random_state = 0

################################################################################
# DATA READING AND PREPROCESSING
################################################################################

df = pd.read_csv(CSV_PATH).drop(columns=['Unnamed: 0'])
noisy_lookup = df.pivot('x','y','y_pred').fillna(-1).values
clean_lookup = df.pivot('y','x','Megaclasse').fillna(-1).values

coords = get_2Dcoordinates_matrix(noisy_lookup.shape, window_size=window_size)

coords = df[['y','x']]
labels = df[['Megaclasse']]
criteria = (
    coords>coords.min()+int(window_size/2)+1
    ).all(axis=1) & (
    coords<coords.max()-int(window_size/2)
    ).all(axis=1)
coords = coords[criteria].astype(int).values
labels = labels[criteria].astype(int).values
#X = get_patches(coords, noisy_lookup, window_size)
#y = get_patches(coords, clean_lookup, window_size)

denoiser = DenoiserAE((window_size,window_size,int(clean_lookup.max())))
#denoiser.fit(X, y, batch_size=256, epochs=1)


print(f'Stratified Splitting: {n_splits_denoiser} splits, {int(coords.shape[0]/n_splits_denoiser)} pixels per split')
skf = StratifiedKFold(n_splits = n_splits_denoiser, shuffle=True, random_state=random_state)
for epoch in range(1,n_epochs+1):
    for i, (_, split_indices) in enumerate(skf.split(coords, labels)):
        print(f'Epoch {epoch}/{n_epochs}, Split {i+1}/{n_splits_denoiser}')
        _coords = coords[split_indices]
        X = get_patches(_coords, noisy_lookup, window_size)
        y = get_patches(_coords, clean_lookup, window_size)
        denoiser.fit(X, y, batch_size=256, epochs=1)
