import sys
import os
PROJ_PATH = os.path.realpath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(PROJ_PATH)
print(PROJ_PATH)

import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
## filters
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

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
from src.reporting.visualize import plot_image

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
read_pickle  = False        # Read pickled SentinelProductReader object
read_csv     = False        # Read saved csv with band values
K            = 10           # Number of components for PCA
center_pixel = (4500,4500)  # Center for the study area
width_height = 750          # Height and width of study (test) area
## pixel selection configs
use_ps = False
filters = (
    ('RandomForestClassifier', RandomForestClassifier(random_state=random_state)),
    ('DecisionTreeClassifier', DecisionTreeClassifier(random_state=random_state)),
    ('LogisticRegression', LogisticRegression(random_state=random_state)),
    ('MLPClassifier', MLPClassifier(random_state=random_state))
)                           # Classifiers used for filtering data
n_splits_ps  = 7            # Numb of splits. total filters = len(filters)*n_splits_ps
granularity  = 10           # Hyperparam used to compute the number of clusters. More granularity = more clusters
keep_rate    = 0.7          # Rough % of observations to keep for each class
## Hybrid Spectral Net configs
n_epochs     = 2            # Numb of epochs
n_splits_cnn = 100          # Numb of splits on the training data. This is done to lower memory usage
window_size  = 25           # Size of the window to pass to the CNN
output_units = 9            # Number of labels


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
    coimbrinhas.get_X_array()
    coimbrinhas.get_y_array()
    #coimbrinhas.dump(INTERIM_PATH+'coimbra.pkl')
    #coimbrinhas.plot(alpha=0.5)
else:
    print('Loading pickle object...')
    coimbrinhas = pickle.load(open(INTERIM_PATH+'coimbra.pkl', 'rb'))
    coimbrinhas.add_indices(['NDVI', 'NDBI', 'NDMI', 'NDWI'])
    coimbrinhas.get_X_array()
    coimbrinhas.get_y_array()



## select area subset
#print('Preprocessing data: Selecting area subset...')
x_center, y_center = center_pixel
margin = int((width_height/2))
x_lowlim  = x_center-margin
x_highlim = x_center+margin
y_lowlim  = y_center-margin
y_highlim = y_center+margin

# UNFINISHED PART - PROBABLY GOING TO BE REMOVED AND ADAPTED TO WORK ON THE AREA SUBSET ONLY
if read_csv:
    print('Reading csv...')
    df = pd.read_csv(CSV_PATH)
else:
    print('Selecting study area subset...')
    coimbrinhas.X_array = coimbrinhas.X_array\
        [x_lowlim-margin:x_highlim+margin,
        y_lowlim-margin:y_highlim+margin]
    coimbrinhas.y_array = coimbrinhas.y_array\
        [x_lowlim-margin:x_highlim+margin,
        y_lowlim-margin:y_highlim+margin]

    print('Generating csv...')
    df = coimbrinhas.to_pandas()
    #df.to_csv(CSV_PATH, index=False)

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

## divide data from dataframe into train and testing
df_pre_train = df[
    ((df['x']<margin)|(df['x']>=3*margin))
    |
    ((df['y']<margin)|(df['y']>=3*margin))
]

df_test = df[
    ((df['x']>=margin)&(df['x']<3*margin))
    &
    ((df['y']>=margin)&(df['y']<3*margin))
]


################################################################################
# PIXEL SELECTION
################################################################################
if use_ps:
    clusters, status = KMeans_filtering(
        df_pre_train.drop(columns=['Megaclasse']).values,
        df_pre_train['Megaclasse'].values,
        filters,
        n_splits_ps,
        granularity,
        keep_rate,
        random_state=random_state
    )
    df_pre_train['clusters'] = clusters
    df_pre_train['status']   = status
    df_pre_train = df_pre_train[df_pre_train['Megaclasse']!=-1]
    df_pre_train.to_csv(PROCESSED_PATH+'df_pre_train.csv')

    df_selected = df_pre_train[df_pre_train['status']&(df_pre_train['Megaclasse']!=-1)]
    X_coords    = df_selected[['x', 'y']].astype(int).values
    y_coords    = df_selected['Megaclasse'].values
else:
    df_selected = df_pre_train[df_pre_train['Megaclasse']!=-1]
    X_coords    = df_selected[['x', 'y']].astype(int).values
    y_coords    = df_selected['Megaclasse'].values

################################################################################
# Hybrid Spectral Net
################################################################################
## model setup
print('Setting model...')
ConvNet = HybridSpectralNet(input_shape=(window_size, window_size, K), output_units=output_units)

## train model
print(f'Stratified Splitting: {n_splits_cnn} splits, {int(X_coords.shape[0]/n_splits_cnn)} pixels per split')
skf = StratifiedKFold(n_splits = n_splits_cnn, shuffle=True, random_state=random_state)
for epoch in range(1,n_epochs+1):
    i = 0
    for _, split_indices in skf.split(X_coords, y_coords):
        i+=1
        print(f'Epoch {epoch}/{n_epochs}, Split {i}/{n_splits_cnn}')
        X_split, y_split = X_coords[split_indices], y_coords[split_indices]
        X_patches = get_patches(X_split+int(window_size/2), X_lookup, window_size)
        ConvNet.fit(X_patches, y_split, batch_size=256, epochs=1)


################################################################################
# REPORTING AND IMAGE GENERATION
################################################################################










#####



xmin = x_highlim-x_lowlim-margin-int(window_size/2)
ymin = y_highlim-y_lowlim-margin-int(window_size/2)

#X_test_lookup = X_lookup#\
    #[xmin:xmin+margin*2+int(window_size/2)*2, ymin:ymin+margin*2+int(window_size/2)*2]
#del X_lookup

X_test_coords = get_2Dcoordinates_matrix(X_lookup.shape, window_size).reshape(2,(margin*4)**2).T[:,[1,0]]

indices = []
y_pre   = []
i = 0
for _, split_indices in skf.split(X_test_coords, np.zeros(X_test_coords.shape[0])):
    i+=1; print(f'Prediction progress: {(i/n_splits_cnn)*100}%')
    X_split = X_test_coords[split_indices]
    X_patches = get_patches(X_split, X_lookup, window_size)
    indices.append(split_indices)
    y_pre.append(ConvNet.predict(X_patches, filepath='models/ps_best_model.hdf5'))


df_final = pd.DataFrame(data=X_test_coords, columns=['x','y'])
y_pred = pd.Series(data=np.concatenate(y_pre), index=np.concatenate(indices), name='y_pred').sort_index()

df_test2 = df_test[['x','y','Megaclasse', 'B04', 'B03', 'B02']].rename(columns={'Megaclasse':'y_true'}).astype(int)
df_test2['x'] = df_test2['x']-xmin
df_test2['y'] = df_test2['y']-ymin

df_final = df_final.join(y_pred).join(df_test2.set_index(['x','y']), on=['x', 'y'])
df_final.to_csv(PROCESSED_PATH+'df_pred.csv')

def pivot_multispectral(df, xy_cols, bands):
    rgb = []
    for band in bands:
        rgb.append(df.pivot(xy_cols[0], xy_cols[1], band).values)
    return np.moveaxis(np.array(rgb), 0, -1)#.T.swapaxes(0,1)

xy_cols = ('x', 'y')
rgb_cols = ('B04', 'B03', 'B02')

total_rgb = pivot_rgb(df, xy_cols, rgb_cols) # to plot entire study area
train_rgb = pivot_rgb(df_pre_train, xy_cols, rgb_cols) # to plot training area
test_rgb  = pivot_rgb(df_test, xy_cols, rgb_cols) # to plot test area

total_gt = df.pivot('x', 'y', 'Megaclasse').values # to plot entire study area's ground truth
train_gt = df_pre_train.pivot('x', 'y', 'Megaclasse').values # to plot training area's ground truth
test_gt  = df_test.pivot('x', 'y', 'Megaclasse').values # to plot test area's ground truth

train_rgb # to plot training area
ps_rgb = pivot_rgb(df_selected, xy_cols, rgb_cols) # to plot selected pixels
ps_gt  = df_selected.pivot('x', 'y', 'Megaclasse').values # to plot selected pixels' ground truth

test_rgb # to plot test area
test_gt # to plot ground truth
pred_labels = df_final.pivot('x', 'y', 'y_pred').values # to plot predictions

reports(df_final['y_true'], df_final['y_pred'], labels)[-1]

rgbrgb = pivot_rgb(df_final, xy_cols, rgb_cols)

plot_image([
#    np.moveaxis(total_rgb,0,1), np.moveaxis(train_rgb,0,1), np.moveaxis(test_rgb,0,1),
#    np.moveaxis(total_gt,0,1), np.moveaxis(train_gt,0,1), np.moveaxis(test_gt,0,1),
#    np.moveaxis(train_rgb,0,1), np.moveaxis(ps_rgb,0,1), np.moveaxis(ps_gt,0,1),
    np.moveaxis(test_rgb,0,1), np.moveaxis(test_gt,0,1), pred_labels
    ],
#    num_rows=4#, figsize=(80,20)
)
