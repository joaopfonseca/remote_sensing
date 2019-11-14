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
from sklearn.decomposition import PCA
## filters
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
## own libraries
from src.preprocess.readers import SentinelProductReader
from src.models.HybridSpectralNet import HybridSpectralNet
from src.preprocess.utils import (
    ZScoreNormalization,
    pad_X,
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
read_csv     = False        # Read saved csv with band values
random_state = 0            # Random state for everything. Generally left unchanged.
debuffer_cos = True         # Removes labels of pixels close to the border of each polygon
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
n_epochs     = 3            # Numb of epochs
n_splits_cnn = 100          # Numb of splits on the training data. This is done to lower memory usage
window_size  = 25           # Size of the window to pass to the CNN
output_units = 9            # Number of labels



################################################################################
# DATA READING AND PREPROCESSING
################################################################################

if read_csv:
    df = pd.read_csv(CSV_PATH)
else:
    print('Reading image...')
    coimbrinhas = SentinelProductReader(
        bands_path=BANDS_PATH,
        labels_shapefile=COS_PATH,
        label_col='Megaclasse'
    )
    coimbrinhas.add_indices(['NDVI', 'NDBI', 'NDMI', 'NDWI'])
    coimbrinhas.get_X_array()
    coimbrinhas.get_y_array()

    print('Selecting study area subset...')
    x_center, y_center = center_pixel
    margin = int((width_height/2))
    x_lowlim  = x_center-margin
    x_highlim = x_center+margin
    y_lowlim  = y_center-margin
    y_highlim = y_center+margin

    coimbrinhas.X_array = coimbrinhas.X_array\
        [x_lowlim-margin:x_highlim+margin,
            y_lowlim-margin:y_highlim+margin]
    coimbrinhas.y_array = coimbrinhas.y_array\
        [x_lowlim-margin:x_highlim+margin,
            y_lowlim-margin:y_highlim+margin]

    df = coimbrinhas.to_pandas()
    del coimbrinhas

## Normalize and dimensionality reduction
bands_list = df.drop(columns=['y','x','Megaclasse']).sort_index(axis=1).columns
zscorers = df[bands_list].apply(lambda x: ZScoreNormalization(x, axes=(0))[1])
pca = PCA(n_components=10, random_state=random_state)

norm_pca_vals = pca.fit_transform(df[bands_list].apply(lambda x: zscorers[x.name](x)))
norm_pca_cols = ['comp_'+str(x) for x in range(10)]
df_norm_pca = pd.DataFrame(norm_pca_vals, columns=norm_pca_cols)
df = df.drop(columns=bands_list).join([df[bands_list], df_norm_pca])

# TODO: PICKLE zscorers and PCA!

## divide data from dataframe into train and testing
margin = int((width_height/2))

df['train_set'] = df.apply(lambda row: (
    row['x']<margin or row['x']>=3*margin or row['y']<margin or row['y']>=3*margin
), axis=1)


################################################################################
# PIXEL SELECTION
################################################################################

# TODO




################################################################################
# Hybrid Spectral Net
################################################################################
def pivot_multispectral(df, xy_cols, bands):
    rgb = []
    for band in bands:
        rgb.append(df.pivot(xy_cols[0], xy_cols[1], band).values)
    return np.moveaxis(np.array(rgb), 0, -1)#.T.swapaxes(0,1)


## model setup
print('Setting model...')
ConvNet = HybridSpectralNet(input_shape=(window_size, window_size, K), output_units=output_units)
ConvNet.load_weights('best_model.hdf5')

coords = df[['y','x']]
criteria = (coords>coords.min()+int(window_size/2)+1).all(axis=1) & (coords<coords.max()-int(window_size/2)).all(axis=1)
X_coords = df[['y','x']][df['train_set']&criteria].astype(int).values
y_labels = df['Megaclasse'][df['train_set']&criteria].astype(int).values

X_lookup = pivot_multispectral(df, ['y','x'], norm_pca_cols)

print(f'Stratified Splitting: {n_splits_cnn} splits, {int(X_coords.shape[0]/n_splits_cnn)} pixels per split')
skf = StratifiedKFold(n_splits = n_splits_cnn, shuffle=True, random_state=random_state)
for epoch in range(1,n_epochs+1):
    i = 0
    for _, split_indices in skf.split(X_coords, y_labels):
        i+=1
        print(f'Epoch {epoch}/{n_epochs}, Split {i}/{n_splits_cnn}')
        X_split, y_split = X_coords[split_indices], y_labels[split_indices]
        X_patches = get_patches(X_split, X_lookup, window_size)
        ConvNet.fit(X_patches, y_split, batch_size=256, epochs=1)

################################################################################
# REPORTING AND IMAGE GENERATION
################################################################################

X_lookup = pivot_multispectral(df, ['y','x'], norm_pca_cols)

coords = df[['y','x']]
criteria = (coords>coords.min()+int(window_size/2)+1).all(axis=1) & (coords<coords.max()-int(window_size/2)).all(axis=1)

X_coords = coords[criteria].astype(int).values

indices = []
y_pre   = []
i = 0
skf = StratifiedKFold(n_splits = n_splits_cnn, shuffle=True, random_state=random_state)
for _, split_indices in skf.split(X_coords, np.zeros(X_coords.shape[0])):
    i+=1; print(f'Prediction progress: {(i/n_splits_cnn)*100}%')
    X_split = X_coords[split_indices]
    X_patches = get_patches(X_split, X_lookup, window_size)
    indices.append(X_split)
    y_pre.append(ConvNet.predict(X_patches))

#df_final = coords.copy()
#y_pred = pd.Series(data=np.concatenate(y_pre), index=np.concatenate(indices), name='y_pred').sort_index()
#df_final = df_final.join(y_pred)

y_pred = pd.DataFrame(
    data=np.concatenate([np.expand_dims(np.concatenate(y_pre),1), np.concatenate(indices)], axis=1),
    columns=['y_pred', 'y', 'x']
)

df = df.join(y_pred.set_index(['y','x']), on=['y','x'])
df.to_csv(PROCESSED_PATH+'classification_results.csv')



reports(df[df['train_set']]['Megaclasse'], df[df['train_set']]['y_pred'], labels)[-1]
