import os
import pandas as pd
import numpy as np
from src.preprocess.data_selection import pixel_selection
from src.reporting.visualize import plot_image
from sklearn.preprocessing import LabelEncoder

## configs
DATA_PATH = 'data/DGT/'
RAW_CSV_PATH = DATA_PATH+'raw/'
MERGED_CSV = DATA_PATH+'interim/all_outputs.csv'
RESULTS_PATH = DATA_PATH+'processed/data_selection_results.csv'

## merge all csv's together and write to disk
dfs = []
for file in os.listdir(RAW_CSV_PATH):
    dfs.append(pd.read_csv(RAW_CSV_PATH+file))
pd.concat(dfs).drop(columns=['Unnamed: 0']).to_csv(MERGED_CSV, index=False)

## read merged data
df = pd.read_csv(MERGED_CSV)

## discard unnecessary data cols
labels_coords_cols = ['X','Y','Object','Label']
band_cols = [x for x in df.columns if (x.startswith('X') and len(x)>1)]
df = df[labels_coords_cols+band_cols]

## drop rows with missing values
df = df.dropna()

## run pixel_selection method 1: SOM+SOM
ps = pixel_selection(df, polygon_id_col='Object', class_col='Label')
ps.get_clusters(method='SOM', cluster_col='clusters', random_state=0)
df_selection, clusters = ps.get_consistency_analysis(method='SOM', consistency_col='consistency_results', random_state=0)
df_selection['consistency'] = df_selection['consistency_results'].apply(lambda x: x.split('_')[-1])
df_selection.to_csv(RESULTS_PATH) # save results

## run pixel_selection method 2: SOM+drop minority clusters+Bhattacharyya (Paris et al. 2019)

## run pixel_selection method 3: SOM+K-means

## run pixel_selection method 4: K-means+SOM

## William's reference polygons: 71125 (conifers), 8458 (baresoil)
## plot results
obj = df_selection[df_selection['Object']==8458]
obj[['X','Y']] = ((obj[['X','Y']] - obj[['X','Y']].min()) / 10).astype(int)
obj['consistency'] = obj['clusters'].apply(lambda x: x.split('_')[-1])

majority_index = obj.groupby(['consistency']).size().index[0]

_accepted = (obj.pivot('X','Y','consistency').values==majority_index).T.astype(float)
accepted = img*np.array([_accepted for i in range(3)]).T.swapaxes(0,1)
_rejected = (obj.pivot('X','Y','consistency').values!=majority_index).T.astype(float).T.swapaxes(0,1)
rejected = img*np.array([_rejected for i in range(3)]).T.swapaxes(0,1)

img = np.array([obj.pivot('X', 'Y', band).values for band in ['X2017.12.21.B4', 'X2017.12.21.B3', 'X2017.12.21.B2']]).T

plot_image([img,rejected,accepted], num_rows=1, figsize=(40, 20), dpi=20)
