import os
import pandas as pd
import numpy as np
from src.preprocess.data_selection import pixel_selection
from src.reporting.visualize import plot_image

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
ps = pixel_selection(df[['Object', 'Label']+band_cols], polygon_id_col='Object', class_col='Label')
ps.get_clusters(method='som', cluster_col='clusters', random_state=0)
df_selection, clusters = ps.get_consistency_analysis(method='som', consistency_col='consistency_results', random_state=0)
df_selection = df_selection.join(df[['X','Y']])
df_selection.to_csv(DATA_PATH+'processed/SOM+SOM.csv') # save results

## run pixel_selection method 2: SOM+Bhattacharyya (pseudo Paris et al. 2019)
ps = pixel_selection(df[['Object', 'Label']+band_cols], polygon_id_col='Object', class_col='Label')
ps.get_clusters(method='som', cluster_col='clusters', random_state=0)
df_selection, clusters = ps.get_consistency_analysis(method='bhattacharyya', consistency_col='consistency_results', random_state=0)
df_selection = df_selection.join(df[['X','Y']])
df_selection.to_csv(DATA_PATH+'processed/SOM+paris.csv') # save results


## run pixel_selection method 3: SOM+K-means -----> Seems to work well
ps = pixel_selection(df[['Object', 'Label']+band_cols], polygon_id_col='Object', class_col='Label')
ps.get_clusters(method='som', cluster_col='clusters', random_state=0)
df_selection, clusters = ps.get_consistency_analysis(method='kmeans', consistency_col='consistency_results', random_state=0)
df_selection = df_selection.join(df[['X','Y']])
df_selection.to_csv(DATA_PATH+'processed/SOM+Kmeans.csv') # save results

## run pixel_selection method 4: K-means+SOM
ps = pixel_selection(df[['Object', 'Label']+band_cols], polygon_id_col='Object', class_col='Label')
ps.get_clusters(method='kmeans', cluster_col='clusters', random_state=0)
df_selection, clusters = ps.get_consistency_analysis(method='som', consistency_col='consistency_results', random_state=0)
df_selection = df_selection.join(df[['X','Y']])
df_selection.to_csv(DATA_PATH+'processed/Kmeans+SOM.csv')  # save results

## run pixel_selection method 4: K-means+K-means ------> Seems to work well
ps = pixel_selection(df[['Object', 'Label']+band_cols], polygon_id_col='Object', class_col='Label')
ps.get_clusters(method='kmeans', cluster_col='clusters', random_state=0)
df_selection, clusters = ps.get_consistency_analysis(method='kmeans', consistency_col='consistency_results', random_state=0)
df_selection = df_selection.join(df[['X','Y']])
df_selection.to_csv(DATA_PATH+'processed/Kmeans+Kmeans.csv')  # save results

## run pixel_selection method 4: K-means+hierarchical clustering (single linkage) ------> Pure garbage
ps = pixel_selection(df[['Object', 'Label']+band_cols], polygon_id_col='Object', class_col='Label')
ps.get_clusters(method='kmeans', cluster_col='clusters', random_state=0)
df_selection, clusters = ps.get_consistency_analysis(method='hierarchical', consistency_col='consistency_results', random_state=0)
df_selection = df_selection.join(df[['X','Y']])
df_selection.to_csv(DATA_PATH+'processed/K-means+hierarchical.csv')  # save results

## run pixel_selection method 2: SOM+minority rejection+Bhattacharyya (pseudo Paris et al. 2019)
ps = pixel_selection(df[['Object', 'Label']+band_cols], polygon_id_col='Object', class_col='Label')
ps.get_clusters(method='som', cluster_col='clusters', identify_dominant_cluster=True, random_state=0)
ps.df = ps.df[ps.df['clusters'].apply(lambda x: x.split('_')[-1])=='True']
df_selection, clusters = ps.get_consistency_analysis(method='bhattacharyya', consistency_col='consistency_results', random_state=0)
df_selection = df.join(df_selection['consistency_results'])
df_selection.to_csv(DATA_PATH+'processed/som+minority_rej+bhattacharyya.csv') # save results

## PURE PARIS
ps = pixel_selection(df[['Object', 'Label']+band_cols], polygon_id_col='Object', class_col='Label')
ps.get_clusters(method='kmeans', cluster_col='clusters', identify_dominant_cluster=True, random_state=0)
ps.df = ps.df[ps.df['clusters'].apply(lambda x: x.split('_')[-1])=='True']
df_selection, clusters = ps.get_consistency_analysis(method='bhattacharyya', consistency_col='consistency_results', random_state=0)
df_selection = df.join(df_selection['consistency_results'])
df_selection.to_csv(DATA_PATH+'processed/kmeans+minority_rej+bhattacharyya.csv') # save results





## reference polygons: 87177 (shrubland), 14545 (rainfed), 69070 (conifers), 8499+8546+8591 (baresoil), 18129 (rice field), 91666 (wetlands), 17492+15930 (irrigated)
id_label = df_selection.groupby(['Object', 'Label']).size().reset_index()
id_label[id_label['Label']=='baresoil']
df_selection['Object'].unique()
objects = df_selection.groupby(['Object', 'Label']).size().sort_values(ascending=False).reset_index()
## plot results
obj = df_selection[df_selection['Object'] == 17492]

obj[['X', 'Y']] = ((obj[['X', 'Y']] - obj[['X', 'Y']].min()) / 10).astype(int)

img = np.array([obj.pivot('X', 'Y', band).values for band in ['X2017.12.21.B4', 'X2017.12.21.B3', 'X2017.12.21.B2']]).swapaxes(0, 1).swapaxes(1, 2).swapaxes(0, 1)

_accepted = (obj.pivot('X', 'Y', 'consistency_results').values == 1).astype(float)
accepted = img*np.array([_accepted for i in range(3)]).swapaxes(0, 1).swapaxes(1, 2).swapaxes(0, 1)

_rejected = (obj.pivot('X', 'Y', 'consistency_results').values != 1).astype(float)
rejected = img*np.array([_rejected for i in range(3)]).swapaxes(0, 1).swapaxes(1, 2).swapaxes(0, 1)

plot_image([np.flip(img, 0), np.flip(rejected, 0), np.flip(accepted, 0)], num_rows=1, figsize=(40, 20), dpi=20)
