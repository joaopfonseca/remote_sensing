"""
TODO:
    - Visualize SOM
"""

import os
import pandas as pd
import numpy as np
from src.preprocess.data_selection import pixel_selection
from src.reporting.visualize import plot_image



## configs
DATA_PATH = 'data/DGT/'
RAW_CSV_PATH = DATA_PATH+'raw/'
MERGED_CSV = DATA_PATH+'interim/all_outputs.csv'
RESULTS_PATH = DATA_PATH+'processed'

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

## Label-wide pixel selection method
df_results = pixel_selection(df[['Label']+band_cols], polygon_id_col='Label', som_architecture=(100,100))\
        .get_clusters(method='som', cluster_col='clusters', random_state=random_state)

clusters_labels = df_results[['clusters', 'Label']].drop_duplicates().set_index('clusters')
clusters = df_results.groupby(['clusters']).mean().drop(columns=['Label']).join(clusters_labels).reset_index()

#pixels_per_cluster = df_results.groupby(['clusters']).size().sort_values(ascending=False).to_frame().rename(columns={0:'consistency_results'})
#keep_percentile = pixels_per_cluster.quantile(0.65).iloc[0]


df_consistency = pixel_selection(clusters[['Label']+band_cols], polygon_id_col='Label', k_max=2)\
        .get_clusters(method='bhattacharyya', cluster_col='clusters2', identify_dominant_cluster=True, random_state=random_state)
df_consistency['consistency_results'] = df_consistency['clusters2'].apply(lambda x: x.split('_')[-1]=='True')

df_results = df.join(df_results['clusters']).join((pixels_per_cluster>keep_percentile), on='clusters')
df_results.to_csv(DATA_PATH+'processed/class_selection.csv')
