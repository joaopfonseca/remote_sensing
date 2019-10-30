import sys
import os
PROJ_PATH = os.path.realpath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(PROJ_PATH)
print(os.path.realpath(os.path.join(os.path.dirname(__file__), '../')))
import pandas as pd
import numpy as np
from src.preprocess.data_selection import pixel_selection
from src.reporting.visualize import plot_image

## configs
DATA_PATH = PROJ_PATH+'/data/DGT/'
RAW_CSV_PATH = DATA_PATH+'raw/'
MERGED_CSV = DATA_PATH+'interim/all_outputs.csv'
RESULTS_PATH = DATA_PATH+'processed/data_selection_results.csv'

## read merged data
df = pd.read_csv(MERGED_CSV)

## discard unnecessary data cols
labels_coords_cols = ['X','Y','Object','Label']
band_cols = [x for x in df.columns if (x.startswith('X') and len(x)>1)]
df = df[labels_coords_cols+band_cols]

## drop rows with missing values
df = df.dropna()

## Paris implementation
ps = pixel_selection(df[['Object', 'Label']+band_cols], polygon_id_col='Object', class_col='Label')
ps.get_clusters(method='kmeans', cluster_col='clusters', identify_dominant_cluster=True, random_state=0)
ps.df = ps.df[ps.df['clusters'].apply(lambda x: x.split('_')[-1])=='True']
df_selection, clusters = ps.get_consistency_analysis(method='bhattacharyya', consistency_col='status', random_state=0)
df_selection = df.join(df_selection['status'])
df_selection.to_csv(DATA_PATH+'processed/ds_paris.csv') # save results
