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

## normalize band values
#df[band_cols] = 

## run pixel_selection method 1: SOM+SOM
ps = pixel_selection(df[['Object', 'Label']+band_cols], polygon_id_col='Object', class_col='Label')
ps.get_clusters(method='som', cluster_col='clusters', random_state=0)
df_selection, clusters = ps.get_consistency_analysis(method='som', consistency_col='consistency_results', random_state=0)
df_selection.to_csv(RESULTS_PATH) # save results

## run pixel_selection method 2: SOM+(drop minority clusters+Bhattacharyya) (Paris et al. 2019)


## run pixel_selection method 3: SOM+K-means
ps = pixel_selection(df[['Object', 'Label']+band_cols], polygon_id_col='Object', class_col='Label')
ps.get_clusters(method='som', cluster_col='clusters', random_state=0)
df_selection, clusters = ps.get_consistency_analysis(method='kmeans', consistency_col='consistency_results', random_state=0)
df_selection.to_csv(RESULTS_PATH) # save results

## run pixel_selection method 4: K-means+SOM
ps = pixel_selection(df[['Object', 'Label']+band_cols], polygon_id_col='Object', class_col='Label')
ps.get_clusters(method='kmeans', cluster_col='clusters', random_state=0)
df_selection, clusters = ps.get_consistency_analysis(method='som', consistency_col='consistency_results', random_state=0)
df_selection.to_csv(RESULTS_PATH)  # save results

## run pixel_selection method 4: K-means+K-means
ps = pixel_selection(df[['Object', 'Label']+band_cols].iloc[:50000], polygon_id_col='Object', class_col='Label')
ps.get_clusters(method='kmeans', cluster_col='clusters', random_state=0)
df_selection, clusters = ps.get_consistency_analysis(method='kmeans', consistency_col='consistency_results', random_state=0)
df_selection.to_csv(RESULTS_PATH)  # save results






























## William's reference polygons: 71125 (conifers), 8458 (baresoil)
## plot results (majority cluster vs rest)
obj = df_selection[df_selection['Object'] == 8693]
obj[['X','Y']] = ((obj[['X','Y']] - obj[['X','Y']].min()) / 10).astype(int)
obj['consistency'] = obj['clusters'].apply(lambda x: x.split('_')[-1])

majority_index = obj.groupby(['consistency']).size().index[0]

img = np.array([obj.pivot('X', 'Y', band).values for band in ['X2017.12.21.B4', 'X2017.12.21.B3', 'X2017.12.21.B2']]).T

_accepted = (obj.pivot('X','Y','consistency').values==majority_index).T.astype(float)
accepted = img*np.array([_accepted for i in range(3)]).T.swapaxes(0,1)
_rejected = (obj.pivot('X','Y','consistency').values!=majority_index).T.astype(float).T.swapaxes(0,1)
rejected = img*np.array([_rejected for i in range(3)]).T.swapaxes(0,1)

plot_image([img,rejected,accepted], num_rows=1, figsize=(40, 20), dpi=20)





## other results
obj = df_selection[df_selection['Object'] == 8458]

som_architectures = get_2Dcoordinates_matrix((5, 5)).reshape((2, -1))
som_architectures = som_architectures[:, np.apply_along_axis(lambda x: (x != 0).all() and (x != 1).any(), 0, som_architectures)]

obj['cluster_labels'] = find_optimal_architecture_and_cluster(
    ZScoreNormalization(obj.drop(columns=['X', 'Y', 'Object', 'Label', 'clusters', 'consistency_results']), axes=(0))[0].values, 
    som_architectures.T)
majority_index = obj.groupby(['cluster_labels']).size().sort_values(ascending=False).index[0]


#obj['cluster_labels'] = dbscan(
#    ZScoreNormalization(obj.drop(columns=[
#                        'X', 'Y', 'Object', 'Label', 'clusters', 'consistency_results']), axes=(0))[0],
#    eps=4
#)[-1]




obj[['X', 'Y']] = ((obj[['X', 'Y']] - obj[['X', 'Y']].min()) / 10).astype(int)

img = np.array([obj.pivot('X', 'Y', band).values for band in [
               'X2017.12.21.B4', 'X2017.12.21.B3', 'X2017.12.21.B2']]).T

_accepted = (obj.pivot('X', 'Y', 'cluster_labels').values == majority_index
            ).T.astype(float)
accepted = img*np.array([_accepted for i in range(3)]).T.swapaxes(0, 1)
_rejected = (obj.pivot('X', 'Y', 'cluster_labels').values != majority_index
            ).T.astype(float).T.swapaxes(0, 1)
rejected = img*np.array([_rejected for i in range(3)]).T.swapaxes(0, 1)

plot_image([img, rejected, accepted], num_rows=1, figsize=(40, 20), dpi=20)
