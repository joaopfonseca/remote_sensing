################################################################################
# Mega SOM test
################################################################################
"""
A rule of thumb to set the size of the grid for a dimensionality
reduction task is that it should contain 5*sqrt(N) neurons
where N is the number of samples in the dataset to analyze.

E.g. if your dataset has 150 samples, 5*sqrt(150) = 61.23
hence a map 8-by-8 should perform well.
"""
import sys
import os
PROJ_PATH = os.path.realpath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(PROJ_PATH)
print(os.path.realpath(os.path.join(os.path.dirname(__file__), '../')))

from minisom import MiniSom
from copy import deepcopy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# filter classifiers
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

## configs
DATA_PATH = PROJ_PATH+'/data/DGT/'
RAW_CSV_PATH = DATA_PATH+'raw/'
MERGED_CSV = DATA_PATH+'interim/all_outputs.csv'
RESULTS_PATH = DATA_PATH+'processed'

som_outlier_detection = False
plot_quantization_errors = True
plot_som = False
random_state = 0
n_splits = 10
granularity = 5
keep_rate = 0.65

## read merged data
df = pd.read_csv(MERGED_CSV)

## discard unnecessary data cols
labels_coords_cols = ['X','Y','Object','Label']
band_cols = [x for x in df.columns if (x.startswith('X') and len(x)>1)]
#band_cols = [x for x in df.columns if (x.startswith('X2018.07') and len(x)>1)]
df = df[labels_coords_cols+band_cols]

## drop rows with missing values
df = df.dropna()


def som_outlier_detection(X, granularity=5):
    global error_treshold
    if granularity>=np.sqrt(X.shape[0]):
        granularity=int(np.sqrt(X.shape[0]))-1
        print(f'Granularity too high for passed dataset, clipping to {granularity}')

    neurons = int(np.sqrt(granularity*np.sqrt(X.shape[0])))

    som = MiniSom(neurons, neurons, X.shape[1], sigma=1, learning_rate=0.6,
                  neighborhood_function='gaussian', random_seed=1)
    som.train_batch(X, X.shape[0]*5, verbose=True)
    def labeller(x):
        win = som.winner(x)
        return f'{win[0]}_{win[1]}'
    labels = np.apply_along_axis(labeller, 1, X).astype(str)
    quantization_errors = np.linalg.norm(som.quantization(X) - X, axis=1)

    #error_treshold = np.percentile(quantization_errors,
    #                               100*(1-outliers_percentage)+5)
    q75, q25 = np.percentile(quantization_errors,[75, 25])
    iqr = q75 - q25
    error_treshold = q75 + (q75 - q25)*2

    is_outlier = quantization_errors > error_treshold
    return is_outlier, quantization_errors, labels, som

labels_list = []
index_list = []
is_outlier_list = []
quant_errors = []
for analysis_label in df['Label'].unique():
    print(f'Label: {analysis_label}')
    df_label = df[df['Label']==analysis_label]
    X = df_label[band_cols].values

    is_outlier, quantization_errors, labels, som = som_outlier_detection(X, granularity)
    index_list.append(df_label.index)
    is_outlier_list.append(is_outlier)
    quant_errors.append(quantization_errors)
    labels_list.append(labels)
    if plot_quantization_errors:
        plt.hist(quantization_errors)
        plt.axvline(error_treshold, color='k', linestyle='--')
        plt.xlabel('error')
        plt.ylabel('frequency')
        plt.show()

    if plot_som:
        plt.figure(figsize=(7, 7))
        # Plotting the response for each pattern in the iris dataset
        plt.pcolor(som.distance_map().T, cmap='bone_r')  # plotting the distance map as background
        plt.colorbar()
        plt.show()

outliers = pd.Series(data=np.concatenate(is_outlier_list), index=np.concatenate(index_list), name='is_outlier')
quants = pd.Series(data=np.concatenate(quant_errors), index=np.concatenate(index_list), name='quantization_errors')
labels_col = pd.Series(data=np.concatenate(labels_list), index=np.concatenate(index_list), name='cluster')

## "candidates" dataset
df_final = df.join(quants).join(outliers).join(labels_col)
#df_final.to_csv(DATA_PATH+'processed/class_selection.csv')

################################################################################
# phase 2
################################################################################

label_encoder = LabelEncoder()

if som_outlier_detection:
    df_no_outliers = df_final[df_final['is_outlier'] == False].copy()
else:
    df_no_outliers = df_final.copy()

X = df_no_outliers[band_cols].values
y = label_encoder.fit_transform(df_no_outliers['Label'])

## filtering classifiers
filters = (
    ('RandomForestClassifier', RandomForestClassifier(random_state=random_state)),
#    ('GradientBoostingClassifier', GradientBoostingClassifier(random_state=random_state)),
    ('DecisionTreeClassifier', DecisionTreeClassifier(random_state=random_state)),
    ('LogisticRegression', LogisticRegression(random_state=random_state)),
#    ('KNeighborsClassifier', KNeighborsClassifier()),
    ('MLPClassifier', MLPClassifier(random_state=random_state))
)


skf = StratifiedKFold(n_splits = n_splits, shuffle=True, random_state=random_state)
splits = []
for _, split_indices in skf.split(X, y):
    splits.append(split_indices)

filter_outputs = {}
for n, split in enumerate(splits):
    print(f'Applying filter {n}')
    for name, clf in filters:
        classifier = deepcopy(clf)
        classifier.fit(X[split], y[split])
        filter_outputs[f'filter_{n}_{name}'] = classifier.predict(X)
        print(f'Applied classifier {name} (part of filter {n})')

pd.DataFrame(filter_outputs).join(pd.Series(y, name='y'))

## mislabel rate
total_filters = len(filter_outputs.keys())
mislabel_rate = (total_filters - np.apply_along_axis(lambda x: x==y, 0, pd.DataFrame(filter_outputs).values).astype(int).sum(axis=1))/total_filters

#df_after_filter = pd.concat([df_no_outliers, pd.DataFrame(filter_outputs, index=df_no_outliers.index)], axis=1)
df_no_outliers['mislabel_rate'] = mislabel_rate



df_cluster_info_grouped = df_no_outliers.groupby(['Label', 'cluster'])\
            .agg({'mislabel_rate':np.mean, 'X':'count'})\
            .sort_values(['mislabel_rate'])
df_cluster_info_A = df_cluster_info_grouped.groupby(['Label']).cumsum().rename(columns={'X':'cumsum'}).drop(columns=['mislabel_rate'])
df_cluster_info_B = df_cluster_info_grouped.groupby(['Label', 'cluster']).agg({'mislabel_rate':np.mean})
df_cluster_info = df_cluster_info_A.join(df_cluster_info_B).join(df_cluster_info_grouped['X'])

thresholds = df_cluster_info.groupby('Label').max()['cumsum']*keep_rate
actual_thresholds = df_cluster_info[df_cluster_info['cumsum']/thresholds>=1]['cumsum'].groupby('Label').min()
df_cluster_info['status'] = df_cluster_info['cumsum']/actual_thresholds<=1

#df_cluster_info['cluster_status'] = df_cluster_info['X'].cumsum()<df_cluster_info['X'].sum()*0.7

print(df_cluster_info.groupby(['Label','status']).agg({'mislabel_rate':np.mean, 'X':np.sum}))

df_results = df_final.join(df_cluster_info['status'], on=['Label', 'cluster'])
df_results.to_csv(DATA_PATH+f'processed/ps_som_gran_{granularity}_n_filter_clf_{len(filters)*n_splits}_keep_rate_{keep_rate}.csv')


################################################################################
# plotting examples to check results
################################################################################
"""
import matplotlib.pyplot as plt
from src.reporting.visualize import plot_image

## reference polygons: 87177 (shrubland), 14545 (rainfed), 69070 (conifers), 8499+8546+8591 (baresoil), 18129 (rice field), 91666 (wetlands), 17492+15930 (irrigated)


## helpers
id_label = df_results.groupby(['Object', 'Label']).size().reset_index()
id_label[id_label['Label']=='baresoil']
df_results['Object'].unique()
objects = df_results.groupby(['Object', 'Label']).size().sort_values(ascending=False).reset_index()


## plot results
obj = df_results[df_results['Object'] == 14545]

obj[['X', 'Y']] = ((obj[['X', 'Y']] - obj[['X', 'Y']].min()) / 10).astype(int)

img = np.array([obj.pivot('X', 'Y', band).values for band in ['X2018.07.29.B4', 'X2018.07.29.B3', 'X2018.07.29.B2']]).swapaxes(0, 1).swapaxes(1, 2).swapaxes(0, 1)

_accepted = (obj.pivot('X', 'Y', 'cluster_status').values == 1).astype(float)
accepted = img*np.array([_accepted for i in range(3)]).swapaxes(0, 1).swapaxes(1, 2).swapaxes(0, 1)

_rejected = (obj.pivot('X', 'Y', 'cluster_status').values != 1).astype(float)
rejected = img*np.array([_rejected for i in range(3)]).swapaxes(0, 1).swapaxes(1, 2).swapaxes(0, 1)

plot_image([np.flip(img, 0), np.flip(rejected, 0), np.flip(accepted, 0)], num_rows=1, figsize=(40, 20), dpi=20)
"""
