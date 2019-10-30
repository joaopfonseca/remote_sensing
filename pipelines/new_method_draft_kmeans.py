from copy import deepcopy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

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
DATA_PATH = 'data/DGT/'
RAW_CSV_PATH = DATA_PATH+'raw/'
MERGED_CSV = DATA_PATH+'interim/all_outputs.csv'
RESULTS_PATH = DATA_PATH+'processed'

random_state = 0
n_splits = 10
granularity = 3
keep_rate = 0.65

## read merged data
df = pd.read_csv(MERGED_CSV)

## discard unnecessary data cols
labels_coords_cols = ['X','Y','Object','Label']
#band_cols = [x for x in df.columns if (x.startswith('X') and len(x)>1)]
band_cols = [x for x in df.columns if (x.startswith('X2018.07') and len(x)>1)]
df = df[labels_coords_cols+band_cols]

## drop rows with missing values
df = df.dropna()


def KMeans_outlier_detection(X, granularity=5, random_state=None):
    global error_treshold
    if granularity>=np.sqrt(X.shape[0]):
        granularity=int(np.sqrt(X.shape[0]))-1
        print(f'Granularity too high for passed dataset, clipping to {granularity}')

    k = int(granularity*np.sqrt(X.shape[0]))

    kmeans = KMeans(k, random_state=random_state)
    labels = kmeans.fit_predict(X).astype(str)
    return labels, kmeans


labels_list = []
index_list  = []
for analysis_label in df['Label'].unique():
    print(f'Label: {analysis_label}')
    df_label = df[df['Label']==analysis_label]
    X = df_label[band_cols].values

    labels, kmeans = KMeans_outlier_detection(X, granularity, random_state)
    index_list.append(df_label.index)
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
df_final.to_csv(DATA_PATH+'processed/class_selection.csv')

label_encoder = LabelEncoder()

df_no_outliers = df_final#[df_final['is_outlier'] == False]
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
mislabel_rate = (total_filters - np.apply_along_axis(lambda x: x==y, 0, pd.DataFrame(filter_outputs).values).astype(int).sum(axis=1))/40

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
df_cluster_info['cluster_status'] = df_cluster_info['cumsum']/actual_thresholds<=1

#df_cluster_info['cluster_status'] = df_cluster_info['X'].cumsum()<df_cluster_info['X'].sum()*0.7

print(df_cluster_info.groupby(['Label','cluster_status']).agg({'mislabel_rate':np.mean, 'X':np.sum}))

df_results = df_final.join(df_cluster_info['cluster_status'], on=['Label', 'cluster'])
df_results.to_csv(DATA_PATH+f'processed/ps_kmeans_gran_{granularity}_n_filter_clf_{len(filters)*n_splits}_keep_rate_{keep_rate}.csv')
