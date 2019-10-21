import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import (
    cross_validate,
    StratifiedKFold
)
from sklearn.metrics import (
    make_scorer,
    confusion_matrix,
    accuracy_score,
    f1_score
)
from imblearn.metrics import geometric_mean_score

from src.reporting.reports import reports
from src.reporting.visualize import plot_image

## configs
DATA_PATH = 'data/DGT/'
RESULTS_PATH = DATA_PATH+'processed'

scorers = {
    'accuracy': make_scorer(accuracy_score),
    'f_score_mean_macro_score': make_scorer(f1_score, average='macro'),
    'geometric_mean_macro_score': make_scorer(geometric_mean_score, average='macro')
}
cv = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)

results = {}
for file in os.listdir(RESULTS_PATH):
    if file.split('.')[-1] == 'csv':
        df = pd.read_csv(os.path.join(RESULTS_PATH, file))
        df = df[df['consistency_results'].astype(float)==1.0]
        band_cols = [x for x in df.columns if (x.startswith('X') and len(x)>1)]
        X = df[band_cols].values
        y = df['Label']
        pca = PCA(n_components=30, whiten=True)
        X_pca = pca.fit_transform(X)
        rf = RandomForestClassifier(n_estimators=100, random_state=0)
        model_cv = cross_validate(
            estimator=rf, X=X_pca, y=y,
            cv=cv, scoring=scorers,
            return_estimator=True, n_jobs=-1,
        )
        results[file.split('.')[0]] = model_cv

cols = ['OA', 'G-Mean', 'F-Score']
values = []
index = []
for sel_strat in results.keys():
    accuracy = np.mean(results[sel_strat]['test_accuracy'])
    gmean    = np.mean(results[sel_strat]['test_geometric_mean_macro_score'])
    fscore   = np.mean(results[sel_strat]['test_f_score_mean_macro_score'])
    index.append(sel_strat)
    values.append([accuracy, gmean, fscore])

classification_results = pd.DataFrame(data=values, index=index, columns=cols)
classification_results.to_csv('reports/selection_strategies_scores.csv')

## confusion matrix here



polygons = [87177, 14545, 69070, 8499, 8546, 8591, 18129, 91666, 17492, 15930]
labels   = ['shrubland', 'rainfed', 'conifers', 'baresoil', 'baresoil', 'baresoil', 'rice field', 'wetlands', 'irrigated', 'irrigated']
for file in os.listdir(RESULTS_PATH):
    if file.split('.')[-1] == 'csv':
        df = pd.read_csv(os.path.join(RESULTS_PATH, file))
        for key, label in zip(polygons, labels):
            obj = df[df['Object'] == key]
            obj[['X', 'Y']] = ((obj[['X', 'Y']] - obj[['X', 'Y']].min()) / 10).astype(int)

            img = np.array([obj.pivot('X', 'Y', band).values for band in ['X2017.12.21.B4', 'X2017.12.21.B3', 'X2017.12.21.B2']]).swapaxes(0, 1).swapaxes(1, 2).swapaxes(0, 1)

            _accepted = (obj.pivot('X', 'Y', 'consistency_results').values == 1).astype(float)
            accepted = img*np.array([_accepted for i in range(3)]).swapaxes(0, 1).swapaxes(1, 2).swapaxes(0, 1)

            _rejected = (obj.pivot('X', 'Y', 'consistency_results').values != 1).astype(float)
            rejected = img*np.array([_rejected for i in range(3)]).swapaxes(0, 1).swapaxes(1, 2).swapaxes(0, 1)

            plot_image([np.flip(img, 0), np.flip(rejected, 0), np.flip(accepted, 0)], num_rows=1, figsize=(40, 7), dpi=60)
            plt.suptitle(f'Object: {key}, Label: {label}, Strategy: {file.split(".")[0]}', fontsize=24)
            plt.savefig(f'reports/RF_DGT_data_selection_results/object_{key}_label_{label}_strategy_{file.split(".")[0]}.png')
