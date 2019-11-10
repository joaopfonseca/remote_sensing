import sys
import os
PROJ_PATH = os.path.realpath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(PROJ_PATH)
print(PROJ_PATH)

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from src.reporting.reports import reports

## configs
random_state = 0
DATA_PATH = PROJ_PATH+'/data/DGT/'
RESULTS_PATH = DATA_PATH+'processed'
MERGED_CSV = DATA_PATH+'interim/all_outputs.csv'

pixel_selection_csv  = os.listdir(RESULTS_PATH)
pixel_selection_col  = 'status'


## read merged data and get train and test ids
print('Getting train/test indices...')
df_original = pd.read_csv(MERGED_CSV).sort_values(['X', 'Y']).dropna()
band_cols = [x for x in df_original.columns if (x.startswith('X') and len(x)>1)]

y_total = df_original['Label'].values
train_id, test_id, _, _ = train_test_split(np.arange(len(y_total)),
    y_total,
    test_size=0.33,
    shuffle=True,
    stratify=y_total,
    random_state=random_state
)

y_test = df_original[ 'Label' ].values[test_id]
X_test = df_original[band_cols].values[test_id]

label_encoder = LabelEncoder().fit(y_test)
target_names = {k:v for k, v in enumerate(label_encoder.classes_)}

## train methods
models = {}
for file in pixel_selection_csv[1:3]:
    if file.endswith('.csv'):
        print(f'Starting experiment {file}...')
        df = pd.read_csv(os.path.join(RESULTS_PATH, file)).sort_values(['X', 'Y'])
        try:
            df = df.iloc[train_id].loc[df[pixel_selection_col].astype(float)==1.0]
        except KeyError:
            df = df.iloc[train_id].loc[df['cluster_status'].astype(float)==1.0]

        X = df[band_cols].values
        y = df['Label'].values
        print(f'Training Random Forest...')
        rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
        rf.fit(X, y)
        y_pred = label_encoder.transform(rf.predict(X_test))
        models[file] = reports(label_encoder.transform(y_test), y_pred, target_names)

file = 'no_selection'
print(f'Starting experiment {file}...')
df = pd.read_csv(MERGED_CSV).sort_values(['X', 'Y']).dropna().iloc[train_id]
X = df[band_cols].values
y = df['Label'].values
print(f'Training Random Forest...')
rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
rf.fit(X, y)
y_pred = label_encoder.transform(rf.predict(X_test))
models[file] = reports(label_encoder.transform(y_test), y_pred, target_names)

scores = {}
for name, results in models.items():
    scores[name] = results[-1]

pd.concat(scores).reset_index().pivot('level_0','level_1','Score')\
    .sort_values('ACCURACY', ascending=False)\
    .to_csv('experiment_results.csv')
