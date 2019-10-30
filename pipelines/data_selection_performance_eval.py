import sys
import os
PROJ_PATH = os.path.realpath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(PROJ_PATH)
print(os.path.realpath(os.path.join(os.path.dirname(__file__), '../')))

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

pixel_selection_csv  = ['kmeans+minority_rej+bhattacharyya.csv', 'class_selection.csv', 'class_selection_filter.csv']
pixel_selection_cols = ['consistency_results', 'cluster_status', 'status']


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
for file, pixel_selection_col in zip(pixel_selection_csv, pixel_selection_cols):
    print(f'Starting experiment {file.split(".")[0]}...')
    df = pd.read_csv(os.path.join(RESULTS_PATH, file)).sort_values(['X', 'Y'])
    df = df.iloc[train_id].loc[df[pixel_selection_col].astype(float)==1.0]

    X = df[band_cols].values
    y = df['Label'].values
    print(f'Training Random Forest...')
    rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    rf.fit(X, y)
    y_pred = label_encoder.transform(rf.predict(X_test))
    models[file.split('.')[0]] = {
        'estimator': rf,
        'reports': reports(label_encoder.transform(y_test), y_pred, target_names)
    }
