# core
import numpy as np
import pandas as pd

# utilities
import pickle
from src.experiment.utils import check_pipelines#, check_fit_params
from rlearn.model_selection import ModelSearchCV
from rlearn.tools.reporting import report_model_search_results
from sklearn.model_selection import StratifiedKFold

# data normalization
from sklearn.preprocessing import StandardScaler

# classifiers
from sklearn.ensemble import RandomForestClassifier

# scorers
from sklearn.metrics import SCORERS, make_scorer
from imblearn.metrics import geometric_mean_score



# configs
random_state = 0
DATA_DIR = 'T29SNC/data/preprocessed/2019_02.csv'


# read data
df = pd.read_csv(DATA_DIR)
df = df.dropna()

# split by feature type
# split by feature type
df_meta = df[['x','y','Megaclasse']]
df_bands = df.drop(columns=df_meta.columns)

# normalize
znorm = StandardScaler()
df_bands = pd.DataFrame(znorm.fit_transform(df_bands.values), columns=df_bands.columns, index=df_bands.index)

# get data in simple format
X = df_bands.values
y = df_meta.Megaclasse.values.astype(int)

months = np.array([c.split('_')[1]  for c in df_bands.columns])
months[months=='12'] = '00'
bands  = np.array([c.split('_')[-1] for c in df_bands.columns])
order = np.argsort(np.array([f'{m}_{b}' for m,b in zip(months, bands)]))

# util functions
def make_data_cubes(X, order):
    """order is the indices of the ordered bands/months"""
    X_reshaped = np.reshape(
        X[:,order],
        (X.shape[0], 4, 3, int(X.shape[-1]/(12)))
    )
    return X_reshaped

def make_data_arrays(X, order):
    """order is the indices of the ordered bands/months"""
    X_reshaped = np.reshape(
        X[:,order],
        (X.shape[0], 12, int(X.shape[-1]/(12)))
    )
    return X_reshaped


# experiment setup
classifiers = [
    ('RFC', RandomForestClassifier(n_estimators=100), {})
]

SCORERS['geometric_mean_macro'] = make_scorer(
    lambda y_true, y_pred: geometric_mean_score(y_true, y_pred, average='macro')
)

pipelines, param_grid = check_pipelines(
    [classifiers],
    random_state=random_state,
    n_runs=1
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

# run experiment
model_search = ModelSearchCV(
    pipelines,
    param_grid,
    scoring=['accuracy', 'f1_macro', 'geometric_mean_macro'],
    refit='accuracy',
    n_jobs=-1,
    cv=cv,
    verbose=2
)
model_search.fit(X,y)

pickle.dump(model_search, open('baseline.pkl','wb'))
df_results_feature = report_model_search_results(model_search)\
    .sort_values('mean_test_score', ascending=False)
