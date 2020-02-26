
# core
import pandas as pd
import numpy as np

# utilities
import sys
import os
import pickle

# utilities v2
from src.experiment.utils import check_pipelines, check_fit_params
from rlearn.model_selection import ModelSearchCV
from rlearn.tools.reporting import report_model_search_results
from sklearn.model_selection import StratifiedKFold, train_test_split

# data normalization
from sklearn.preprocessing import StandardScaler

# data filtering
from sklearn.ensemble import IsolationForest
from src.preprocess.data_selection import (
    SingleFilter,
    ConsensusFilter,
    MajorityVoteFilter,
    MBKMeansFilter,
    MBKMeansFilter_reversed,
    PerClassiForest,
    ParisDataFiltering
)

# classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# scorers
from sklearn.metrics import SCORERS, make_scorer
from imblearn.metrics import geometric_mean_score

## configs
DATA_PATH = 'T29SNC/data/preprocessed/2019_02_RS_0_n_features_320.csv'
RESULTS_PATH = 'T29SNC/results/'
random_state=0

## set up classifiers used as filters in anomally detection step
filts = [
    ('RandomForestClassifier', RandomForestClassifier(n_estimators=25, random_state=random_state)),
    ('RandomForestClassifier', RandomForestClassifier(n_estimators=10, random_state=random_state)),
    ('DecisionTreeClassifier', DecisionTreeClassifier(random_state=random_state)),
    ('LogisticRegression', LogisticRegression(solver='lbfgs', random_state=random_state, multi_class='auto', max_iter=750)),
    ('MLPClassifier', MLPClassifier(random_state=random_state, max_iter=2000))
]
single_filter = [DecisionTreeClassifier(random_state=random_state)]

anomally_filters = [
    ('Single', {'filters': single_filter}),
    ('Consensus', {'filters': filts}),
    ('Majority', {'filters': filts}),
    ('OwnMethod1', {'filters': filts}),
    ('OwnMethod2', {'filters': filts}),
]

## set up experiment objects
anomally_detection = [
    ('NONE', None, {}),
    ('Single', SingleFilter(), {'n_splits':[3,4,5,6,7,8]}),
    ('Consensus', ConsensusFilter(), {'n_splits':[3,4,5,6,7,8]}),
    ('Majority', MajorityVoteFilter(), {'n_splits':[3,4,5,6,7,8]}),
    ('OwnMethod1', MBKMeansFilter(),
            {
            'n_splits':[3,4,5,6,7], 'granularity':[.1,.5,1,3,4,5],
            'method':['obs_percent', 'mislabel_rate'],
            'threshold':[.25, .5, .75, .99]
            }),
    ('OwnMethod2', MBKMeansFilter_reversed(),
            {
            'n_splits':[2,3,4,5,6,7], 'granularity':[.5,1,1.5,2,3,4,5],
            'method':['obs_percent', 'mislabel_rate'],
            'threshold':[.25, .5, .75, .99]
            }),
    ('PCiForest', PerClassiForest(n_estimators=100), {}),
#    ('Paris', ParisDataFiltering(k_max=5), {})
]

classifiers_1 = [
    ('RFC', RandomForestClassifier(n_estimators=100), {})
]

## setup scorers
def geometric_mean_macro(X, y):
    return geometric_mean_score(X, y, average='macro')
SCORERS['geometric_mean_macro'] = make_scorer(geometric_mean_macro)
scorers = ['accuracy', 'f1_macro', 'geometric_mean_macro']

## read data, sample, normalize and split among feature types
# read and drop missing values (it's not our goal to study imputation methods)
df = pd.read_csv(DATA_PATH).dropna()

# split by feature type
df_meta = df[['x','y','Megaclasse']]
df_bands = df.drop(columns=df_meta.columns)

# normalize
znorm = StandardScaler()
df_bands = pd.DataFrame(znorm.fit_transform(df_bands.values), columns=df_bands.columns, index=df_bands.index)

X = df_bands.values
y = df_meta['Megaclasse'].values

# sample data
X, _, y, _ = train_test_split(X, y, train_size=.1, shuffle=True, stratify=y, random_state=random_state)


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

## Experiment 2 (anomally detection)
pipelines_anomally, param_grid_anomally = check_pipelines(
    [anomally_detection, classifiers_1],
    random_state=0,
    n_runs=1
)

fit_params_anomally = check_fit_params(anomally_filters)

model_search_anomally = ModelSearchCV(
    pipelines_anomally,
    param_grid_anomally,
    scoring=scorers,
    n_jobs=-1,
    cv=cv,
    verbose=1
)
model_search_anomally.fit(X,y, **fit_params_anomally)

df_results_anomally = report_model_search_results(model_search_anomally)\
    .sort_values('mean_test_score', ascending=False)
df_results_anomally.to_csv('results_anomally.csv')
pickle.dump(model_search_anomally, open('model_search_anomally.pkl','wb'))
