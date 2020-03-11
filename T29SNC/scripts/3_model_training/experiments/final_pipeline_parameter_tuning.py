
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
    MBKMeansFilter_reversed
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
FEATURE_RANK_PATH = RESULTS_PATH+'feature_rankings.csv'
random_state=0

## set up classifiers used as filters in anomally detection step
pre_fit_params = [
    ('OwnMethod2', {'filters': [
        ('RandomForestClassifier', RandomForestClassifier(n_estimators=100, random_state=random_state)),
        ('RandomForestClassifier', RandomForestClassifier(n_estimators=50, random_state=random_state)),
        ('DecisionTreeClassifier', DecisionTreeClassifier(random_state=random_state)),
        ('LogisticRegression', LogisticRegression(solver='lbfgs', random_state=random_state, multi_class='auto', max_iter=750)),
        ('MLPClassifier', MLPClassifier(random_state=random_state, max_iter=2000))
    ]})
]

## set up experiment objects
anomally_detection = [
    ('NONE', None, {}),
    ('OwnMethod2', MBKMeansFilter_reversed(),
            {
            'n_splits':[3,4,5], 'granularity':[.1,.25,.5,.75,1],
            'method':['obs_percent'],
            'threshold':[.9,.93,.96,.97,.98,.99]
            }),
]

classifiers_1 = [
    ('RFC', RandomForestClassifier(), {
        'n_estimators':[100, 300, 500, 700, 900, 1100],
        'criterion': ['entropy', 'gini']
        })
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

# drop least important features
features = pd.read_csv(FEATURE_RANK_PATH).iloc[:70,0]
df_bands = df[features.to_list()]

# normalize
znorm = StandardScaler()
df_bands = pd.DataFrame(znorm.fit_transform(df_bands.values), columns=df_bands.columns, index=df_bands.index)

X = df_bands.values
y = df_meta['Megaclasse'].values

# sample data
X, _, y, _ = train_test_split(X, y, train_size=.3, shuffle=True, stratify=y, random_state=random_state)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

## Final model - Parameter tuning
pipelines, param_grids = check_pipelines(
    [anomally_detection, classifiers_1],
    random_state=0,
    n_runs=1
)

fit_params = check_fit_params(pre_fit_params)

model_search = ModelSearchCV(
    pipelines,
    param_grids,
    scoring=scorers,
    refit='accuracy',
    n_jobs=-1,
    cv=cv,
    verbose=1
)
model_search.fit(X,y, **fit_params)

pickle.dump(model_search, open(RESULTS_PATH+'final_pipeline_parameter_tuning.pkl','wb'))
df_results = report_model_search_results(model_search)\
    .sort_values('mean_test_accuracy', ascending=False)
df_results_anomally.to_csv('results_anomally.csv')
