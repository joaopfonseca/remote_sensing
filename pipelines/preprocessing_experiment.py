
# the obvious stuff
import pandas as pd
import numpy as np

# utilities
import sys
import os
PROJ_PATH = '.' # temporary
#PROJ_PATH = os.path.realpath(os.path.join(os.path.dirname(__file__), '../../'))
#sys.path.append(PROJ_PATH)
import pickle

# utilities v2
from src.experiment.utils import check_pipelines, check_fit_params
from rlearn.model_selection import ModelSearchCV
from rlearn.tools.reporting import report_model_search_results
from sklearn.model_selection import StratifiedKFold

# data normalization
from sklearn.preprocessing import StandardScaler

# feature selection
from sklearn.feature_selection import SelectFromModel
from src.preprocess.feature_selection import (
    CorrelationBasedFeatureSelection,
    PermutationRF,
)
from src.preprocess.relieff import ReliefF

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

## configs
DATA_PATH = PROJ_PATH+'/data/DGT/william_data/'
INTERIM_PATH = DATA_PATH+'interim/'
PROCESSED_PATH = DATA_PATH+'processed/'
DATA = INTERIM_PATH+'all_outputs.csv'
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
feature_selection = [
    ('NONE', None, {}),
    ('CorrelationBased', CorrelationBasedFeatureSelection(), {
            'corr_type':['pearson'], 'threshold':[.7, .8, .9]
        }
    ),
    ('Permutation', PermutationRF(), {'n_estimators': [100], 'max_features': [None, 30, 50, 70]}),
    ('RFGini', SelectFromModel(
        estimator=RandomForestClassifier(n_estimators=100, criterion='gini', random_state=0),
        prefit=False), {
            'max_features': [30, 50, 70]
        }
    ),
    ('RFEntropy', SelectFromModel(
        estimator=RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0),
        prefit=False), {
            'max_features': [30, 50, 70]
        }
    ),
    ('ReliefF', ReliefF(), {'n_neighbors': [40, 100], 'n_features_to_keep': [30, 50, 70]})
]

anomally_detection = [
    ('NONE', None, {}),
    ('Single', SingleFilter(), {'n_splits':[3,5,7]}),
    ('Consensus', ConsensusFilter(), {'n_splits':[3,4,7]}),
    ('Majority', MajorityVoteFilter(), {'n_splits':[3,4,7]}),
    ('OwnMethod1', MBKMeansFilter(),
            {
            'n_splits':[3,5,7], 'granularity':[.1,1,4],
            'method':['obs_percent'],
            'threshold':[.25, .5, .75]
            }),
    ('OwnMethod2', MBKMeansFilter_reversed(),
            {
            'n_splits':[3,5,7], 'granularity':[.1,1,4],
            'method':['obs_percent'],
            'threshold':[.25, .5, .75]
            }),
    ('PerClassiForest', PerClassiForest(n_estimators=100), {}),
#    ('Paris', ParisDataFiltering(k_max=5), {})
]

classifiers_1 = [
    ('RFC', RandomForestClassifier(n_estimators=100), {})
]

#classifiers_2 = [
#    ('RFC', RandomForestClassifier(n_estimators=100), {})
#]

## read data, normalize and split among feature types
# read and drop missing values (it's not our goal to study imputation methods)
df = pd.read_csv(DATA).dropna()
# split by feature type
df_meta = df.drop(df.columns[df.columns.str.startswith('X201')|df.columns.str.startswith('ND')], axis=1)
df_bands = df.drop(columns=df_meta.columns)
# normalize
znorm = StandardScaler()
df_bands = pd.DataFrame(znorm.fit_transform(df_bands.values), columns=df_bands.columns, index=df_bands.index)

X = df_bands.values
y = df_meta['Label'].values
ids = df_meta['Object'].values

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

## Experiment 1 (feature selection)
pipelines_feature, param_grid_feature = check_pipelines(
    [feature_selection, classifiers_1],
    random_state=0,
    n_runs=1
)

model_search_feature = ModelSearchCV(pipelines_feature, param_grid_feature, n_jobs=-1, cv=cv, verbose=1)
model_search_feature.fit(X,y)

df_results_feature = report_model_search_results(model_search_feature)\
    .sort_values('mean_test_score', ascending=False)
df_results_feature.to_csv('results_feature_selection.csv')
pickle.dump(model_search_feature, open('model_search_feature_selection.pkl','wb'))


## Experiment 2 (anomally detection)
pipelines_anomally, param_grid_anomally = check_pipelines(
    [anomally_detection, classifiers_1],
    random_state=0,
    n_runs=1
)

fit_params_anomally = check_fit_params(anomally_filters)

model_search_anomally = ModelSearchCV(pipelines_anomally, param_grid_anomally, n_jobs=-1, cv=cv, verbose=1)
model_search_anomally.fit(X,y, **fit_params_anomally)

df_results_anomally = report_model_search_results(model_search_anomally)\
    .sort_values('mean_test_score', ascending=False)
df_results_anomally.to_csv('results_anomally.csv')
pickle.dump(model_search_anomally, open('model_search_anomally.pkl','wb'))
