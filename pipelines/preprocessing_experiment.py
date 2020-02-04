import sys
import os
PROJ_PATH = '.'
#PROJ_PATH = os.path.realpath(os.path.join(os.path.dirname(__file__), '../../'))
#sys.path.append(PROJ_PATH)

import pandas as pd
import numpy as np

# data normalization
from sklearn.preprocessing import StandardScaler

# feature selection
from sklearn.feature_selection import SelectFromModel

# data filtering
from sklearn.ensemble import IsolationForest

# classifiers
from sklearn.ensemble import RandomForestClassifier

## configs
DATA_PATH = PROJ_PATH+'/data/DGT/william_data/'
INTERIM_PATH = DATA_PATH+'interim/'
PROCESSED_PATH = DATA_PATH+'processed/'
DATA = INTERIM_PATH+'all_outputs.csv'

## set up classifiers used as filters in anomally detection step
filts = (
    ('RandomForestClassifier', RandomForestClassifier(n_estimators=25, random_state=random_state)),
    ('RandomForestClassifier', RandomForestClassifier(n_estimators=10, random_state=random_state)),
    ('DecisionTreeClassifier', DecisionTreeClassifier(random_state=random_state)),
    ('LogisticRegression', LogisticRegression(solver='lbfgs', random_state=random_state, multi_class='auto', max_iter=750)),
    ('MLPClassifier', MLPClassifier(random_state=random_state, max_iter=2000))
)
single_filter = DecisionTreeClassifier(random_state=random_state)

## set up experiment objects
feature_selection = [
    ('NONE', None, {}),
    ('CorrelationBased', CorrelationBasedFeatureSelection(), {
            'corr_type':['pearson'], 'threshold':[.7, .8, .9]
        }
    ),
    ('Permutation', PermutationRF(), {'n_estimators': [100], 'max_features': [None, 30, 50, 70]}),
    ('RFGini', SelectFromModel(prefit=False), {
            'estimator': [RandomForestClassifier(n_estimators=100, criterion='gini', random_state=0)],
            'max_features': [30, 50, 70]
        }
    ),
    ('RFEntropy', SelectFromModel(prefit=False), {
            'estimator': [RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)],
            'max_features': [30, 50, 70]
        }
    ),
    ('ReliefF', ReliefF(), {'n_neighbors': [40, 100], 'n_features_to_keep': [30, 50, 70]})
]

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
    ('PerClassiForest', PerClassiForest(n_estimators=100), {}),
    ('StandardiForest', IsolationForest(n_estimators=100, contamination=0.1, behavior='new'), {})
    ('Paris', ParisDataFiltering(k_max=5), {})
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
