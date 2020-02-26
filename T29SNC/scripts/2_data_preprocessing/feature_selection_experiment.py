
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

# feature selection
from sklearn.feature_selection import SelectFromModel
from src.preprocess.feature_selection import (
    CorrelationBasedFeatureSelection,
    PermutationRF,
)
from src.preprocess.relieff import ReliefF
from src.preprocess.utils import SelectFeaturesFromList


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
            'max_features': [15, 30, 40, 50, 60, 70]
        }
    ),
    ('RFEntropy', SelectFromModel(
        estimator=RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0),
        prefit=False), {
            'max_features': [15, 30, 40, 50, 60, 70]
        }
    ),
    ('ReliefF', ReliefF(), {'n_neighbors': [40, 100], 'n_features_to_keep': [30, 50, 70]})
]

classifiers = [
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

## Experiment 1 (feature selection)
pipelines_feature, param_grid_feature = check_pipelines(
    [feature_selection, classifiers],
    random_state=0,
    n_runs=1
)

model_search_feature = ModelSearchCV(
    pipelines_feature,
    param_grid_feature,
    scoring=scorers,
    refit='accuracy',
    n_jobs=-1,
    cv=cv,
    verbose=1
)
model_search_feature.fit(X,y)

df_results_feature = report_model_search_results(model_search_feature)\
    .sort_values('mean_test_accuracy', ascending=False)
df_results_feature.to_csv('results_feature_selection.csv')
pickle.dump(model_search_feature, open('model_search_feature_selection.pkl','wb'))

best_feature_selection_model = model_search_feature.best_estimator_.named_steps['ReliefF']

features = pd.DataFrame(
    np.array(
        [df_bands.columns,best_feature_selection_model.top_features]
        ).T,
    columns=['feature', 'rank']).sort_values('rank')
features.to_csv(RESULTS_PATH+'feature_rankings.csv', index=False)

## Select optimal number of features
optimal_num_features = [
    (
        'DimReduct',
        SelectFeaturesFromList(feature_rankings=best_feature_selection_model.top_features),
        {'n_features': list(range(1, len(best_feature_selection_model.top_features)+1))}
    )
]

pipelines_dimreduct, param_grid_dimreduct = check_pipelines(
    [optimal_num_features, classifiers],
    random_state=0,
    n_runs=1
)

model_search_dimreduct = ModelSearchCV(
    pipelines_dimreduct,
    param_grid_dimreduct,
    scoring=scorers,
    refit='accuracy',
    n_jobs=-1,
    cv=cv,
    verbose=1
)

model_search_dimreduct.fit(X,y)
