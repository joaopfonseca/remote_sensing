
################################################################################
## Imports and configurations

import sys
import os
PROJ_PATH = '.'
#PROJ_PATH = os.path.realpath(os.path.join(os.path.dirname(__file__), '../../'))
#sys.path.append(PROJ_PATH)

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# feature selection
from sklearn.feature_selection import SelectFromModel
from rfpimp import importances as permutation_importances, plot_importances

# classifiers
from sklearn.ensemble import RandomForestClassifier

# reporting
from src.reporting.reports import reports

## configs
DATA_PATH = PROJ_PATH+'/data/DGT/central_pt/'
RAW_PATH = DATA_PATH+'raw/'
PROCESSED_PATH = DATA_PATH+'processed/'
TRAIN_DATA = RAW_PATH+'training.csv'
TEST_DATA = RAW_PATH+'testing.csv'
LABELS_PATH = RAW_PATH+'Class_legend.txt'

random_state = 0

################################################################################
## read data and preprocess
# read
df_train = pd.read_csv(TRAIN_DATA).drop(columns='Unnamed: 0')
X = df_train.drop(columns='CLASS')
y = df_train['CLASS'].astype(int)
# get feature names and labels
feat_labels = list(X.columns)
class_labels = pd.read_csv(LABELS_PATH, sep='\t', header=None,
    index_col=0)[1].to_dict()
# standardize
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

################################################################################
## feature selection
# Split data into 40% test and 60% training
_X_tr, _X_te, _y_tr, _y_te = train_test_split(X, y, test_size=0.4,
    random_state=random_state)

# Create and train a random forest classifier
clf = RandomForestClassifier(n_estimators=100, n_jobs=-1,
    random_state=random_state)
clf.fit(_X_tr, _y_tr)

# Gini Index Importance Feature Selection Method
gini_imp_feat_sel = SelectFromModel(clf, prefit=True, threshold='.8*mean')
gini_accepted = gini_imp_feat_sel.get_support()

# Permutation
imp = permutation_importances(
    clf,
    pd.DataFrame(_X_te, columns=feat_labels),
    pd.Series(_y_te, name='CLASS')
)
permutation_accepted = (imp['Importance']>0).loc[feat_labels].values

# Keep the ones accepted with both methods
accepted_feats = (gini_accepted.astype(int)+permutation_accepted.astype(int))==2

# save feature selection results
feat_sel_results = pd.DataFrame(
    np.array([gini_accepted, permutation_accepted, accepted_feats]).T,
    index=feat_labels,
    columns=['Gini', 'Permutation', 'Selected']
)
feat_sel_results.to_csv(PROCESSED_PATH+'feature_selection_results.csv')

################################################################################
## test different methods using test set
df_train = pd.read_csv(TRAIN_DATA).drop(columns='Unnamed: 0')
X_train = df_train.drop(columns='CLASS')
y_train = df_train['CLASS'].astype(int)

df_test  = pd.read_csv(TEST_DATA).drop(columns='Unnamed: 0')
X_test = df_test.drop(columns='CLASS')
y_test = df_test['CLASS'].astype(int)

features_selected = pd.read_csv(PROCESSED_PATH+'feature_selection_results.csv')\
    .rename(columns={'Unnamed: 0': 'features'}).set_index('features')
features_selected['Original'] = True

#pd.DataFrame(features_selected[features_selected].count(),
#    columns=['# features used'])\
#    .sort_values('# features used', ascending=False)\
#    .to_csv('feature_selection_count.csv')

# get feature names and labels
feat_labels = list(X_train.columns)
class_labels = pd.read_csv(LABELS_PATH, sep='\t', header=None,
    index_col=0)[1].to_dict()

# standardize
scaler = StandardScaler()
scaler.fit(X_train)
scaler.transform(X_train.values, copy=False)
scaler.transform(X_test.values, copy=False)

scores = []
for method in features_selected.columns:
    rfc = RandomForestClassifier(100, random_state=0)
    features = features_selected[method]
    _X_tr = X_train[features[features].index]
    _y_tr = y_train.copy()
    rfc.fit(_X_tr, _y_tr)
    _X_te = X_test[features[features].index]
    _y_te = y_test.copy()
    _y_pred = rfc.predict(_X_te)
    scores.append(reports(_y_te, _y_pred)[-1].rename({'Score': method}))

pd.DataFrame(features_selected[features_selected].count(),
    columns=['# features used'])\
    .join(pd.concat(scores, 1).T)\
    .sort_values('# features used', ascending=False)\
    .rename(index={'Selected':'Intersect'})\
    .to_csv('feature_selection_results.csv')


################################################################################
## define noise introduction procedure

## define filters

## define classifiers

## setup and run experiment

## save results

## setup and train models using hyperparameters with best scores

## get testing dataset scores
