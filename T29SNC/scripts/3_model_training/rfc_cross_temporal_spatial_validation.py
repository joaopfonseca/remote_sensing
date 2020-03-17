# core
import numpy as np
import pandas as pd

# utilities
import ast
import pickle
from sklearn.model_selection import train_test_split
from src.reporting.reports import reports
from imblearn.pipeline import Pipeline

# data normalization
from sklearn.preprocessing import StandardScaler

# anomally
from src.preprocess.data_selection import MBKMeansFilter_reversed

# classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# configs
random_state = 0
DATA_PATH = 'T29SNC/data/preprocessed/'
MODELS_PATH = 'T29SNC/models/'
RESULTS_PATH = 'T29SNC/results/'
FEATURE_RANK_PATH = RESULTS_PATH+'feature_rankings_standardized.csv'

# read data
df = pd.read_csv(DATA_PATH+'2019_02_RS_0_n_features_320.csv')
df = df.dropna()

# split by feature type
df_meta = df[['x','y','Megaclasse']]

# drop least important features
features = pd.read_csv(FEATURE_RANK_PATH).iloc[:70,0]
cols_mapper = df.columns.to_series()\
    .apply(lambda x: x.split('_')[1]+'_'+x.split('_')[-1] if len(x.split('_'))==4 else x)\
    .to_dict()
df_bands = df.rename(columns=cols_mapper)[features.to_list()]

# get data in simple format
X = df_bands.values
y = df_meta.Megaclasse.values.astype(int)

#months = np.array([c.split('_')[1]  for c in df_bands.columns])
#months[months=='12'] = '00'
#bands  = np.array([c.split('_')[-1] for c in df_bands.columns])
#order = np.argsort(np.array([f'{m}_{b}' for m,b in zip(months, bands)]))

# get optimal parameters
anomally_exp = pd.read_csv(RESULTS_PATH+'model_search_anomally.csv')
params = {
    k:v
    for k,v in ast.literal_eval(anomally_exp.loc[anomally_exp['mean_test_accuracy'].idxmax(), 'params']).items()
    if not k.startswith('RFC')
}

filts = [
    ('RandomForestClassifier', RandomForestClassifier(n_estimators=25, random_state=random_state)),
    ('RandomForestClassifier', RandomForestClassifier(n_estimators=10, random_state=random_state)),
    ('DecisionTreeClassifier', DecisionTreeClassifier(random_state=random_state)),
    ('LogisticRegression', LogisticRegression(solver='lbfgs', random_state=random_state, multi_class='auto', max_iter=750)),
    ('MLPClassifier', MLPClassifier(random_state=random_state, max_iter=2000))
]


# set up model
znorm = StandardScaler()
ownmethod = MBKMeansFilter_reversed()
rfc = RandomForestClassifier(n_estimators=500, random_state=random_state)

clf = Pipeline([('ZNorm', znorm), ('OwnMethod2',ownmethod), ('RFC', rfc)])
clf.set_params(**params)

## model
clf.fit(X, y, **{'OwnMethod2__filters': filts})
pickle.dump(clf, open(MODELS_PATH+'near_final_clf_.pkl','wb'))

## model2
rfc = RandomForestClassifier(n_estimators=500, random_state=random_state, n_jobs=-1)
rfc.fit(X, y)
pickle.dump(rfc, open(MODELS_PATH+'final_RFC.pkl','wb'))

# ---------------------------------------------------------------------------- #
# Cross Spatial Validation
# ---------------------------------------------------------------------------- #

# read data
df = pd.read_csv(DATA_PATH+'2019_02_RS_1_n_features_320.csv')
df = df.dropna()

# split by feature type
df_meta = df[['x','y','Megaclasse']]

# drop least important features
features = pd.read_csv(FEATURE_RANK_PATH).iloc[:70,0]
cols_mapper = df.columns.to_series()\
    .apply(lambda x: x.split('_')[1]+'_'+x.split('_')[-1] if len(x.split('_'))==4 else x)\
    .to_dict()
df_bands = df.rename(columns=cols_mapper)[features.to_list()]

# get data in simple format
X = df_bands.values
y = df_meta.Megaclasse.values.astype(int)

# make predictions
y_pred = clf.predict(X)
cross_spatial_results = reports(y, y_pred)
pickle.dump(cross_spatial_results, open(RESULTS_PATH+'near_final_cross_spatial_results.pkl','wb'))

# rfc
y_pred = rfc.predict(X)
cross_spatial_results = reports(y, y_pred)
pickle.dump(cross_spatial_results, open(RESULTS_PATH+'near_final_RF_cross_spatial_results.pkl','wb'))

# ---------------------------------------------------------------------------- #
# Cross Temporal Validation
# ---------------------------------------------------------------------------- #

# read data
df = pd.read_csv(DATA_PATH+'2020_01_RS_1_n_features_320.csv')
df = df.dropna()

# split by feature type
df_meta = df[['x','y','Megaclasse']]

# drop least important features
features = pd.read_csv(FEATURE_RANK_PATH).iloc[:70,0]
cols_mapper = df.columns.to_series()\
    .apply(lambda x: x.split('_')[1]+'_'+x.split('_')[-1] if len(x.split('_'))==4 else x)\
    .to_dict()
df_bands = df.rename(columns=cols_mapper)[features.to_list()]

# get data in simple format
X = df_bands.values
y = df_meta.Megaclasse.values.astype(int)

# make predictions
y_pred = clf.predict(X)
cross_temporal_results = reports(y, y_pred)
pickle.dump(cross_temporal_results, open(RESULTS_PATH+'near_final_cross_temporal_results.pkl','wb'))

# rfc
y_pred = rfc.predict(X)
cross_temporal_results = reports(y, y_pred)
pickle.dump(cross_temporal_results, open(RESULTS_PATH+'near_final_RF_cross_temporal_results.pkl','wb'))
