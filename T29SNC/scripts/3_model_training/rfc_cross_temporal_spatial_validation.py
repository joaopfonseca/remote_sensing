# core
import numpy as np
import pandas as pd

# utilities
import ast
import pickle
from sklearn.model_selection import train_test_split
from src.reporting.reports import reports

# data normalization
from sklearn.preprocessing import StandardScaler

# anomally
from src.preprocess.data_selection import MBKMeansFilter_reversed

# classifiers
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression

# configs
random_state = 0
DATA_PATH = 'T29SNC/data/preprocessed/2019_02_RS_0.csv'
MODELS_PATH = 'T29SNC/models/'
RESULTS_PATH = 'T29SNC/results/'
FEATURE_RANK_PATH = RESULTS_PATH+'feature_rankings.csv'

# read data
df = pd.read_csv(DATA_PATH)
df = df.dropna()

# split by feature type
df_meta = df[['x','y','Megaclasse']]

# drop least important features
features = pd.read_csv(FEATURE_RANK_PATH).iloc[:70,0]
df_bands = df[features.to_list()]

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

# set up model
znorm = StandardScaler()
ownmethod = MBKMeansFilter_reversed()
rfc = RandomForestClassifier(n_estimators=500, random_state=random_state)

## Random Forest
rfc.fit(X, y)
pickle.dump(rfc, open(MODELS_PATH+'random_forest_.pkl','wb'))
