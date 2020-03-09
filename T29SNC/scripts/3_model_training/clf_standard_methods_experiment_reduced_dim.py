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
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression

# scorers
from sklearn.metrics import SCORERS, make_scorer
from imblearn.metrics import geometric_mean_score


# configs
random_state = 0
DATA_DIR = 'T29SNC/data/preprocessed/2019_02_RS_0_n_features_320.csv'
FEATURE_RANK_PATH = 'T29SNC/results/feature_rankings.csv'
n_features = 70

# read data
df = pd.read_csv(DATA_DIR).dropna()

# split by feature type
df_meta = df[['x','y','Megaclasse']]

# drop least important features
features = pd.read_csv(FEATURE_RANK_PATH).iloc[:n_features,0]
df_bands = df[features.to_list()]


# normalize
znorm = StandardScaler()
df_bands = pd.DataFrame(znorm.fit_transform(df_bands.values), columns=df_bands.columns, index=df_bands.index)

# get data in simple format
X = df_bands.values
y = df_meta['Megaclasse'].values

months = np.array([c.split('_')[1]  for c in df_bands.columns])
months[months=='12'] = '00'
bands  = np.array([c.split('_')[-1] for c in df_bands.columns])
order = np.argsort(np.array([f'{m}_{b}' for m,b in zip(months, bands)]))

# experiment setup
classifiers = [
    ('RFC', RandomForestClassifier(), {
        'n_estimators': [100, 150, 200],
        'criterion': ['gini', 'entropy']
    }),
    ('ABC', AdaBoostClassifier(), {
        'n_estimators': [100, 150, 200],
        'learning_rate': [.01, .005, .001]
    }),
    ('GBC', GradientBoostingClassifier(), {
        'n_estimators': [100, 150, 200],
    }),
    ('LR', LogisticRegression(solver='lbfgs', max_iter=1000), {
        'multi_class': ['ovr', 'multinomial'],
        'penalty': ['l2', 'none']
    }),
]

# setup scorers
def geometric_mean_macro(X, y):
    return geometric_mean_score(X, y, average='macro')
SCORERS['geometric_mean_macro'] = make_scorer(geometric_mean_macro)
scorers = ['accuracy', 'f1_macro', 'geometric_mean_macro']

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
    scoring=scorers,
    refit='accuracy',
    n_jobs=-1,
    cv=cv,
    verbose=1
)
model_search.fit(X,y)

pickle.dump(model_search, open(f'classifier_search_{n_features}_features.pkl','wb'))
df_results_feature = report_model_search_results(model_search)\
    .sort_values('mean_test_accuracy', ascending=False)
