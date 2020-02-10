
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

# classifiers
from sklearn.ensemble import RandomForestClassifier

## configs
DATA_PATH = PROJ_PATH+'/data/DGT/william_data/'
INTERIM_PATH = DATA_PATH+'interim/'
PROCESSED_PATH = DATA_PATH+'processed/'
DATA = INTERIM_PATH+'all_outputs.csv'
random_state=0

## set up experiment objects
feature_selection = [
    ('NONE', None, {}),
#    ('CorrelationBased', CorrelationBasedFeatureSelection(), {
#            'corr_type':['pearson'], 'threshold':[.7, .8, .9]
#        }
#    ),
#    ('Permutation', PermutationRF(), {'n_estimators': [100], 'max_features': [None, 30, 50, 70]}),
    ('RFGini', SelectFromModel(
        estimator=RandomForestClassifier(n_estimators=100, criterion='gini', random_state=0),
        prefit=False), {
            'max_features': [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110]
        }
    ),
#    ('RFEntropy', SelectFromModel(
#        estimator=RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0),
#        prefit=False), {
#            'max_features': [30, 50, 70]
#        }
#    ),
#    ('ReliefF', ReliefF(), {'n_neighbors': [40, 100], 'n_features_to_keep': [30, 50, 70]})
]

classifiers_1 = [
    ('RFC', RandomForestClassifier(n_estimators=100), {})
]


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
#df_results_feature.to_csv('results_feature_selection.csv')
pickle.dump(model_search_feature, open('gini_feature_selection.pkl','wb'))




rfc = RandomForestClassifier(
    n_estimators=100,
    criterion='gini',
    random_state=random_state
)
rfc.fit(X, y)
# Gini feature importance results
pd.Series(rfc.feature_importances_, index=df_bands.columns)\
    .sort_values(ascending=False)

import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="white")

def make_correlation_matrix(df):
    # Compute the correlation matrix
    corr = df.corr()
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=np.bool))
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})


df = df_bands.copy()

mean_band_names = []
for band in ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12',
            'NDVI', 'NDBI', 'NDMIR']:
    mean_band_names.append(band)
    df[band] = df[df.columns[df.columns.str.endswith(band)]].mean(1)

df = df[mean_band_names]

make_correlation_matrix(df)

df['Class'] = y

sns.pairplot(df.sample(frac=.01, random_state=random_state),
    hue="Class",
    palette="husl",
    markers="+",
    corner=True,
    plot_kws={
            'alpha': .3,
        }
    )
sns.despine(
    top=True,
    right=True,
    left=True,
    bottom=True
)

plt.figure(figsize=(16,9))
for c in df['Class'].unique():
    sns.distplot(df_bands[df['Class']==c]['X2018.03.21.B11'], label=c, hist=False, kde_kws={'shade': True})
plt.legend()

plt.figure(figsize=(16,9))
for c in df['Class'].unique():
    sns.distplot(df_bands[df['Class']==c]['X2018.09.27.B2'], label=c, hist=False, kde_kws={'shade': True})
plt.legend()



df = df_bands.copy()
df['Class'] = y
dates = np.unique(
    np.vectorize(lambda x: '.'.join(x[:-1]))\
    (df.columns[df.columns.str.startswith('X20')].str.split('.').values)
)
df_sample = df.sample(frac=.01, random_state=random_state)
for d in dates:
    data = df_sample[
        list(df_sample.columns[df_sample.columns.str.startswith(d)])+['Class']
        ]

    data = data.rename(columns={k: k.split('.')[-1] for k in data.columns})

    g = sns.pairplot(
        data,
        hue="Class",
        palette="husl",
        markers="+",
        corner=True,
        plot_kws={
                'alpha': .3,
            },
        height=1,
        aspect=1
        )
    sns.despine(
        top=True,
        right=True,
        left=True,
        bottom=True
    )
    g.fig.suptitle(d.replace('.', '/').replace('X', ''), fontsize=20)

    plt.savefig(f'reports/pairplot_{d.replace(".", "_").replace("X", "")}.png')

# convert -delay 150 -loop 0 *.png bands.gif
