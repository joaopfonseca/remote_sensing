from copy import deepcopy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# filter classifiers
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

## configs
DATA_PATH = 'data/DGT/'
RAW_CSV_PATH = DATA_PATH+'raw/'
MERGED_CSV = DATA_PATH+'interim/all_outputs.csv'
RESULTS_PATH = DATA_PATH+'processed'

plot_quantization_errors = True
plot_som = False
random_state = 0
n_splits = 10
granularity = 5

## read merged data
df = pd.read_csv(MERGED_CSV)

## discard unnecessary data cols
labels_coords_cols = ['X','Y','Object','Label']
band_cols = [x for x in df.columns if (x.startswith('X') and len(x)>1)]
#band_cols = [x for x in df.columns if (x.startswith('X2018.07') and len(x)>1)]
df = df[labels_coords_cols+band_cols]

## drop rows with missing values
df = df.dropna()


################################################################################
# filter phase
################################################################################

label_encoder = LabelEncoder()

df_no_outliers = df.copy()
X = df_no_outliers[band_cols].values
y = label_encoder.fit_transform(df_no_outliers['Label'])

## filtering classifiers
filters = (
    ('RandomForestClassifier', RandomForestClassifier(random_state=random_state)),
#    ('GradientBoostingClassifier', GradientBoostingClassifier(random_state=random_state)),
    ('DecisionTreeClassifier', DecisionTreeClassifier(random_state=random_state)),
    ('LogisticRegression', LogisticRegression(random_state=random_state)),
#    ('KNeighborsClassifier', KNeighborsClassifier()),
    ('MLPClassifier', MLPClassifier(random_state=random_state))
)


skf = StratifiedKFold(n_splits = n_splits, shuffle=True, random_state=random_state)
splits = []
for _, split_indices in skf.split(X, y):
    splits.append(split_indices)

filter_outputs = {}
for n, split in enumerate(splits):
    print(f'Applying filter {n}')
    for name, clf in filters:
        classifier = deepcopy(clf)
        classifier.fit(X[split], y[split])
        filter_outputs[f'filter_{n}_{name}'] = classifier.predict(X)
        print(f'Applied classifier {name} (part of filter {n})')

pd.DataFrame(filter_outputs).join(pd.Series(y, name='y'))

## mislabel rate
total_filters = len(filter_outputs.keys())
mislabel_rate = (total_filters - np.apply_along_axis(lambda x: x==y, 0, pd.DataFrame(filter_outputs).values).astype(int).sum(axis=1))/total_filters

#df_after_filter = pd.concat([df_no_outliers, pd.DataFrame(filter_outputs, index=df_no_outliers.index)], axis=1)
df_no_outliers['mislabel_rate'] = mislabel_rate
df_no_outliers['status'] = df_no_outliers['mislabel_rate']<0.5
df_no_outliers.to_csv(DATA_PATH+'processed/class_selection_filter.csv')
