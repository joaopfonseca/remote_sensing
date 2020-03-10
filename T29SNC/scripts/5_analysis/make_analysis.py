# core
import pandas as pd
import numpy as np
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt

import pickle
import ast

from sklearn.preprocessing import StandardScaler
from src.experiment.utils import geometric_mean_macro
from rlearn.tools.reporting import report_model_search_results

# configs
RESULTS_PATH = 'T29SNC/results/'
ANALYSIS_PATH = 'T29SNC/analysis/'
DATA_FILE = 'T29SNC/data/preprocessed/2019_02_RS_0_n_features_320.csv'

# Get feature importance rankings
model_search_feature_selection = pickle.load(
    open('T29SNC/results/model_search_feature_selection_which_method.pkl','rb')
)
fs = model_search_feature_selection.best_estimator_.steps[0][-1]
scores = StandardScaler().fit_transform(-np.sort(-fs.feature_scores).reshape(-1,1))

feature_rankings = pd.read_csv(RESULTS_PATH+'feature_rankings.csv')
feature_rankings['month'] = feature_rankings['feature']\
    .apply(lambda x: x.split('_')[1] if x.startswith('20') else None)
feature_rankings['band'] = feature_rankings['feature']\
    .apply(lambda x: x.split('_')[-1] if x.startswith('20') else x)
feature_rankings['norm_score'] = scores

# top and least 10
features = feature_rankings\
    .sort_values('norm_score', ascending=False)\
    .loc[:10,'feature'].rename('Top 10').reset_index(drop=True).to_frame()

bot10 = feature_rankings\
    .sort_values('norm_score', ascending=True)\
    .loc[:10,'feature'].reset_index(drop=True)
features['Bottom 10'] = bot10
features.to_csv(ANALYSIS_PATH+'top_bottom_10_features.csv', index=False)

# average scores per band
scores_per_band = feature_rankings[
    feature_rankings['band'].apply(lambda x: '_' not in x)
].groupby('band').mean()[['norm_score']]
scores_per_band.to_csv(ANALYSIS_PATH+'scores_per_band.csv')

# average scores per month
scores_per_month = feature_rankings.groupby('month').mean()[['norm_score']]
scores_per_month.to_csv(ANALYSIS_PATH+'scores_per_month.csv')

# Dimensionality reduction results
results_dimreduct = report_model_search_results(model_search_feature_selection)
results_dimreduct['Method'] = results_dimreduct['models'].apply(lambda x: x.split('|')[0])
mapper = {
    'mean_test_accuracy':'Accuracy',
    'mean_test_f1_macro':'F-score',
    'mean_test_geometric_mean_macro':'G-mean'
}
dimreduct_scores = results_dimreduct.rename(columns=mapper)\
    .groupby('Method').max()\
    .drop(columns=['models', 'mean_fit_time'])\
    .rename(index={'NONE': 'None'})\
    .applymap('{:,.3f}'.format)
dimreduct_scores.to_csv(ANALYSIS_PATH+'dimreduct_scores.csv')

#results_dimreduct['n_features'] = results_dimreduct[
#    results_dimreduct['Method'].isin(['ReliefF', 'RFEntropy'])
#    ]['params']\
#    .apply(lambda x: x['RFEntropy__max_features'] \
#        if 'RFEntropy__max_features' in x \
#        else x['ReliefF__n_features_to_keep'])


# Land Cover Change confusion matrix
cos_changes = pickle.load(open(RESULTS_PATH+'cos_changes_y_true_18_y_pred_15.pkl', 'rb'))
labels = {v:k.split('.')[-1].strip() for k,v in pickle.load(open('T29SNC/labels_dict_cos18.pkl','rb')).items()}
mapper_cols = dict([(k,labels[k]) if k in labels else (k,k) for k in cos_changes[1].columns])
mapper_index = dict([(k,labels[k]) if k in labels else (k,k) for k in cos_changes[1].index])
cm = cos_changes[1].rename(columns=mapper_cols, index=mapper_index)
cm.iloc[-1,-1] = cos_changes[-1]['Score'].map('{:,.3f}'.format)['ACCURACY']
cm.to_csv(ANALYSIS_PATH+'land_cover_change_2015_2018.csv')

# Filter results
df_anomally = pd.read_csv(RESULTS_PATH+'model_search_anomally.csv')
df_anomally['Method'] = df_anomally['models'].apply(lambda x: x.split('|')[0])
mapper = {
    'mean_test_accuracy':'Accuracy',
    'mean_test_f1_macro':'F-score',
    'mean_test_geometric_mean_macro':'G-mean'
}
df_anomally_scores = df_anomally.rename(columns=mapper)\
    .groupby('Method').max()\
    .drop(columns=['params','models', 'mean_fit_time'])\
    .rename(index={'NONE': 'None', 'PCiForest':'iForest'})\
    .applymap('{:,.3f}'.format)\
    .reindex(['None', 'Single', 'Consensus', 'Majority', 'iForest', 'OwnMethod1', 'OwnMethod2'])
df_anomally_scores.to_csv(ANALYSIS_PATH+'anomally_detection_results.csv')


# Cross-spatial classification results (70 features)
df_classifier = pd.read_csv(RESULTS_PATH+'classifier_search_70_features.csv')
df_classifier_scores = df_classifier.rename(columns=mapper)\
    .groupby('models').max()\
    .drop(columns=['params','Unnamed: 0', 'n_estimators', 'mean_fit_time'])\
    .applymap('{:,.3f}'.format)
df_classifier_scores.index = df_classifier_scores.index+'70'

# Cross-spatial classification results (320 features)
df_classifier2 = pd.read_csv(RESULTS_PATH+'classifier_search_320_features.csv')
df_classifier_scores2 = df_classifier2.rename(columns=mapper)\
    .groupby('models').max()\
    .drop(columns=['params','Unnamed: 0', 'n_estimators', 'mean_fit_time'])\
    .applymap('{:,.3f}'.format)
df_classifier_scores2.index = df_classifier_scores2.index+'320'

# plotting
df_dimreduct = pd.read_csv(RESULTS_PATH+'results_dimreduct.csv').rename(columns=mapper)
df_dimreduct['n_features'] = df_dimreduct['params'].apply(
    lambda x: ast.literal_eval(x)['DimReduct__n_features']
)

sns.set()
sns.lineplot(
    x='Number of Features',
    y='Score',
    hue='Metric',
    style='Metric',
    data=df_dimreduct[['n_features', 'Accuracy', 'F-score', 'G-mean']]\
        .rename(columns={'n_features': 'Number of Features'})\
        .melt('Number of Features')\
        .rename(columns={'value':'Score', 'variable':'Metric'})
).get_figure().savefig(ANALYSIS_PATH+'n_features_scores.png', dpi=360)
plt.clf()

sns.set(style='darkgrid')
plt.figure(figsize=(10,5))
plt.ylabel('Normalized Feature Importance')
plt.xlabel('Features')
plt.subplots_adjust(left=.07, bottom=.35, right=.98, top=.98)
pd.concat([feature_rankings.iloc[:10], feature_rankings.iloc[-10:]])\
    .set_index('feature')['norm_score'].plot.bar().get_figure()\
    .savefig(ANALYSIS_PATH+'top_bottom_10_features_scores.png', dpi=360)
plt.clf()

df = pd.read_csv(DATA_FILE)
df['Megaclasse'] = df['Megaclasse'].map(labels)

# plot most important feature
plt.figure(figsize=(14,7))
for c in df['Megaclasse'].unique():
    feature = feature_rankings.iloc[0]['feature']
    col = StandardScaler().fit_transform(df[feature].values.reshape(-1, 1))
    sns.distplot(pd.Series(col[df['Megaclasse']==c].squeeze(), name=feature), label=c, hist=False, kde_kws={'shade': True})
plt.xlim(2*-1.96, 2*1.96)
plt.savefig(ANALYSIS_PATH+'top_feature_dist.png', dpi=360)

# plot least important feature
plt.figure(figsize=(14,7))
for c in df['Megaclasse'].unique():
    feature = feature_rankings.iloc[-1]['feature']
    col = StandardScaler().fit_transform(df[feature].values.reshape(-1, 1))
    sns.distplot(pd.Series(col[df['Megaclasse']==c].squeeze(), name=feature), label=c, hist=False, kde_kws={'shade': True})
plt.xlim(2*-1.96, 2*1.96)
plt.savefig(ANALYSIS_PATH+'bottom_feature_dist.png', dpi=360)
