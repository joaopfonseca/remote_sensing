import pandas as pd
import numpy as np
from copy import deepcopy
from collections import Counter
from sklearn.metrics import calinski_harabasz_score
from sklearn.cluster import (
    KMeans,
    AgglomerativeClustering,
    MiniBatchKMeans
)
from minisom import MiniSom
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedShuffleSplit
)
from imblearn.under_sampling.base import BaseCleaningSampler

from .utils import get_2Dcoordinates_matrix

from sklearn.ensemble import IsolationForest

################################################################################
# iForest
################################################################################

class PerClassiForest(BaseCleaningSampler):
    def __init__(self,
            n_estimators=100,
            max_samples='auto',
            contamination=0.1,
            max_features=1.0,
            bootstrap=False,
            n_jobs=None,
            behaviour='new',
            random_state=None,
            verbose=0,
            warm_start=False
        ):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.behaviour = behaviour
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.iForest_ = IsolationForest(
            n_estimators = self.n_estimators,
            max_samples = self.max_samples,
            contamination = self.contamination,
            max_features = self.max_features,
            bootstrap = self.bootstrap,
            n_jobs = self.n_jobs,
            behaviour = self.behaviour,
            random_state = self.random_state,
            verbose = self.verbose,
            warm_start = self.warm_start
        )

    def fit(self, X, y):
        self.iforests = {}
        #outcome = np.zeros(X.shape[0])
        for label in np.unique(y):
            iforest = deepcopy(self.iForest_)
            #outcome[y==label] = iforest.fit_predict(X[y==label])
            self.iforests[label] = iforest.fit(X[y==label])
        return self

    def resample(self, X, y):
        outcome = np.zeros(X.shape[0])
        for label in np.unique(y):
            outcome[y==label] = self.iforests[label].predict(X[y==label])
        return X[outcome==1], y[outcome==1]

    def _fit_resample(self, X, y):
        self.iforests = {}
        outcome = np.zeros(X.shape[0])
        for label in np.unique(y):
            iforest = deepcopy(self.iForest_)
            outcome[y==label] = iforest.fit_predict(X[y==label])
            self.iforests[label] = iforest.fit(X[y==label])
        return X[outcome==1], y[outcome==1]

    def fit_resample(self, X, y):
        return self._fit_resample(X, y)




################################################################################
# Paris new
################################################################################
class ParisDataFiltering(BaseCleaningSampler):
    def __init__(self, k_max=6, random_state=None):
        self.k_max = k_max
        self.random_state = random_state

    def fit(self, X, y, ids=None):
        return self

    def resample(self, X, y, ids=None):
        if ids is None:
            ids=y

        status = np.zeros(y.shape)*np.nan
        cluster = np.zeros(y.shape)*np.nan
        for pol_id in np.unique(ids):
            _labels = _find_optimal_k_and_cluster(X=X[ids==pol_id], k_max=self.k_max, random_state=self.random_state)
            cluster[ids==pol_id] = _labels
            status[ids==pol_id] = get_dominant_pixels(_labels)

        final_status = np.zeros(y.shape).astype(bool)
        for label in np.unique(y):
            _final_status = final_status[y==label]
            _clu = cluster[y==label][status[y==label].astype(bool)]
            _ids = ids[y==label][status[y==label].astype(bool)]
            _ban = X[y==label][status[y==label].astype(bool)]

            unique_ids = np.unique(_ids)
            b_dist = np.zeros(unique_ids.shape)*np.nan
            for i, polygon_cluster_id in enumerate(unique_ids):
                b = _ban[_ids==polygon_cluster_id]
                b_dist[i] = Bhattacharyya(_ban, b)
            ranks = b_dist.argsort().argsort()
            accepted = unique_ids[ranks<int(np.ceil(ranks.shape[0]*.65))]
            _final_status[status[y==label].astype(bool)] = np.isin(_ids, accepted)
            final_status[y==label] = _final_status

        return X[final_status]

    def _fit_resample(self, X, y, ids=None):
        return self.transform(X, y, ids)

    def fit_resample(self, X, y, ids=None):
        return self.resample(X, y, ids)


def _find_optimal_k_and_cluster(X, k_max=12, random_state=None):
    label_list = []
    CH_score = []
    for k in range(2,k_max+1):
        if X.shape[0] > k:
            labels = KMeans(n_clusters=k, n_init=10, max_iter=300, random_state=random_state, n_jobs=None).fit_predict(X)
            score = calinski_harabasz_score(X, labels)
            label_list.append(labels)
            CH_score.append(score)
    return label_list[np.argmax(CH_score)]

def get_dominant_pixels(labels):
    return labels==Counter(labels).most_common(1)[0][0]

def Bhattacharyya(a, b):

    a_mean = np.expand_dims(a.mean(axis=0), 1)
    a_cov  = np.cov(a.T)
    b_mean = np.expand_dims(b.mean(axis=0), 1)
    b_cov  = np.cov(b.T)

    sigma = (a_cov + b_cov)/2
    sigma_inv = np.linalg.inv(sigma)

    term_1 = (1/8)*np.dot(np.dot((a_mean-b_mean).T,sigma_inv),(a_mean-b_mean))
    #term_2 = (1/2)*np.log(np.linalg.det(sigma)/np.sqrt(np.linalg.det(a_cov)*np.linalg.det(b_cov)))
    #return float(np.squeeze(term_1+term_2))
    return term_1



################################################################################
# Filter based methods
################################################################################

class MBKMeansFilter(BaseCleaningSampler):
    """My own method"""
    def __init__(self, n_splits=5, granularity=5, method='obs_percent', threshold=0.5, random_state=None):
        assert method in ['obs_percent', 'mislabel_rate'], 'method must be either \'obs_percent\', \'mislabel_rate\''
        super().__init__(sampling_strategy='all')
        self.n_splits = n_splits
        self.granularity = granularity
        self.method = method
        self.threshold = threshold
        self.random_state = random_state

    def _fit_resample(self, X, y, filters):
        #assert X.shape[0]==y.shape[0], 'X and y must have the same length.'
        ## cluster data
        #print('n_splits:', self.n_splits, ', granularity:', self.granularity, ', method:', self.method, ', threshold:', self.threshold, ', random_state:', self.random_state)
        self.filters = deepcopy(filters)
        index = np.arange(len(y))
        clusters_list = []
        index_list  = []
        self.kmeans = {}
        for analysis_label in np.unique(y):
            label_indices = index[y==analysis_label]
            X_label = X[y==analysis_label]
            clusters, kmeans = self._KMeans_clustering(X_label)
            self.kmeans[analysis_label] = kmeans
            index_list.append(label_indices)
            clusters_list.append(clusters)

        ## cluster labels
        cluster_col = pd.Series(
            data=np.concatenate(clusters_list),
            index=np.concatenate(index_list),
            name='cluster')\
            .sort_index()

        ## apply filters
        label_encoder = LabelEncoder()
        y_ = label_encoder.fit_transform(y)

        self.stratifiedkfold = StratifiedKFold(n_splits = self.n_splits, shuffle=True, random_state=self.random_state)
        self.filter_list = {}
        filter_outputs   = {}
        for n, (_, split) in enumerate(self.stratifiedkfold.split(X, y_)):
            for name, clf in self.filters:
                classifier = deepcopy(clf)
                classifier.fit(X[split], y_[split])
                filter_outputs[f'filter_{n}_{name}'] = classifier.predict(X)
                self.filter_list[f'{n}_{name}'] = classifier

        ## mislabel rate
        total_filters = len(filter_outputs.keys())
        mislabel_rate = (total_filters - \
            np.apply_along_axis(
                lambda x: x==y_, 0, pd.DataFrame(filter_outputs).values)\
                .astype(int).sum(axis=1)
                )/total_filters

        ## crunch data
        mislabel_col = pd.Series(data=mislabel_rate, index=index, name='mislabel_rate')
        y_col = pd.Series(data=y, index=index, name='y')
        df = cluster_col.to_frame().join(y_col).join(mislabel_col)
        df['count'] = 1
        df_cluster_info_grouped = df.groupby(['y', 'cluster'])\
                    .agg({'mislabel_rate':np.mean, 'count':'count'})\
                    .sort_values(['mislabel_rate'])
        df_cluster_info_A = df_cluster_info_grouped.groupby(['y']).cumsum()\
            .rename(columns={'count':'cumsum'}).drop(columns=['mislabel_rate'])
        df_cluster_info = df_cluster_info_grouped.join(df_cluster_info_A)

        if self.method=='mislabel_rate':
            df_cluster_info['status'] = df_cluster_info['mislabel_rate']<=self.threshold
        elif self.method=='obs_percent':
            thresholds = df_cluster_info.groupby('y').max()['cumsum']*self.threshold
            actual_thresholds = df_cluster_info[
                    df_cluster_info['cumsum']/thresholds>=1
                ]['cumsum'].groupby('y').min()
            df_cluster_info['status'] = df_cluster_info['cumsum']/actual_thresholds<=1

        # always accept cluster with lowest mislabel rate for each class by default
        index_keys = df_cluster_info.reset_index().groupby('y').apply(
            lambda x: x.sort_values('mislabel_rate').iloc[0]
            )[['y','cluster']].values
        df_cluster_info.loc[[tuple(i) for i in index_keys], 'status'] = True

        results = df.join(df_cluster_info['status'], on=['y','cluster'])

        self.status = results['status'].values
        return X[self.status], y[self.status]

    def fit(self, X, y, filters):
        """Fits filter to X, y."""
        self._fit_resample(X, y, filters)
        return self

    def resample(self, X, y):
        index = np.arange(len(y))
        clusters_list = []
        index_list  = []
        for analysis_label in np.unique(y):
            label_indices = index[y==analysis_label]
            X_label = X[y==analysis_label]

            clusters = self.kmeans[analysis_label].predict(X_label)
            index_list.append(label_indices)
            clusters_list.append(clusters)

        ## cluster labels
        cluster_col = pd.Series(
            data=np.concatenate(clusters_list),
            index=np.concatenate(index_list),
            name='cluster')\
            .sort_index()

        ## apply filters
        label_encoder = LabelEncoder()
        y_ = label_encoder.fit_transform(y)

        filter_outputs   = {}
        for name, classifier in self.filter_list.items():
            filter_outputs[f'filter_{name}'] = classifier.predict(X)

        ## mislabel rate
        total_filters = len(filter_outputs.keys())
        mislabel_rate = (total_filters - \
            np.apply_along_axis(
                lambda x: x==y_, 0, pd.DataFrame(filter_outputs).values)\
                .astype(int).sum(axis=1)
                )/total_filters

        ## crunch data
        mislabel_col = pd.Series(data=mislabel_rate, index=index, name='mislabel_rate')
        y_col = pd.Series(data=y, index=index, name='y')
        df = cluster_col.to_frame().join(y_col).join(mislabel_col)
        df['count'] = 1
        df_cluster_info_grouped = df.groupby(['y', 'cluster'])\
                    .agg({'mislabel_rate':np.mean, 'count':'count'})\
                    .sort_values(['mislabel_rate'])
        df_cluster_info_A = df_cluster_info_grouped.groupby(['y']).cumsum()\
            .rename(columns={'count':'cumsum'}).drop(columns=['mislabel_rate'])
        df_cluster_info = df_cluster_info_grouped.join(df_cluster_info_A)

        if self.method=='mislabel_rate':
            df_cluster_info['status'] = df_cluster_info['mislabel_rate']<=self.threshold
        elif self.method=='obs_percent':
            thresholds = df_cluster_info.groupby('y').max()['cumsum']*self.threshold
            actual_thresholds = df_cluster_info[
                    df_cluster_info['cumsum']/thresholds>=1
                ]['cumsum'].groupby('y').min()
            df_cluster_info['status'] = df_cluster_info['cumsum']/actual_thresholds<=1

        # always accept cluster with lowest mislabel rate for each class by default
        index_keys = df_cluster_info.reset_index().groupby('y').apply(
            lambda x: x.sort_values('mislabel_rate').iloc[0]
            )[['y','cluster']].values
        df_cluster_info.loc[[tuple(i) for i in index_keys], 'status'] = True

        results = df.join(df_cluster_info['status'], on=['y','cluster'])
        self.status = results['status'].values
        return X[self.status], y[self.status]

    def fit_resample(self, X, y, filters):
        return self._fit_resample(X, y, filters)

    def _KMeans_clustering(self, X):
        """Private function to..."""
        if self.granularity>=np.sqrt(X.shape[0]):
            self.granularity=int(np.sqrt(X.shape[0]))-1
        k = int(self.granularity*np.sqrt(X.shape[0]))
        k = k if k>=1 else 1
        kmeans = MiniBatchKMeans(k, max_iter=5*k, tol=0, max_no_improvement=400, random_state=self.random_state)
        labels = kmeans.fit_predict(X).astype(str)
        return labels, kmeans

class EnsembleFilter(BaseCleaningSampler):
    """Identifying Mislabeled Training Data, by Brodley and Friedl (1999)"""
    def __init__(self, n_splits=4, threshold=0.5, random_state=None):
        super().__init__(sampling_strategy='all')
        self.n_splits = n_splits
        self.threshold = threshold
        self.random_state = random_state

    def _fit_resample(self, X, y, filters):
        self.filters = deepcopy(filters)

        label_encoder = LabelEncoder()
        y_ = label_encoder.fit_transform(y)

        ## run filter
        self.filter_list = {}
        filter_outputs = {f'filter_{name}':np.zeros((y.shape))-1 for name, _ in self.filters}
        self.stratifiedkfold = StratifiedKFold(n_splits = self.n_splits, shuffle=True, random_state=self.random_state)
        for n, (train_indices, test_indices) in enumerate(self.stratifiedkfold.split(X, y_)):
            for name, clf in self.filters:
                classifier = deepcopy(clf)
                classifier.fit(X[train_indices], y_[train_indices])
                filter_outputs[f'filter_{name}'][test_indices] = classifier.predict(X[test_indices])
                self.filter_list[f'{n}_{name}'] = classifier
        ## mislabel rate
        total_filters = len(filter_outputs.keys())
        mislabel_rate = (total_filters - \
            np.apply_along_axis(
                lambda x: x==y_, 0, pd.DataFrame(filter_outputs).values)\
                .astype(int).sum(axis=1)
                )/total_filters
        ## filter data
        self.status = mislabel_rate<=self.threshold
        return X[self.status], y[self.status]

    def fit(self, X, y, filters):
        self._fit_resample(X, y, filters)
        return self

    def resample(self, X, y):

        label_encoder = LabelEncoder()
        y_ = label_encoder.fit_transform(y)

        ## run filter
        indices = []
        filter_outputs = {f'filter_{name}':np.zeros((y.shape))-1 for name, _ in self.filters}
        for n, (train_indices, test_indices) in enumerate(self.stratifiedkfold.split(X, y_)):
            for name in dict(self.filters).keys():
                filter_outputs[name][test_indices] = self.filter_list[f'{n}_{name}'].predict(X[test_indices])

        ## mislabel rate
        total_filters = len(filter_outputs.keys())
        mislabel_rate = (total_filters - \
            np.apply_along_axis(
                lambda x: x==y_, 0, pd.DataFrame(filter_outputs).values)\
                .astype(int).sum(axis=1)
                )/total_filters
        ## filter data
        self.status = mislabel_rate<=self.threshold
        return X[self.status], y[self.status]

    def fit_resample(self, X, y, filters):
        return self._fit_resample(X, y, filters)

class ChainFilter(BaseCleaningSampler):
    """Own method"""
    def __init__(self, filter_obj, stopping_criteria='manual', tol=None, max_iter=40, random_state=None):
        assert stopping_criteria in ['auto', 'manual'],  '`stopping_criteria` must be either `auto` or `manual`'
        if stopping_criteria=='auto': assert tol, '`tol` must be defined while `stopping_criteria` is defined as `auto`'
        self.filter_methods = [deepcopy(filter_obj) for _ in range(max_iter)]
        self.random_state = random_state
        self.tol = tol
        self.max_iter = max_iter
        self.stopping_criteria = stopping_criteria

    def _fit_resample(self, X, y, filters):
        self.filters = deepcopy(filters)
        X_nnf, y_nnf = X.copy(), y.copy()
        self.filter_list = {}
        for n, filter in enumerate(self.filter_methods):
            filter = filter.fit(X_nnf, y_nnf, self.filters)
            X_nnf, y_nnf = filter.resample(X, y)
            self.filter_list[n] = filter
            if n!=0 and self.stopping_criteria=='auto':
                not_changed = dict(Counter(self.filter_list[n-1].status == self.filter_list[n].status))
                percent_changes = not_changed[False]/sum(not_changed.values())
                print(f'Percentage of status changes: {percent_changes*100}%')
                if percent_changes<=self.tol:
                    break

        self.final_filter = filter
        return X_nnf, y_nnf

    def fit_resample(self, X, y, filters):
        return self._fit_resample(X, y, filters)

class ConsensusFilter(EnsembleFilter):
    """Identifying Mislabeled Training Data, by Brodley and Friedl (1999)"""
    def __init__(self, n_splits=4, random_state=None):
        super().__init__(n_splits=n_splits, threshold=1-.9e-15, random_state=random_state)

class MajorityVoteFilter(EnsembleFilter):
    """Identifying Mislabeled Training Data, by Brodley and Friedl (1999)"""
    def __init__(self, n_splits=4, random_state=None):
        super().__init__(n_splits=n_splits, threshold=.5, random_state=random_state)

class SingleFilter(EnsembleFilter):
    """Identifying Mislabeled Training Data, by Brodley and Friedl (1999)"""
    def __init__(self, n_splits=4, random_state=None):
        super().__init__(n_splits=n_splits, threshold=.5, random_state=random_state)

    def fit_resample(self, X, y, filters):
        if type(filters)==list: filters = [(filters[0].__class__.__name__,filters[0])]
        else: filters = [(filters.__class__.__name__,filters)]
        return super()._fit_resample(X, y, filters)

class MBKMeansFilter_reversed(BaseCleaningSampler):
    """My own method"""
    def __init__(self, n_splits=5, granularity=5, method='obs_percent', threshold=0.5, random_state=None):
        assert method in ['obs_percent', 'mislabel_rate'], 'method must be either \'obs_percent\', \'mislabel_rate\''
        super().__init__(sampling_strategy='all')
        self.n_splits = n_splits
        self.granularity = granularity
        self.method = method
        self.threshold = threshold
        self.random_state = random_state

    def _fit_resample(self, X, y, filters):
        #assert X.shape[0]==y.shape[0], 'X and y must have the same length.'
        ## cluster data
        #print('n_splits:', self.n_splits, ', granularity:', self.granularity, ', method:', self.method, ', threshold:', self.threshold, ', random_state:', self.random_state)
        self.filters = deepcopy(filters)
        index = np.arange(len(y))
        clusters_list = []
        index_list  = []
        self.kmeans = {}
        for analysis_label in np.unique(y):
            label_indices = index[y==analysis_label]
            X_label = X[y==analysis_label]
            clusters, kmeans = self._KMeans_clustering(X_label)
            self.kmeans[analysis_label] = kmeans
            index_list.append(label_indices)
            clusters_list.append(clusters)

        ## cluster labels
        cluster_col = pd.Series(
            data=np.concatenate(clusters_list),
            index=np.concatenate(index_list),
            name='cluster')\
            .sort_index()

        ## apply filters
        label_encoder = LabelEncoder()
        y_ = label_encoder.fit_transform(y)

        self.stratifiedkfold = StratifiedKFold(n_splits = self.n_splits, shuffle=True, random_state=self.random_state)
        self.filter_list = {}
        filter_outputs   = {f'filter_{name}':np.zeros((y.shape))-1 for name, _ in self.filters}
        for n, (train_indices, test_indices) in enumerate(self.stratifiedkfold.split(X, y_)):
            for name, clf in self.filters:
                classifier = deepcopy(clf)
                classifier.fit(X[train_indices], y_[train_indices])
                filter_outputs[f'filter_{name}'][test_indices] = classifier.predict(X[test_indices])
                self.filter_list[f'{n}_{name}'] = classifier

        ## mislabel rate
        total_filters = len(filter_outputs.keys())
        mislabel_rate = (total_filters - \
            np.apply_along_axis(
                lambda x: x==y_, 0, pd.DataFrame(filter_outputs).values)\
                .astype(int).sum(axis=1)
                )/total_filters

        ## crunch data
        mislabel_col = pd.Series(data=mislabel_rate, index=index, name='mislabel_rate')
        y_col = pd.Series(data=y, index=index, name='y')
        df = cluster_col.to_frame().join(y_col).join(mislabel_col)
        df['count'] = 1
        df_cluster_info_grouped = df.groupby(['y', 'cluster'])\
                    .agg({'mislabel_rate':np.mean, 'count':'count'})\
                    .sort_values(['mislabel_rate'])
        df_cluster_info_A = df_cluster_info_grouped.groupby(['y']).cumsum()\
            .rename(columns={'count':'cumsum'}).drop(columns=['mislabel_rate'])
        df_cluster_info = df_cluster_info_grouped.join(df_cluster_info_A)

        if self.method=='mislabel_rate':
            df_cluster_info['status'] = df_cluster_info['mislabel_rate']<=self.threshold
        elif self.method=='obs_percent':
            thresholds = df_cluster_info.groupby('y').max()['cumsum']*self.threshold
            actual_thresholds = df_cluster_info[
                    df_cluster_info['cumsum']/thresholds>=1
                ]['cumsum'].groupby('y').min()
            df_cluster_info['status'] = df_cluster_info['cumsum']/actual_thresholds<=1

        # always accept cluster with lowest mislabel rate for each class by default
        index_keys = df_cluster_info.reset_index().groupby('y').apply(
            lambda x: x.sort_values('mislabel_rate').iloc[0]
            )[['y','cluster']].values
        df_cluster_info.loc[[tuple(i) for i in index_keys], 'status'] = True

        results = df.join(df_cluster_info['status'], on=['y','cluster'])

        self.status = results['status'].values
        return X[self.status], y[self.status]

    def fit(self, X, y, filters):
        """Fits filter to X, y."""
        self._fit_resample(X, y, filters)
        return self

    def resample(self, X, y):
        index = np.arange(len(y))
        clusters_list = []
        index_list  = []
        for analysis_label in np.unique(y):
            label_indices = index[y==analysis_label]
            X_label = X[y==analysis_label]

            clusters = self.kmeans[analysis_label].predict(X_label)
            index_list.append(label_indices)
            clusters_list.append(clusters)

        ## cluster labels
        cluster_col = pd.Series(
            data=np.concatenate(clusters_list),
            index=np.concatenate(index_list),
            name='cluster')\
            .sort_index()

        ## apply filters
        label_encoder = LabelEncoder()
        y_ = label_encoder.fit_transform(y)

        filter_outputs   = {}
        for name, classifier in self.filter_list.items():
            filter_outputs[f'filter_{name}'] = classifier.predict(X)

        ## mislabel rate
        total_filters = len(filter_outputs.keys())
        mislabel_rate = (total_filters - \
            np.apply_along_axis(
                lambda x: x==y_, 0, pd.DataFrame(filter_outputs).values)\
                .astype(int).sum(axis=1)
                )/total_filters

        ## crunch data
        mislabel_col = pd.Series(data=mislabel_rate, index=index, name='mislabel_rate')
        y_col = pd.Series(data=y, index=index, name='y')
        df = cluster_col.to_frame().join(y_col).join(mislabel_col)
        df['count'] = 1
        df_cluster_info_grouped = df.groupby(['y', 'cluster'])\
                    .agg({'mislabel_rate':np.mean, 'count':'count'})\
                    .sort_values(['mislabel_rate'])
        df_cluster_info_A = df_cluster_info_grouped.groupby(['y']).cumsum()\
            .rename(columns={'count':'cumsum'}).drop(columns=['mislabel_rate'])
        df_cluster_info = df_cluster_info_grouped.join(df_cluster_info_A)

        if self.method=='mislabel_rate':
            df_cluster_info['status'] = df_cluster_info['mislabel_rate']<=self.threshold
        elif self.method=='obs_percent':
            thresholds = df_cluster_info.groupby('y').max()['cumsum']*self.threshold
            actual_thresholds = df_cluster_info[
                    df_cluster_info['cumsum']/thresholds>=1
                ]['cumsum'].groupby('y').min()
            df_cluster_info['status'] = df_cluster_info['cumsum']/actual_thresholds<=1

        # always accept cluster with lowest mislabel rate for each class by default
        index_keys = df_cluster_info.reset_index().groupby('y').apply(
            lambda x: x.sort_values('mislabel_rate').iloc[0]
            )[['y','cluster']].values
        df_cluster_info.loc[[tuple(i) for i in index_keys], 'status'] = True

        results = df.join(df_cluster_info['status'], on=['y','cluster'])
        self.status = results['status'].values
        return X[self.status], y[self.status]

    def fit_resample(self, X, y, filters):
        return self._fit_resample(X, y, filters)

    def _KMeans_clustering(self, X):
        """Private function to..."""
        if self.granularity>=np.sqrt(X.shape[0]):
            self.granularity=int(np.sqrt(X.shape[0]))-1
        k = int(self.granularity*np.sqrt(X.shape[0]))
        k = k if k>=1 else 1
        kmeans = MiniBatchKMeans(k, max_iter=5*k, tol=0, max_no_improvement=400, random_state=self.random_state)
        labels = kmeans.fit_predict(X).astype(str)
        return labels, kmeans

## Algorithms that require testing/debugging/edition

class YuanGuanZhu(BaseCleaningSampler):
    """
    Novel mislabeled training data detection algorithm, Yuan, Guan, Zhu et al. (2018)
    Filters used in paper: naive Bayes, decision tree, and 3-Nearest Neighbor
    """
    def __init__(self, n_splits=3, t=40, method='majority', random_state=None):
        """method: `majority` or `consensus`"""
        assert method in ['majority', 'consensus'], '`method` must be either `majority` or `minority`.'
        if   method == 'majority':  method = 'MFMF'
        elif method == 'consensus': method = 'CFMF'
        super().__init__(sampling_strategy='all')
        self.t = t
        self.method = method
        self.n_splits = 3
        self.random_state = random_state
        self.composite_filter = CompositeFilter(
            method=self.method,
            n_splits=self.n_splits,
            random_state=self.random_state
        )

    def _fit_resample(self, X, y, filters):
        self.filters = deepcopy(filters)
        _sfk = StratifiedKFold(n_splits = self.t, shuffle=True, random_state=self.random_state)
        statuses = np.zeros(y.shape)
        for _, subset in _sfk.split(X, y):
            compfilter = deepcopy(self.composite_filter)
            compfilter.fit(X[subset],y[subset], self.filters)
            statuses[subset] = compfilter.status
        self.status = statuses
        return X[self.status], y[self.status]

    def fit_resample(self, X, y, filters):
        return self._fit_resample(X, y, filters)

class CompositeFilter(BaseCleaningSampler):
    """
    Based on "Novel mislabeled training data detection algorithm",
    Yuan, Guan, Zhu et al. (2018).
    `method`: `MFMF`, `CFCF`, `CFMF`, `MFCF`
    """
    def __init__(self, method='MFMF', n_splits=4, random_state=None):
        assert  len(method)==4\
            and method[-2:] in ['MF', 'CF']\
            and method[:2] in ['MF', 'CF'], \
            'Invalid `method` value passed.'

        super().__init__(sampling_strategy='all')
        self.method = method
        self.n_splits = n_splits
        self.random_state = random_state

    def _fit_resample(self, X, y, filters):
        self.filters = deepcopy(filters)
        if self.method.startswith('MF'): self.threshold_1 = .5
        else: self.threshold_1 = 1-.9e-15

        if self.method.endswith('MF'): self.threshold_2 = .5
        else: self.threshold_2 = 1-.9e-15

        label_encoder = LabelEncoder()
        y_ = label_encoder.fit_transform(y)

        ## run filter
        self.filter_list = {}
        voted_outputs_1 = {}
        indices = []
        self.stratifiedkfold = StratifiedKFold(n_splits = self.n_splits, shuffle=True, random_state=self.random_state)
        for n, (train_indices, test_indices) in enumerate(self.stratifiedkfold.split(X, y_)):
            filter_outputs = {}
            for name, clf in self.filters:
                classifier = deepcopy(clf)
                classifier.fit(X[train_indices], y_[train_indices])
                filter_outputs[f'filter_{name}'] = classifier.predict(X)
                self.filter_list[f'{n}_{name}'] = classifier
            total_filters = len(filter_outputs.keys())
            voted_outputs_1[n] = ((total_filters - \
                np.apply_along_axis(
                    lambda x: x==y_, 0, pd.DataFrame(filter_outputs).values)\
                    .astype(int).sum(axis=1)
                    )/total_filters) <= self.threshold_1

        ## mislabel rate
        total_votes = len(voted_outputs_1.keys())
        mislabel_rate = (pd.DataFrame(voted_outputs_1).values\
                .astype(int).sum(axis=1))/total_votes
        ## filter data
        self.status = mislabel_rate<=self.threshold_2
        return X[self.status], y[self.status]

    def fit(self, X, y, filters):
        self._fit_resample(X, y, filters)
        return self

    def resample(self, X, y):
        if self.method.startswith('MF'): self.threshold_1 = .5
        else: self.threshold_1 = 1-.9e-15

        if self.method.endswith('MF'): self.threshold_2 = .5
        else: self.threshold_2 = 1-.9e-15

        label_encoder = LabelEncoder()
        y_ = label_encoder.fit_transform(y)

        ## run filter
        voted_outputs_1 = {}
        for n, (train_indices, test_indices) in enumerate(self.stratifiedkfold.split(X, y_)):
            filter_outputs = {}
            for name, clf in self.filters:
                filter_outputs[f'filter_{name}'] = self.filter_list[f'{n}_{name}'].predict(X)

            total_filters = len(filter_outputs.keys())
            voted_outputs_1[n] = ((total_filters - \
                np.apply_along_axis(
                    lambda x: x==y_, 0, pd.DataFrame(filter_outputs).values)\
                    .astype(int).sum(axis=1)
                    )/total_filters) <= self.threshold_1

        ## mislabel rate
        total_votes = len(voted_outputs_1.keys())
        mislabel_rate = (pd.DataFrame(voted_outputs_1).values\
                .astype(int).sum(axis=1))/total_votes
        ## filter data
        self.status = mislabel_rate<=self.threshold_2
        return X[self.status], y[self.status]

    def fit_resample(self, X, y, filters):
        return self._fit_resample(X, y, filters)

################################################################################
# OLD PARIS VERSION (DO NOT USE)
################################################################################

class pixel_selection:
    def __init__(self, df, polygon_id_col, class_col=None, som_architecture=None, k_max=12):
        """df must have only band values, polygon_id_col and class_col"""
        assert type(df)==pd.DataFrame, 'df needs to be of type `pd.DataFrame`.'
        assert type(polygon_id_col)==str and type(class_col) in [str, type(None)], 'Both polygon_id_col and class_col need to be of type `str`.'
        assert polygon_id_col in df.columns, f'{polygon_id_col} not in dataframe.'
        self.methods = ['som', 'bhattacharyya', 'kmeans', 'hierarchical']
        if not hasattr(self, '_previous_cluster_col'): self._previous_cluster_col = False
        self._polygon_id = polygon_id_col
        self.class_col = class_col
        self.df = df.sort_values(by=self._polygon_id)
        self.k = k_max
        if som_architecture:
            self.som_architectures = np.expand_dims(np.array(som_architecture), 0)
        else:
            self.som_architectures = get_2Dcoordinates_matrix((5,5)).reshape((2,-1))
            self.som_architectures = self.som_architectures[:,np.apply_along_axis(lambda x: (x!=0).all() and (x!=1).any(), 0, self.som_architectures)].T

        if self.df[self._polygon_id].dtype == np.dtype('O'):
            self.is_string_identifier = True
            self.label_encoder = LabelEncoder().fit(self.df['Label'])
            self.df[self._polygon_id] = self.label_encoder.transform(self.df[self._polygon_id])
        else:
            self.is_string_identifier = False

        if class_col: drop_cols = [self._polygon_id, self.class_col]
        else: drop_cols = [self._polygon_id]

        polygon_list = np.split(self.df.drop(columns=drop_cols), np.where(np.diff(self.df[self._polygon_id]))[0]+1)
        # drop polygons with too few pixels to be relevant for classification
        self._polygon_list = [x for x in polygon_list]# if len(x)>=10]

    def get_clusters(self, method='som', cluster_col='clusters', identify_dominant_cluster=False, random_state=None):
        """stage 1"""
        assert method in self.methods, f'Method {method} not implemented. Possible options are {self.methods}'
        assert self._previous_cluster_col or method!='bhattacharyya', f'bhattacharyya method should only be used for consistency analysis.'

        if method == 'som':
            # Polygon clustering (SOM)
            self.som_list = []
            labels = []
            indices = []
            total = len(self._polygon_list) # testing
            i=1
            for polygon in self._polygon_list:
                print(f'Clustering process: {i}/{total}'); i+=1
                indices.append(polygon.index)
                _labels, som = SOM_find_optimal_architecture_and_cluster(polygon.values, self.som_architectures, random_state)
                self.som_list.append(som)
                # generally you will want to use get_dominant_pixels only if the majority cluster is being passed for consistency analysis
                if identify_dominant_cluster:
                    labels.append(get_dominant_pixels(_labels))
                else:
                    labels.append(_labels)

        elif method == 'bhattacharyya':
            labels = []
            indices = []

            for polygon in self._polygon_list:
                a = self._df[self._df[self._previous_cluster_col].isin(polygon.index)]\
                    .drop(columns=[self._polygon_id, self._previous_polygon_id, self._previous_cluster_col])\
                    .values
                clusters_per_label = list(polygon.index)
                pre_indices = []
                pre_labels  = []
                for clust in clusters_per_label:
                    b = self._df[self._df[self._previous_cluster_col]==clust]\
                        .drop(columns=[self._polygon_id, self._previous_polygon_id, self._previous_cluster_col])\
                        .values
                    distance = Bhattacharyya(a, b)
                    pre_indices.append([clust])
                    pre_labels.append([distance])
                indices_labels = np.array([pre_indices,pre_labels]).squeeze().T
                indices_labels = indices_labels[indices_labels[:,1].astype(float).argsort()]
                percentile_65 = int(indices_labels.shape[0]*.65)
                indices_labels[:percentile_65,1] = True
                indices_labels[percentile_65:,1] = False
                labels.append(indices_labels[:,1])
                indices.append(indices_labels[:,0].astype(str))
            self.labels = labels
            self.indices = indices
            #indices = np.expand_dims(indices_labels[:,0], 1).astype(str).tolist()
            #labels  = np.expand_dims(indices_labels[:,1], 1).tolist()

        elif method in ['kmeans', 'hierarchical']:
            labels = []
            indices = []
            total = len(self._polygon_list)  # testing
            i = 1
            for polygon in self._polygon_list:
                print(f'Clustering process: {i}/{total}')
                i += 1
                indices.append(polygon.index)
                _labels = find_optimal_k_and_cluster(X=polygon.values, k_max=self.k, method=method, random_state=random_state)
                # generally you will want to use get_dominant_pixels only if the majority cluster is being passed for consistency analysis
                if identify_dominant_cluster:
                    labels.append(get_dominant_pixels(_labels))
                else:
                    labels.append(_labels)

        else:
            raise ValueError('method not yet implemented')

        self.cluster_col = cluster_col
        clusters = pd.Series(data=np.concatenate(labels), index=np.concatenate(indices), name=self.cluster_col)
        self.df = self.df.join(clusters)
        self.df[self.cluster_col] = self.df[self._polygon_id].astype(str)+'_'+self.df[self.cluster_col].astype(str)
        return self.df

    def get_consistency_analysis(self, consistency_col, method='som', class_col=None, cluster_col=None, random_state=None, som_architecture=None, k_max=None):
        """
        stage 2
        - SOM: Runs clustering based on Kohonen self-organizing maps
        - Bhattacharyya: Distance based selection (keeps 65% of the clusters closest to the "centroid of centroids")
        """
        if class_col: self.class_col = class_col
        if cluster_col: self.cluster_col = cluster_col
        assert self.cluster_col in self.df.columns, f'No columns with cluster id detected ({self.cluster_col}). Run self.get_clusters or manually add column with cluster values (pass column name on `cluster_col`)'
        assert type(self.cluster_col)!=type(None), '`cluster_col` is not defined.'
        assert method in self.methods, f'Method {method} not implemented. Possible options are {self.methods}'
        assert self.class_col in self.df.columns, f'{self.class_col} not in dataframe.'

        self._previous_polygon_id  = deepcopy(self._polygon_id)
        self._previous_cluster_col = deepcopy(self.cluster_col)
        self._df = deepcopy(self.df)

        pre_class_wide_clusters = self.df[[self.cluster_col, self.class_col]].drop_duplicates().set_index(self.cluster_col)
        class_wide_clusters = self.df.drop(columns=[self._polygon_id, self.class_col]).groupby([self.cluster_col]).mean()
        class_wide_clusters = class_wide_clusters.join(pre_class_wide_clusters)

        self.__init__(class_wide_clusters, self.class_col)

        if som_architecture:
            self.som_architectures = np.expand_dims(np.array(som_architecture), 0)
        else:
            self.som_architectures = get_2Dcoordinates_matrix((5,5)).reshape((2,-1))
            self.som_architectures = self.som_architectures[:,np.apply_along_axis(lambda x: (x!=0).all() and (x!=1).any(), 0, self.som_architectures)].T

        if k_max:
            self.k = k_max
        else:
            self.k = 2

        cluster_info = self.get_clusters(method=method, cluster_col=consistency_col, identify_dominant_cluster=True, random_state=random_state)
        mapper = cluster_info[consistency_col].apply(lambda x: x.split('_')[-1]=='True')\
            .astype(int).to_dict()
        self._df[consistency_col] = self._df[self._previous_cluster_col].map(mapper)
        return self._df, cluster_info

def SOM_clustering(X, grid_shape, random_state=None):
    # setup SOM
    som = MiniSom(grid_shape[0],grid_shape[1],X.shape[1],sigma=0.8,learning_rate=0.6,random_seed=random_state)
    # fit SOM
    som.train_random(X, 2000)
    # assign labels to node
    labels = np.apply_along_axis(som.winner, 1, X).astype(str)
    return np.char.add(labels[:,0], labels[:,1]), som

def SOM_find_optimal_architecture_and_cluster(X, nodes, random_state=None):
    label_list = []
    CH_score = []
    som_list = []
    for architecture in nodes:
        if X.shape[0]>=architecture[0]*architecture[1]:
            labels, som = SOM_clustering(X,architecture,random_state=random_state)
            # Paris et al. 2019 uses the Calinski Harabasz score to identify the number of clusters to use
            score = calinski_harabasz_score(X, labels)
            label_list.append(labels)
            CH_score.append(score)
            som_list.append(som)

    while len(label_list)==0:
        nodes = np.clip(nodes-1, 1, None)
        for architecture in nodes:
            if X.shape[0]>=architecture[0]*architecture[1]:
                labels, som = SOM_clustering(X,architecture,random_state=random_state)
                label_list.append(labels)
                CH_score.append(0)
                som_list.append(som)

    return label_list[np.argmax(CH_score)], som_list[np.argmax(CH_score)]

def find_optimal_k_and_cluster(X, k_max=12, method='kmeans', random_state=None):
    label_list = []
    CH_score = []
    for k in range(2,k_max+1):
        if X.shape[0] > k:
            if method == 'kmeans':
                labels = KMeans(n_clusters=k, n_init=10, max_iter=300, random_state=random_state, n_jobs=None).fit_predict(X)
            elif method == 'hierarchical':
                labels = AgglomerativeClustering(n_clusters=k, linkage='single').fit_predict(X)
            score = calinski_harabasz_score(X, labels)
            label_list.append(labels)
            CH_score.append(score)
    return label_list[np.argmax(CH_score)]

def get_dominant_pixels(labels):
    l = pd.Series(labels)
    labels_premapper = l\
            .groupby(labels)\
            .size()\
            .sort_values(ascending=False)\
            .to_frame()
    labels_premapper['labels_choice'] = [True]+[False for i in range(len(labels_premapper)-1)]
    mapper = labels_premapper[['labels_choice']].to_dict()['labels_choice']
    return l.map(mapper)

def Bhattacharyya(a, b):
    a_mean = np.expand_dims(a.mean(axis=0), 1)
    a_cov  = np.cov(a.T)
    b_mean = np.expand_dims(b.mean(axis=0), 1)
    b_cov  = np.cov(b.T)

    sigma = (a_cov + b_cov)/2
    sigma_inv = np.linalg.inv(sigma)

    term_1 = (1/8)*np.dot(np.dot((a_mean-b_mean).T,sigma_inv),(a_mean-b_mean))
    term_2 = (1/2)*np.log(np.linalg.det(sigma)/np.sqrt(np.linalg.det(a_cov)*np.linalg.det(b_cov)))
    return float(np.squeeze(term_1+term_2))

## Own methods

def KMeans_filtering(X, y, filters, n_splits, granularity, keep_rate, random_state=None):
    assert X.shape[0]==y.shape[0], 'X and y must have the same length.'

    ## cluster data
    index = np.arange(len(y))
    clusters_list = []
    index_list  = []
    for analysis_label in np.unique(y):
        print(f'Label: {analysis_label}')
        label_indices = index[y==analysis_label]
        X_label = X[y==analysis_label]

        clusters, kmeans = _KMeans_outlier_detection(X_label, granularity, random_state)
        index_list.append(label_indices)
        clusters_list.append(clusters)

    ## cluster labels
    cluster_col = pd.Series(
        data=np.concatenate(clusters_list),
        index=np.concatenate(index_list),
        name='cluster')\
        .sort_index()

    ## apply filters
    label_encoder = LabelEncoder()
    y_ = label_encoder.fit_transform(y)

    skf = StratifiedKFold(n_splits = n_splits, shuffle=True, random_state=random_state)
    splits = []
    for _, split_indices in skf.split(X, y_):
        splits.append(split_indices)

    filter_outputs = {}
    for n, split in enumerate(splits):
        print(f'Applying filter {n}')
        for name, clf in filters:
            classifier = deepcopy(clf)
            classifier.fit(X[split], y_[split])
            filter_outputs[f'filter_{n}_{name}'] = classifier.predict(X)
            print(f'Applied classifier {name} (part of filter {n})')

    ## mislabel rate
    total_filters = len(filter_outputs.keys())
    mislabel_rate = (total_filters - \
        np.apply_along_axis(
            lambda x: x==y_, 0, pd.DataFrame(filter_outputs).values)\
            .astype(int).sum(axis=1)
            )/total_filters

    ## crunch data
    mislabel_col = pd.Series(data=mislabel_rate, index=index, name='mislabel_rate')
    y_col = pd.Series(data=y, index=index, name='y')
    df = cluster_col.to_frame().join(y_col).join(mislabel_col) # cluster, mislabel_rate, y
    df['count'] = 1
    df_cluster_info_grouped = df.groupby(['y', 'cluster'])\
                .agg({'mislabel_rate':np.mean, 'count':'count'})\
                .sort_values(['mislabel_rate'])
    df_cluster_info_A = df_cluster_info_grouped.groupby(['y']).cumsum().rename(columns={'count':'cumsum'}).drop(columns=['mislabel_rate'])
    df_cluster_info = df_cluster_info_grouped.join(df_cluster_info_A)

    thresholds = df_cluster_info.groupby('y').max()['cumsum']*keep_rate
    actual_thresholds = df_cluster_info[df_cluster_info['cumsum']/thresholds>=1]['cumsum'].groupby('y').min()
    df_cluster_info['status'] = df_cluster_info['cumsum']/actual_thresholds<=1

    print(df_cluster_info.groupby(['y','status']).agg({'mislabel_rate':np.mean, 'count':np.sum}))

    results = df.join(df_cluster_info['status'], on=['y','cluster'])

    return results['cluster'].values, results['status'].values

def _KMeans_outlier_detection(X, granularity=5, random_state=None):
    if granularity>=np.sqrt(X.shape[0]):
        granularity=int(np.sqrt(X.shape[0]))-1
        print(f'Granularity too high for passed dataset, clipping to {granularity}')
    k = int(granularity*np.sqrt(X.shape[0]))
    kmeans = MiniBatchKMeans(k, init_size=5*k, tol=0, max_no_improvement=400, random_state=random_state, verbose=1)
    labels = kmeans.fit_predict(X).astype(str)
    return labels, kmeans
