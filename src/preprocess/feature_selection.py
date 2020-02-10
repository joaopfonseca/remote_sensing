import numpy as np
import pandas as pd
#from imblearn.under_sampling.base import BaseCleaningSampler
from sklearn.ensemble import RandomForestClassifier
#from sklearn.utils.validation import check_is_fitted
from sklearn.feature_selection import SelectFromModel
from scipy.stats import spearmanr, pearsonr
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.model_selection import train_test_split

from rfpimp import importances as permutation_importances


class PermutationRF(BaseEstimator, MetaEstimatorMixin):
    def __init__(self, n_estimators=100, test_size=.2, max_features=None, n_jobs=-1, random_state=None):
        super().__init__()
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.test_size = test_size
        self.max_features = max_features
        self.model = RandomForestClassifier(n_estimators=self.n_estimators, n_jobs=self.n_jobs,
            random_state=self.random_state)

    def fit(self, X, y):

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        self.model.fit(X_tr,y_tr)
        self.permutation_importances_ = permutation_importances(
            self.model,
            pd.DataFrame(X_te, columns=np.arange(X_te.shape[-1]).astype(str)),
            pd.Series(y_te)
        )
        return self

    def transform(self, X, y):
        if self.max_features:
            col_ids = self.permutation_importances_\
                .iloc[:self.max_features]\
                .index.values\
                .astype(int)
        else:
            col_ids = self.permutation_importances_\
                .index.values\
                [self.permutation_importances_['Importance']>0]\
                .astype(int)

        return X[:,col_ids], y

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X,y)



class CorrelationBasedFeatureSelection(BaseEstimator, MetaEstimatorMixin):
    def __init__(self, corr_type='pearson', threshold=.6):
        super().__init__()
        self.corr_type = corr_type
        self.threshold = threshold

    def fit(self, X, y=None):
        corr = pd.DataFrame(X).corr(self.corr_type).abs()
        corr_tril = pd.DataFrame(np.tril(corr, -1))
        unstacked = corr_tril.unstack().reset_index()
        self.dropped_features_ = unstacked['level_1'][unstacked[0]>=self.threshold].drop_duplicates().values
        return self

    def transform(self, X, y=None):
        if y is None:
            return np.delete(X, self.dropped_features_, axis=1), y
        else:
            return np.delete(X, self.dropped_features_, axis=1)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X,y)
