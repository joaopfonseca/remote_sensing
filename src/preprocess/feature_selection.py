
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




class RFCGiniImportance(SelectFromModel):
    def __init__(
            n_estimators=100,
            criterion='gini',
            threshold=None,
            norm_order=1,
            max_features=None,
            n_jobs=None,
            random_state=None
        ):
        """
        Meta-transformer for selecting features based on importance weights.

        Parameters
        ----------
        n_estimators : integer, optional (default=100)
            The number of trees in the forest.

        criterion : string, optional (default="gini")
            The function to measure the quality of a split. Supported criteria are
            "gini" for the Gini impurity and "entropy" for the information gain.
            Note: this parameter is tree-specific.

        estimator : object
            The base estimator from which the transformer is built.
            This can be both a fitted (if ``prefit`` is set to True)
            or a non-fitted estimator. The estimator must have either a
            ``feature_importances_`` or ``coef_`` attribute after fitting.

        threshold : string, float, optional default None
            The threshold value to use for feature selection. Features whose
            importance is greater or equal are kept while the others are
            discarded. If "median" (resp. "mean"), then the ``threshold`` value is
            the median (resp. the mean) of the feature importances. A scaling
            factor (e.g., "1.25*mean") may also be used. If None and if the
            estimator has a parameter penalty set to l1, either explicitly
            or implicitly (e.g, Lasso), the threshold used is 1e-5.
            Otherwise, "mean" is used by default.

        prefit : bool, default False
            Whether a prefit model is expected to be passed into the constructor
            directly or not. If True, ``transform`` must be called directly
            and SelectFromModel cannot be used with ``cross_val_score``,
            ``GridSearchCV`` and similar utilities that clone the estimator.
            Otherwise train the model using ``fit`` and then ``transform`` to do
            feature selection.

        norm_order : non-zero int, inf, -inf, default 1
            Order of the norm used to filter the vectors of coefficients below
            ``threshold`` in the case where the ``coef_`` attribute of the
            estimator is of dimension 2.

        max_features : int or None, optional
            The maximum number of features selected scoring above ``threshold``.
            To disable ``threshold`` and only select based on ``max_features``,
            set ``threshold=-np.inf``.
        """

        # SelectFromModel
        self.threshold = threshold
        self.prefit = prefit
        self.norm_order = norm_order
        self.max_features = max_features

        # RFC
        self.n_estimators=n_estimators
        self.criterion=criterion
        self.n_jobs=n_jobs
        self.random_state=random_state

        self.estimator = RandomForestClassifier(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            n_jobs=self.n_jobs,
            random_state=self.random_state
        )

        super().__init__(
            estimator=self.estimator,
            threshold=self.threshold,
            prefit=self.prefit,
            norm_order=self.norm_order,
            max_features=self.max_features
        )

        def fit(X, y):
            pass
