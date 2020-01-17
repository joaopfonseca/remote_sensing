
from imblearn.under_sampling.base import BaseCleaningSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted
from sklearn.feature_selection import SelectFromModel

#class RFCGiniImportance(SelectFromModel):
#    def __init__(estimator, threshold=None, prefit=False, norm_order=1, max_features=None):
#        """See sklearn.feature_selection.SelectFromModel documentation"""
#        super().__init__(
#            estimator=estimator,
#            threshold=threshold,
#            prefit=prefit,
#            norm_order=norm_order,
#            max_features=max_features
#        )
