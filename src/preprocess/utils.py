"""
TODO:
    - Find createImageCubes alternative
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.feature_selection.base import SelectorMixin

class SelectFeaturesFromList(BaseEstimator, MetaEstimatorMixin, SelectorMixin):
    def __init__(self, feature_rankings=None, n_features=None):
        """
        Used to determine the optimal number of features to keep
        feature_rankings: List or array with an importance rank for each feature
        """
        self.feature_rankings = feature_rankings
        self.n_features = n_features

    def fit(self, X, y):
        return self

    def transform(self, X, y=None):
        return X[:, self.feature_rankings<self.n_features]

    def fit_transform(self, X, y):
        return self.transform(X, y)

    def _get_support_mask(self):
        return None


def ZScoreNormalization(X, axes=(0,1), scorer=None):
    """Applies Z-Score Normalization over a multispectral image"""
    if not scorer:
        u = X.mean(axis=axes)
        std = X.std(axis=axes)
        scorer = lambda arr: (arr - u) / std
    X_norm = scorer(X)
    return X_norm, scorer


def get_2Dcoordinates_matrix(shape, window_size=None):
    """
    pass window size value only if X is padded. If so, excludes coordinates in the padded region
    """
    rows, cols = np.arange(shape[0]), np.arange(shape[1])
    rows_arr = np.repeat(np.expand_dims(rows,-1), cols.shape[0], axis=1)
    cols_arr = np.repeat(np.expand_dims(cols, 0), rows.shape[0], axis=0)
    coords   = np.concatenate(
            [np.expand_dims(rows_arr, 0),np.expand_dims(cols_arr, 0)]
            , axis=0)
    if window_size:
        margin = int((window_size - 1) / 2)
        return coords[:,margin:shape[0]-margin, margin:shape[1]-margin]
    else:
        return coords


def get_patches(coords, X, window_size):
    """
    Returns patches with center pixel `coord` and with boudaries
    [coord[0]-`margin`:coord[0]+`margin`+1, coord[1]-`margin`:coord[1]+`margin`+1]
    """
    def _patches(coord, X, margin):
        r, c = coord[0], coord[1]
        patch = X[r - margin:r + margin + 1, c - margin:c + margin + 1]
        return patch
    return np.apply_along_axis(lambda c: _patches(c, X, int(window_size/2)),1,coords)


def pad_X(X, window_size):
    margin = int((window_size - 1) / 2)
    return np.pad(X, ((margin, margin), (margin, margin), (0,0)))


def createImageCubes(X, y, window_size=5, removeNoLabels=True, NoLabelVal=-1):
    """
    Adapted from Hybrid Spectral Net paper.
    - Assumes padding with Zeros
    -
    """
    margin = int((window_size - 1) / 2)
    padded_X = np.pad(X, ((margin, margin), (margin, margin), (0,0)))

    coords = get_2Dcoordinates_matrix(padded_X.shape)\
        [margin:padded_X.shape[0]-margin, margin:padded_X.shape[1]-margin]\
        .reshape((2,rows.shape[0]*cols.shape[0])).T

    patchesData, patchesLabels = np.apply_along_axis(get_patches, 1 ,coords).T
    patchesData = np.stack(patchesData)

    if removeNoLabels:
        patchesData = patchesData[patchesLabels!=NoLabelVal,:,:,:]
        patchesLabels = patchesLabels[patchesLabels!=NoLabelVal]
    return patchesData, patchesLabels


def applyPCA(X, numComponents=10, model=None):
    shp = X.shape
    newX = np.reshape(X, (-1, shp[2]))
    del X
    if model:
        pca = model
        newX = pca.transform(newX)
    else:
        pca = PCA(n_components=numComponents, copy=False, whiten=True)
        newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (shp[0],shp[1], numComponents))
    return newX, pca
