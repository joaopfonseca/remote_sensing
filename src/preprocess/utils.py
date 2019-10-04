"""
TODO:
    - Find createImageCubes alternative
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

def ZScoreNormalization(X, axes=(0,1), scorer=None):
    if not scorer:
        u = X.mean(axis=axes)
        std = X.std(axis=axes)
        scorer = lambda arr: (arr - u / std)
    X_norm = scorer(X)
    return X_norm, scorer

def createImageCubes(X, y, window_size=5, removeNoLabels=True, NoLabelVal=-1):
    """
    Adapted from Hybrid Spectral Net paper.
    - Assumes padding with Zeros (change later)
    -
    """
    margin = int((window_size - 1) / 2)
    padded_X = np.pad(X, ((margin, margin), (margin, margin), (0,0)))

    rows = np.arange(margin, padded_X.shape[0] - margin)
    cols = np.arange(margin, padded_X.shape[1] - margin)
    rows_arr = np.repeat(np.expand_dims(rows,-1), cols.shape[0], axis=1)
    cols_arr = np.repeat(np.expand_dims(cols, 0), rows.shape[0], axis=0)
    coords   = np.concatenate(
                [np.expand_dims(rows_arr, 0),np.expand_dims(cols_arr, 0)]
                , axis=0)\
                .reshape((2,rows.shape[0]*cols.shape[0])).T
    def get_patches(val):
        r, c = val[0], val[1]
        patch = padded_X[r - margin:r + margin + 1, c - margin:c + margin + 1]
        return patch, y[r-margin, c-margin]

    patchesData, patchesLabels = np.apply_along_axis(get_patches, 1 ,coords).T
    patchesData = np.stack(patchesData)

    if removeNoLabels:
        patchesData = patchesData[patchesLabels!=NoLabelVal,:,:,:]
        patchesLabels = patchesLabels[patchesLabels!=NoLabelVal]
    return patchesData, patchesLabels

def applyPCA(X, numComponents=10, model=None):
    newX = np.reshape(X, (-1, X.shape[2]))
    if model:
        pca = model
        newX = pca.transform(newX)
    else
        pca = PCA(n_components=numComponents, whiten=True)
        newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0],X.shape[1], numComponents))
    return newX, pca

def split_data(X, y, test_size=0.7, random_state=0, stratify=None, **kwargs):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
        *kwargs
    )
    return X_train, X_test, y_train, y_test
