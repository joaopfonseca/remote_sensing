"""
Near replication of Hybrid Spectral Net paper
TODO:
    - Apply PCA option (?)
    - Allow padding with other type of values other than zero
    - Vectorize loop in createImageCubes
    - Use sklearn extensions' random state generator (?)
    - Check if kernel sizes are correct (I think they use 7 as kernel width,
      but I think it was meant to be depth)
    - BatchNormalization (?) (Not done in the Hybrid Spectral Net paper)
    - Add verbose option (?)
    - Use os library to get absolute path of filepath dir
    - Finish classification report
    - Cube reconstruction function
    - Entire Image prediction function
"""

import pickle # remove after getting final version of code

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from keras.utils import np_utils
from keras.layers import (
    Input,
    Conv2D,
    Conv3D,
    Flatten,
    Dense,
    Dropout,
    Reshape
)
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint


################################################################################
# preprocessing
################################################################################

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

def applyPCA(X, numComponents=10):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0],X.shape[1], numComponents))
    return newX, pca

def split_data(X, y, test_size=0.7, random_state=0, stratify=None, **kwargs):
    """
    Pretty pointless function...
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
        *kwargs
    )
    return X_train, X_test, y_train, y_test


################################################################################
# model
################################################################################

class HybridSpectralNet:
    def __init__(self, input_shape, output_units):
        """input_shape: (height, width, num_bands)"""
        self.height, self.width, self.num_bands = input_shape
        self.output_units = output_units

        ## input layer
        self.input_layer = Input(
            (
                self.height,
                self.width,
                self.num_bands,
                1
            )
        )

        ########################################################################
        # convolutional layers
        ########################################################################
        self.conv_layer1 = Conv3D(
            filters=8,
            kernel_size=(3, 3, 7),
            activation='relu'
        )(self.input_layer)

        self.conv_layer2 = Conv3D(
            filters=16,
            kernel_size=(3, 3, 5),
            activation='relu'
        )(self.conv_layer1)

        conv_layer3 = Conv3D(
            filters=32,
            kernel_size=(3, 3, 3),
            activation='relu'
        )(self.conv_layer2)

        conv3d_shape = conv_layer3._keras_shape

        self.conv_layer3 = Reshape(
            (
                conv3d_shape[1],
                conv3d_shape[2],
                conv3d_shape[3]*conv3d_shape[4]
            )
        )(conv_layer3)

        self.conv_layer4 = Conv2D(
            filters=64,
            kernel_size=(3,3),
            activation='relu'
        )(self.conv_layer3)

        self.flatten_layer = Flatten()(self.conv_layer4)

        ########################################################################
        # fully connected layers
        ########################################################################
        dense_layer1 = Dense(
            units=256,
            activation='relu'
        )(self.flatten_layer)
        self.dense_layer1 = Dropout(0.4)(dense_layer1)

        dense_layer2 = Dense(
            units=128,
            activation='relu'
        )(self.dense_layer1)
        self.dense_layer2 = Dropout(0.4)(dense_layer2)

        self.output_layer = Dense(
            units=self.output_units,
            activation='softmax'
        )(self.dense_layer2)

        self.model = Model(inputs=self.input_layer, outputs=self.output_layer)
        self.adam = Adam(lr=0.001, decay=1e-06)
        self.model.compile(loss='categorical_crossentropy', optimizer=self.adam, metrics=['accuracy'])
        self.model.summary()

    def fit(self, X, y, batch_size=256, epochs=100, filepath='best_model.hdf5'):
        # transform matrices to correct format
        self.filepath = filepath
        self.num_bands = X.shape[-1]
        self.X = X.reshape(
            -1,
            self.height,
            self.width,
            self.num_bands,
            1
        )
        self.y = np_utils.to_categorical(y)

        checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')
        self.callbacks_list = [checkpoint]
        self.history = self.model.fit(
            x=self.X,
            y=self.y,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=self.callbacks_list
        )

    def predict(self, X):
        self.model.load_weights(self.filepath)
        self.model.compile(loss='categorical_crossentropy', optimizer=self.adam, metrics=['accuracy'])
        X_new = X.reshape(
            -1,
            self.height,
            self.width,
            self.num_bands,
            1
        )
        y_pred = np.argmax(self.model.predict(X_new), axis=1)
        return y_pred

    def classification_report(self, X, y, target_names=None):
        y_pred = self.predict(X).astype(int)
        return classification_report(y.astype(int), y_pred, target_names=target_names)

    def reports(self, X, y): # TODO
        pass

    def dump(self, fname):
        pickle.dump(self, open(fname, 'wb'))


class HybridSpectralNetMine:
    def __init__(self, input_shape, output_units):
        """input_shape: (height, width, num_bands)"""
        self.height, self.width, self.num_bands = input_shape
        self.output_units = output_units

        ## input layer
        self.input_layer = Input(
            (
                self.height,
                self.width,
                self.num_bands,
                1
            )
        )

        ########################################################################
        # convolutional layers
        ########################################################################
        self.conv_layer1 = Conv3D(
            filters=8,
            kernel_size=(3, 3, 2),
            activation='relu'
        )(self.input_layer)

        self.conv_layer2 = Conv3D(
            filters=16,
            kernel_size=(3, 3, 5),
            activation='relu'
        )(self.conv_layer1)

        conv_layer3 = Conv3D(
            filters=32,
            kernel_size=(3, 3, 3),
            activation='relu'
        )(self.conv_layer2)

        conv3d_shape = conv_layer3._keras_shape

        self.conv_layer3 = Reshape(
            (
                conv3d_shape[1],
                conv3d_shape[2],
                conv3d_shape[3]*conv3d_shape[4]
            )
        )(conv_layer3)

        self.conv_layer4 = Conv2D(
            filters=64,
            kernel_size=(3,3),
            activation='relu'
        )(self.conv_layer3)

        self.flatten_layer = Flatten()(self.conv_layer4)

        ########################################################################
        # fully connected layers
        ########################################################################
        dense_layer1 = Dense(
            units=256,
            activation='relu'
        )(self.flatten_layer)
        self.dense_layer1 = Dropout(0.4)(dense_layer1)

        dense_layer2 = Dense(
            units=128,
            activation='relu'
        )(self.dense_layer1)
        self.dense_layer2 = Dropout(0.4)(dense_layer2)

        self.output_layer = Dense(
            units=self.output_units,
            activation='softmax'
        )(self.dense_layer2)

        self.model = Model(inputs=self.input_layer, outputs=self.output_layer)
        self.adam = Adam(lr=0.001, decay=1e-06)
        self.model.compile(loss='categorical_crossentropy', optimizer=self.adam, metrics=['accuracy'])
        self.model.summary()

    def fit(self, X, y, batch_size=256, epochs=100, filepath='best_model.hdf5'):
        # transform matrices to correct format
        self.filepath = filepath
        self.num_bands = X.shape[-1]
        self.X = X.reshape(
            -1,
            self.height,
            self.width,
            self.num_bands,
            1
        )
        self.y = np_utils.to_categorical(y)

        checkpoint = ModelCheckpoint(filepath, monitor='accuracy', verbose=1, save_best_only=True, mode='max')
        self.callbacks_list = [checkpoint]
        self.history = self.model.fit(
            x=self.X,
            y=self.y,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=self.callbacks_list
        )

    def predict(self, X):
        self.model.load_weights(self.filepath)
        self.model.compile(loss='categorical_crossentropy', optimizer=self.adam, metrics=['accuracy'])
        X_new = X.reshape(
            -1,
            self.height,
            self.width,
            self.num_bands,
            1
        )
        y_pred = np.argmax(self.model.predict(X_new), axis=1)
        return y_pred

    def classification_report(self, X, y, target_names=None):
        y_pred = self.predict(X).astype(int)
        return classification_report(y.astype(int), y_pred, target_names=target_names)

    def reports(self, X, y): # TODO
        pass

    def dump(self, fname):
        pickle.dump(self, open(fname, 'wb'))
