"""
Near replication of Hybrid Spectral Net paper
TODO:
    - Finish classification report
"""

import pickle # remove after getting final version of code
import os

import numpy as np
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
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint


class HybridSpectralNet:
    """
    NOTE: 1st kernel size was changed from the original architecture
    """
    def __init__(self, input_shape, output_units, filepath='best_model.hdf5'):
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
        checkpoint = ModelCheckpoint(filepath, monitor='accuracy', verbose=1, save_best_only=True, mode='max')
        self.callbacks_list = [checkpoint]

    def load_weights(self, filepath):
        self.filepath = filepath
        self.model = load_model(filepath)
        self.model.compile(loss='categorical_crossentropy', optimizer=self.adam, metrics=['accuracy'])

    def fit(self, X, y, batch_size=256, epochs=100):
        # transform matrices to correct format
        abspath = os.path.abspath('.')
        self.filepath = os.path.abspath(os.path.join(abspath,filepath))
        self.num_bands = X.shape[-1]
        self.X = X.reshape(
            -1,
            self.height,
            self.width,
            self.num_bands,
            1
        )
        self.y = np_utils.to_categorical(y, num_classes=self.output_units)

        self.history = self.model.fit(
            x=self.X,
            y=self.y,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=self.callbacks_list
        )

    def predict(self, X, filepath=None):
        # assert: self.filepath or filepath must exist
        if filepath:
            self.load_weights(filepath)
            self.model.compile(loss='categorical_crossentropy', optimizer=self.adam, metrics=['accuracy'])
        #else:
        #    self.load_model(self.filepath)
        #self.model.compile(loss='categorical_crossentropy', optimizer=self.adam, metrics=['accuracy'])

        X = X.reshape(
            -1,
            self.height,
            self.width,
            self.num_bands,
            1
        )
        y_pred = np.argmax(self.model.predict(X), axis=1)
        return y_pred


class HybridSpectralNetCustom:
    """
    NOTE: 1st kernel size was changed from the original architecture
    """
    def __init__(self, window_shape, output_units):
        """input_shape: (height, width, num_bands)"""
        self.height, self.width, self.num_bands = window_shape
        self.output_units = output_units

        coords = Input((2,), dtype='int32')
        x_coords = coords[:,0]
        y_coords = coords[:,1]
        self.img = K.variable(np.zeros((10,10)))

        vertical_var = int(self.height/2)
        horizontal_var = int(self.width/2)

        ## input layer
        #self.input_layer = self.img[x_coords-vertical_var:x_coords+vertical_var,
        #    y_coords-horizontal_var:y_coords+horizontal_var]
        self.input_layer = Lambda(lambda x: x[x_coords-vertical_var, y_coords-horizontal_var])(self.img)
        Crop()


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

        self.model = Model(inputs=[self.input_coords, self.img], outputs=self.output_layer)
        self.adam = Adam(lr=0.001, decay=1e-06)
        self.model.compile(loss='categorical_crossentropy', optimizer=self.adam, metrics=['accuracy'])
        self.model.summary()

    def load_weights(self, filepath):
        self.filepath = filepath
        self.model = load_model(filepath)
        self.model.compile(loss='categorical_crossentropy', optimizer=self.adam, metrics=['accuracy'])

    def fit(self, X, y, lookup_img, batch_size=256, epochs=100, filepath='best_model.hdf5'):
        """
        X: pixel coordinates
        y: pixel labels
        lookup_img: image from which image cubes will be retrieved
        """
        abspath = os.path.abspath('.')
        self.filepath = os.path.abspath(os.path.join(abspath,filepath))
        self.X = X.reshape(
            -1,
            2
        )
        self.y = np_utils.to_categorical(y)

        checkpoint = ModelCheckpoint(self.filepath, monitor='accuracy', verbose=1, save_best_only=True, mode='max')
        self.callbacks_list = [checkpoint]
        self.history = self.model.fit(
            x=[self.X, lookup_img],
            y=self.y,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=self.callbacks_list,
            workers=0,
            use_multiprocessing=True
        )

    def predict(self, X, filepath=None):
        # assert: self.filepath or filepath must exist
        if filepath:
            self.load_weights(filepath)
            self.model.compile(loss='categorical_crossentropy', optimizer=self.adam, metrics=['accuracy'])
        #else:
        #    self.load_model(self.filepath)
        #self.model.compile(loss='categorical_crossentropy', optimizer=self.adam, metrics=['accuracy'])

        X = X.reshape(
            -1,
            self.height,
            self.width,
            self.num_bands,
            1
        )
        y_pred = np.argmax(self.model.predict(X), axis=1)
        return y_pred
