
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
from keras.applications import resnet

class ResNet50:
    def __init__(self, input_shape, output_units, filepath='best_model.hdf5'):
        """input_shape: (height, width, num_bands)"""
        self.height, self.width, self.num_bands = input_shape
        self.output_units = output_units

        ## input layer
        self.input_layer = Input(
            (
                self.height,
                self.width,
                self.num_bands
            )
        )

        self.base_model = resnet.ResNet50(weights=None, include_top=False, input_shape= input_shape)(self.input_layer)
        self.flatten_layer = Flatten()(self.base_model)

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
        abspath = os.path.abspath('.')
        self.filepath = os.path.abspath(os.path.join(abspath,filepath))
        checkpoint = ModelCheckpoint(self.filepath, monitor='accuracy', verbose=1, save_best_only=True, mode='max')
        self.callbacks_list = [checkpoint]

    def load_weights(self, filepath):
        self.filepath = filepath
        self.model = load_model(filepath)
        self.model.compile(loss='categorical_crossentropy', optimizer=self.adam, metrics=['accuracy'])

    def fit(self, X, y, batch_size=256, epochs=100):
        # transform matrices to correct format
        self.num_bands = X.shape[-1]
        self.X = X.reshape(
            -1,
            self.height,
            self.width,
            self.num_bands
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
            self.num_bands
        )
        y_pred = np.argmax(self.model.predict(X), axis=1)
        return y_pred



class PixelBasedResNet50:
    def __init__(self, input_shape, output_units, filepath='best_model.hdf5'):
        """input_shape: (height, width)"""
        self.input_shape = input_shape
        self.height_pad = 32-input_shape[0]
        self.width_pad = 32-input_shape[1]
        self.height, self.width = (32, 32)
        self.output_units = output_units

        ## input layer
        self.input_layer = Input(
            (
                self.height,
                self.width,
                1
            )
        )

        self.base_model = resnet.ResNet50(
            weights=None,
            include_top=False,
            input_shape=(32, 32, 1)
        )(self.input_layer)
        self.flatten_layer = Flatten()(self.base_model)

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
        abspath = os.path.abspath('.')
        self.filepath = os.path.abspath(os.path.join(abspath,filepath))
        checkpoint = ModelCheckpoint(self.filepath, monitor='accuracy', verbose=1, save_best_only=True, mode='max')
        self.callbacks_list = [checkpoint]

    def load_weights(self, filepath):
        self.filepath = filepath
        self.model = load_model(filepath)
        self.model.compile(loss='categorical_crossentropy', optimizer=self.adam, metrics=['accuracy'])

    def fit(self, X, y, batch_size=256, epochs=100):
        # transform matrices to correct format
        self.num_bands = X.shape[-1]
        X = X.reshape(
            -1,
            self.input_shape[0],
            self.input_shape[1]
        )
        X = np.expand_dims(X, -1)
        X = np.pad(X, ((0,0), (self.height_pad,0), (self.width_pad,0), (0,0)))
        y = np_utils.to_categorical(y, num_classes=self.output_units)

        self.history = self.model.fit(
            x=X,
            y=y,
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
            self.input_shape[0],
            self.input_shape[1]
        )
        X = np.expand_dims(X, -1)
        X = np.pad(X, ((0,0), (self.height_pad,0), (self.width_pad,0), (0,0)))
        y_pred = np.argmax(self.model.predict(X), axis=1)
        return y_pred
