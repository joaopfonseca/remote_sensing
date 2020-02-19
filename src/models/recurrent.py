
import os

import numpy as np
from keras.utils import np_utils
from keras.layers import (
    Input,
    LSTM,
    Dropout,
    Reshape
)
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

class LSTMNet:
    def __init__(self, input_shape, output_units, filepath='best_model_lstm.hdf5'):
        """input_shape: (height, width, num_bands)"""
        self.height, self.num_bands = input_shape
        self.output_units = output_units

        ## input layer
        self.input_layer = Input(
            (
                self.height,
                self.num_bands
            )
        )

        ########################################################################
        # LSTM layers
        ########################################################################
        dense_layer1 = LSTM(
            units=256,
            activation='tanh',
            return_sequences=True
        )(self.input_layer)
        self.dense_layer1 = Dropout(0.4)(dense_layer1)

        dense_layer2 = LSTM(
            units=128,
            activation='tanh',
            return_sequences=True
        )(self.dense_layer1)
        self.dense_layer2 = Dropout(0.4)(dense_layer2)

        self.output_layer = LSTM(
            units=self.output_units,
            activation='softmax',
            return_sequences=False
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
            self.num_bands
        )
        y_pred = np.argmax(self.model.predict(X), axis=1)
        return y_pred
