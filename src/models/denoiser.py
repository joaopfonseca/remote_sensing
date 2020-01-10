import numpy as np
from keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    Flatten,
    Dense,
    Reshape,
    Conv2DTranspose,
    Activation,
    ZeroPadding2D
)
from keras.models import Model, load_model
from keras.backend import int_shape
from keras.callbacks import ModelCheckpoint

class DenoiserAE:
    def __init__(self, input_shape, kernel_size=3, latent_num_distributions=10, filepath='best_model.hdf5'):
        """
        input_shape: (height, width, num_classes)
        latent_num_distributions: number of distributions per class in the latent space
        """
        self.height, self.width, self.num_classes = input_shape
        self.kernel_size = kernel_size
        # maybe doing it this way doesn't make sense, I'll definitely need to review this part
        self.latent_num_distributions = latent_num_distributions


        # Encoder/Decoder number of CNN layers and filters per layer
        self.layer_filters = [32, 64]

        ## input layer
        self.input_layer = Input(
            (
                self.height,
                self.width,
                self.num_classes
            )
        )

        ## encoder
        encoder = self._encoder(self.input_layer)
        ## latent space
        self.latent_input = Input(shape=(self.latent_num_distributions*self.num_classes,), name='decoder_input')
        ## decoder
        decoder = self._decoder(self.latent_input)

        # Autoencoder = Encoder + Decoder
        # Instantiate Autoencoder Model
        self.model = Model(self.input_layer, decoder(encoder(self.input_layer)), name='autoencoder')
        self.model.summary()
        self.model.compile(loss='mse', optimizer='adam')
        abspath = os.path.abspath('.')
        self.filepath = os.path.abspath(os.path.join(abspath,filepath))
        checkpoint = ModelCheckpoint(self.filepath, monitor='accuracy', verbose=1, save_best_only=True, mode='max')
        self.callbacks_list = [checkpoint]

    def _encoder(self, input_layer):
        x = input_layer
        for filters in self.layer_filters:
            x = Conv2D(filters=filters,
                       kernel_size=self.kernel_size,
                       strides=2,
                       activation='relu',
                       padding='same')(x)
            x = BatchNormalization()(x)

        # Shape info needed to build Decoder Model
        self._shape = int_shape(x)

        # Generate the latent vector
        x = Flatten()(x)
        latent = Dense(self.latent_num_distributions*self.num_classes, name='latent_vector')(x)
        # Instantiate Encoder Model
        encoder = Model(input_layer, latent, name='encoder')
        encoder.summary()
        return encoder

    def _decoder(self, latent_inputs):

        x = Dense(self._shape[1] * self._shape[2] * self._shape[3])(latent_inputs)
        x = Reshape((self._shape[1], self._shape[2], self._shape[3]))(x)

        # Stack of Transposed Conv2D blocks
        # Notes:
        # 1) Use Batch Normalization before ReLU on deep networks
        # 2) Use UpSampling2D as alternative to strides>1
        # - faster but not as good as strides>1
        for filters in self.layer_filters[::-1]:
            x = Conv2DTranspose(filters=filters,
                                kernel_size=self.kernel_size,
                                strides=2,
                                activation='relu',
                                padding='same')(x)

        x = Conv2DTranspose(filters=self.num_classes,
                            kernel_size=self.kernel_size,
                            padding='same')(x)

        outputs = Activation('sigmoid', name='decoder_output')(x)
        _, self._out_height, self._out_width, _ = int_shape(outputs)

        # Instantiate Decoder Model
        decoder = Model(latent_inputs, outputs, name='decoder')
        decoder.summary()
        return decoder

    def load_weights(self, filepath):
        self.filepath = filepath
        self.model = load_model(filepath)
        self.model.compile(loss='mse', optimizer='adam')

    def fit(self, X, y, batch_size=256, epochs=100):
        # transform matrices to correct format
        X = X.reshape(-1, self.height, self.width)
        y = y.reshape(-1, self.height, self.width)

        X = np.array([np.moveaxis(np.array([arr==i for i in range(self.num_classes)]),0,-1) for arr in X])
        y = np.array([np.moveaxis(np.array([arr==i for i in range(self.num_classes)]),0,-1) for arr in y])
        y = np.pad(
            y,
            ((0,0), (self._out_height-self.height, 0), (self._out_width-self.width, 0), (0,0)),
        )

        self.history = self.model.fit(
            x=X,
            y=y,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=self.callbacks_list
        )

    def predict(self, X, filepath=None):
        if filepath:
            self.load_weights(filepath)

        X = X.reshape(
            -1,
            self.height,
            self.width
        )
        y_pred = np.argmax(self.model.predict(X), axis=1)
        return y_pred
