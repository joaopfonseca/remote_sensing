
class DenoiserAE:
    def __init__(self, input_shape, kernel_size=3, latent_dim=16, filepath='best_model.hdf5'):
        """input_shape: (height, width, num_bands)"""
        self.height, self.width = input_shape
        self.output_units = output_units
        self.kernel_size = kernel_size
        self.latent_dim = latent_dim

        # Encoder/Decoder number of CNN layers and filters per layer
        layer_filters = [32, 64]

        ## input layer
        self.input_layer = Input(
            (
                self.height,
                self.width
            )
        )

        ## encoder
        encoder = self._encoder(self.input_layer)
        ## latent space
        self.latent_input = Input(shape=(self.latent_dim,), name='decoder_input')
        ## decoder
        decoder = self._decoder(self.latent_input)

        # Autoencoder = Encoder + Decoder
        # Instantiate Autoencoder Model
        self.model = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
        self.model.summary()
        self.model.compile(loss='mse', optimizer='adam')

    def _encoder(self, input_layer):
        x = input_layer
        for filters in layer_filters:
            x = Conv2D(filters=filters,
                       kernel_size=kernel_size,
                       strides=2,
                       activation='relu',
                       padding='same')(x)
            x = BatchNormalization(x)

        # Shape info needed to build Decoder Model
        self._shape = K.int_shape(x)

        # Generate the latent vector
        x = Flatten()(x)
        latent = Dense(self.latent_dim, name='latent_vector')(x)

        # Instantiate Encoder Model
        encoder = Model(inputs, latent, name='encoder')
        encoder.summary()
        return encoder

    def _decoder(self, latent_input):

        x = Dense(shape[1] * shape[2] * shape[3])(latent_input)
        x = Reshape((shape[1], shape[2], shape[3]))(x)

        # Stack of Transposed Conv2D blocks
        # Notes:
        # 1) Use Batch Normalization before ReLU on deep networks
        # 2) Use UpSampling2D as alternative to strides>1
        # - faster but not as good as strides>1
        for filters in layer_filters[::-1]:
            x = Conv2DTranspose(filters=filters,
                                kernel_size=kernel_size,
                                strides=2,
                                activation='relu',
                                padding='same')(x)

        x = Conv2DTranspose(filters=1,
                            kernel_size=kernel_size,
                            padding='same')(x)

        outputs = Activation('sigmoid', name='decoder_output')(x)

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
        X = X.reshape(
            -1,
            self.height,
            self.width
        )

        y = y.reshape(
            -1,
            self.height,
            self.width,
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
