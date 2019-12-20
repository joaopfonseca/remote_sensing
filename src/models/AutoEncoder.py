

class ExperimentalHSNAutoEncoder:
    """
    Convolutional layers on the encoder part were inspired on the
    HybridSpectralNet architecture.
    """
    def __init__(self, window_shape, filepath='best_model.hdf5'):

        self._decoder(self._encoder((25,25,10)))

        self.model = Model(inputs=self.input_layer, outputs=self.decoder_output)
        self.model.compile(loss='mean_squared_error', optimizer=RMSprop())#, metrics=['accuracy'])
        self.model.summary()
        abspath = os.path.abspath('.')
        self.filepath = os.path.abspath(os.path.join(abspath,filepath))
        checkpoint = ModelCheckpoint(self.filepath, monitor='accuracy', verbose=1, save_best_only=True, mode='max')
        self.callbacks_list = [checkpoint]

    def _encoder(self, window_shape):
        """input_shape: (height, width, num_bands)"""
        self.height, self.width, self.num_bands = window_shape

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
        conv_layer1 = Conv3D(filters=8, kernel_size=(3, 3, 2), activation='relu')(self.input_layer) # 23, 23, 9, 8

        conv_layer2 = Conv3D(filters=16, kernel_size=(3, 3, 5), activation='relu')(conv_layer1) # 21, 21, 5, 16

        conv_layer3 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu')(conv_layer2) # 19, 19, 3, 32

        conv3d_shape = conv_layer3._keras_shape

        conv_layer3 = Reshape((conv3d_shape[1],conv3d_shape[2],conv3d_shape[3]*conv3d_shape[4]))(conv_layer3) # 19, 19, 96

        conv2 = Conv2D(
            filters=64,
            kernel_size=(4,4),
            activation='relu'
        )(conv_layer3) # 16 x 16 x 64
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
        conv2 = BatchNormalization()(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #8 x 8 x 64

        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2) #8 x 8 x 128 (small and thick)
        conv3 = BatchNormalization()(conv3)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
        conv3 = BatchNormalization()(conv3)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3) #8 x 8 x 256 (small and thick)
        conv4 = BatchNormalization()(conv4)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
        self.encoder_output = BatchNormalization()(conv4)
        return self.encoder_output

    def _decoder(self, encoder_output):
        """
        """
        conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(encoder_output) #8 x 8 x 128
        conv5 = BatchNormalization()(conv5)
        conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)
        conv5 = BatchNormalization()(conv5)
        conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv5) #8 x 8 x 64
        conv6 = BatchNormalization()(conv6)
        conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv6)
        conv6 = BatchNormalization()(conv6)
        up1 = UpSampling2D((3,3))(conv6) # 24 x 24 x 64
        conv7 = Conv2D(96, (6, 6), activation='relu')(up1) # 19 x 19 x 96
        conv7 = BatchNormalization()(conv7)
        conv7 = Conv2D(96, (6, 6), activation='relu', padding='same')(conv7)
        conv7 = BatchNormalization()(conv7)
        up2 = UpSampling2D((2,2))(conv7)
        up2shp = up2._keras_shape
        conv7 = Reshape((up2shp[1], up2shp[2], 3, int(up2shp[3]/3)))(up2) # 38, 38, 3, 32
        conv8 = Conv3D(16, kernel_size=(18,18,1), activation='relu')(conv7)
        conv8 = BatchNormalization()(conv8)
        conv8 = Conv3D(16, kernel_size=(18,18,1), activation='relu', padding='same')(conv8)
        conv8 = BatchNormalization()(conv8)
        up3 = UpSampling3D((2,2,4))(conv8)
        conv9 = Conv3D(8, kernel_size=(18,18,3), activation='relu')(up3)
        conv9 = BatchNormalization()(conv9)
        conv9 = Conv3D(8, kernel_size=(3,3,3), activation='relu', padding='same')(conv9)
        conv9 = BatchNormalization()(conv9)
        conv10 = Conv3D(1, kernel_size=(3,3,2), activation='relu', padding='same')(conv9)
        self.decoder_output = BatchNormalization()(conv10)
        return self.decoder_output

    def load_weights(self, filepath):
        self.filepath = filepath
        self.model = load_model(filepath)
        self.model.compile(loss='mean_squared_error', optimizer=RMSprop())

    def fit(self, X, y, batch_size=256, epochs=100):
        # transform matrices to correct format
        self.num_bands = X.shape[-1]
        X = X.reshape(
            -1,
            self.height,
            self.width,
            self.num_bands,
            1
        )
        y = y.reshape(
            -1,
            self.height,
            self.width,
            self.num_bands,
            1
        )

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
            self.model.compile(loss='mean_squared_error', optimizer=RMSprop())

        self.num_bands = X.shape[-1]
        X = X.reshape(
            -1,
            self.height,
            self.width,
            self.num_bands,
            1
        )

        y_pred = np.argmax(self.model.predict(X), axis=1)
        return y_pred

class MLPAutoEncoder:
    """
    """
    def __init__(self, num_bands, filepath='best_model.hdf5'):
        self.num_bands = num_bands

        self._decoder(self._encoder(num_bands))
        self.model = Model(inputs=self.input_layer, outputs=self.decoder_output)
        self.model.compile(loss='mean_squared_error', optimizer=RMSprop())#, metrics=['accuracy'])
        self.model.summary()
        abspath = os.path.abspath('.')
        self.filepath = os.path.abspath(os.path.join(abspath,filepath))
        checkpoint = ModelCheckpoint(self.filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        self.callbacks_list = [checkpoint]

    def _encoder(self, num_bands):
        """
        """
        self.input_layer = Input((num_bands,))
        layer1 = Dense(32, input_shape=self.input_layer._keras_shape, activation='relu')(self.input_layer)
        layer1 = BatchNormalization()(layer1)
        layer2 = Dense(16, activation='relu')(layer1)
        layer2 = BatchNormalization()(layer2)
        layer3 = Dense(4, activation='relu')(layer2)
        self.encoder_output = BatchNormalization()(layer3)
        return self.encoder_output

    def _decoder(self, encoder_output):
        """
        """
        layer4 = Dense(16, input_shape=self.encoder_output._keras_shape, activation='relu')(encoder_output)
        layer4 = BatchNormalization()(layer4)
        layer5 = Dense(32, activation='relu')(layer4)
        layer5 = BatchNormalization()(layer5)
        self.decoder_output = Dense(10, activation=None)(layer5)
        return self.decoder_output

    def load_weights(self, filepath):
        self.filepath = filepath
        self.model = load_model(filepath)
        self.model.compile(loss='mean_squared_error', optimizer=RMSprop())

    def fit(self, X, y, batch_size=256, epochs=100):
        # transform matrices to correct format
        self.num_bands = X.shape[-1]
        X = X.reshape(-1, self.num_bands,)
        y = y.reshape(-1, self.num_bands,)

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
            self.model.compile(loss='mean_squared_error', optimizer=RMSprop())
        #else:
        #    self.load_model(self.filepath)
        #self.model.compile(loss='categorical_crossentropy', optimizer=self.adam, metrics=['accuracy'])

        X_pred = self.model.predict(X)
        mse = ((X_pred-X)**2).mean(axis=1)
        return mse



class MLPEncoderClassifier:
    def __init__(self, encoder_list, num_targets, filepath='best_model.hdf5'):
        self.num_targets = num_targets
        self.num_encoders = len(encoder_list)

        MergedEncoders = Concatenate()([model.encoder_output for model in encoder_list])
        self._MLPClassifier(MergedEncoders)


        self.model = Model(inputs=[model.input_layer for model in encoder_list], outputs=self.output_layer)
        self.adam = Adam(lr=0.001, decay=1e-06)
        self.model.compile(loss='categorical_crossentropy', optimizer=self.adam, metrics=['accuracy'])
        self.model.summary()
        abspath = os.path.abspath('.')
        self.filepath = os.path.abspath(os.path.join(abspath,filepath))
        checkpoint = ModelCheckpoint(self.filepath, monitor='accuracy', verbose=1, save_best_only=True, mode='max')
        self.callbacks_list = [checkpoint]

    def _MLPClassifier(self, merged_encoders_outputs):
        layer1 = BatchNormalization()(merged_encoders_outputs)
        layer1 = Dense(32, activation='relu')(layer1)
        layer1 = BatchNormalization()(layer1)
        layer2 = Dense(16, activation='relu')(layer1)
        layer2 = BatchNormalization()(layer2)
        self.output_layer = Dense(self.num_targets, activation='sigmoid')(layer2)
        return self.output_layer

    def fit(self, X, y, batch_size=256, epochs=100):
        # transform matrices to correct format
        self.num_bands = X.shape[-1]
        X = X.reshape(-1, self.num_bands,)
        y = np_utils.to_categorical(y, num_classes=self.num_targets)

        self.history = self.model.fit(
            x=[X for i in range(self.num_encoders)],
            y=y,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=self.callbacks_list
        )

    def predict(self, X, filepath=None):
        # assert: self.filepath or filepath must exist
        if filepath:
            self.load_weights(filepath)
            self.model.compile(loss='mean_squared_error', optimizer=RMSprop())
        #else:
        #    self.load_model(self.filepath)
        #self.model.compile(loss='categorical_crossentropy', optimizer=self.adam, metrics=['accuracy'])

        y_pred = np.argmax(self.model.predict([X for i in range(self.num_encoders)]), axis=1)
        return y_pred
