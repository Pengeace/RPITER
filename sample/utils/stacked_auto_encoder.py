import numpy as np
from keras.layers import Dense, Dropout
from keras.models import Sequential


def train_auto_encoder(X_train, X_test, layers, batch_size=100, nb_epoch=100, activation='sigmoid', optimizer='adam'):
    trained_encoders = []
    trained_decoders = []
    X_train_tmp = np.copy(X_train)
    X_test_tmp = np.copy(X_test)
    for n_in, n_out in zip(layers[:-1], layers[1:]):
        print('Pre-training the layer: Input {} -> Output {}'.format(n_in, n_out))
        ae = Sequential(
            [Dense(n_out, input_dim=X_train_tmp.shape[1], activation=activation, ),
             Dense(n_in, activation=activation),
             Dropout(0.2)]
        )
        ae.compile(loss='mean_squared_error', optimizer=optimizer)
        ae.fit(X_train_tmp, X_train_tmp, batch_size=batch_size, epochs=nb_epoch, verbose=0, shuffle=True)
        # store trained encoder
        trained_encoders.append(ae.layers[0])
        trained_decoders.append(ae.layers[1])
        # update training data
        encoder = Sequential([ae.layers[0]])
        # encoder.evaluate(X_train_tmp, X_train_tmp, batch_size=batch_size)
        X_train_tmp = encoder.predict(X_train_tmp)
        X_test_tmp = encoder.predict(X_test_tmp)

    return trained_encoders, trained_decoders, X_train_tmp, X_test_tmp
