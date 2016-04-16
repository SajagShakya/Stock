from __future__ import absolute_import
from __future__ import print_function
from keras.datasets import mnist
from keras.models import Sequential, model_from_config
from keras.layers.core import AutoEncoder,Dropout, Dense, Activation, TimeDistributedDense, Flatten
from keras.optimizers import RMSprop
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers.core import Layer
from keras.layers import containers
from keras.utils import np_utils
import numpy as np

nb_classes = 10
batch_size = 100
nb_epoch = 5
activation = 'sigmoid'

input_dim = 784
hidden_dim = 392

max_train_samples = 5000
max_test_samples = 1000


def load_data():
    # the data, shuffled and split between tran and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(60000, input_dim)[:max_train_samples]
    X_test = X_test.reshape(10000, input_dim)[:max_test_samples]
    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")
    X_train /= 255
    X_test /= 255

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)[:max_train_samples]
    Y_test = np_utils.to_categorical(y_test, nb_classes)[:max_test_samples]

    print("X_train: ", X_train.shape)
    print("X_test: ", X_test.shape)

    return X_train,Y_train,X_test,Y_test

def MLP():
    ##########################
    # dense model test       #
    ##########################
    X_train,Y_train,X_test,Y_test=load_data()
    print("Training classical fully connected layer for classification")
    model = Sequential()
    model.add(Dense(200,input_shape=(input_dim,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    #model_classical.get_config(verbose=1)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=False, verbose=0, validation_data=(X_test, Y_test))
    classical_score = model.evaluate(X_test, Y_test, verbose=0, show_accuracy=True)
    print('\nTest Score: %f, Test accuracy:%f'%( classical_score[0],classical_score[1]))

##########################
# autoencoder model test #
##########################


def build_lstm_autoencoder(autoencoder, X_train, X_test):
    X_train = X_train[:, np.newaxis, :]
    X_test = X_test[:, np.newaxis, :]
    print("Modified X_train: ", X_train.shape)
    print("Modified X_test: ", X_test.shape)

    # The TimeDistributedDense isn't really necessary, however you need a lot of GPU memory to do 784x394-394x784
    autoencoder.add(TimeDistributedDense(input_dim, 16))
    autoencoder.add(AutoEncoder(encoder=LSTM(16, 8, activation=activation, return_sequences=True),
                                decoder=LSTM(8, input_dim, activation=activation, return_sequences=True),
                                output_reconstruction=False))
    return autoencoder, X_train, X_test


def build_deep_classical_autoencoder():
    encoder = containers.Sequential([Dense(output_dim=hidden_dim, input_dim=input_dim,activation=activation),
                                     Dense(output_dim=hidden_dim/2,input_dim=hidden_dim,activation=activation)])
    decoder = containers.Sequential([Dense(output_dim=hidden_dim,input_dim=hidden_dim/2,activation=activation),
                                     Dense(output_dim=input_dim,input_dim=hidden_dim,activation=activation)])

    autoencoder=AutoEncoder(encoder=encoder,decoder=decoder,output_reconstruction=True)
    return autoencoder

# Try different things here: 'lstm' or 'classical' or 'denoising'
# or 'deep_denoising'

def train(autoencoder_type="classical"): #'classical', 'lstm'
    X_train,Y_train,X_test,Y_test=load_data()
    print(autoencoder_type)
    print('-'*40)
    # Build our autoencoder model
    model= Sequential()

    if autoencoder_type == 'lstm':
        print("Training LSTM AutoEncoder")
        autoencoder, X_train, X_test = build_lstm_autoencoder(X_train, X_test)
    elif autoencoder_type == 'classical':
        print("Training Classical AutoEncoder")
        autoencoder= build_deep_classical_autoencoder()
        model.add(autoencoder)
        model.compile(loss='mean_squared_error', optimizer='sgd')
        model.fit(X=X_train, y=X_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=False, verbose=1)


    print('validing....')
    # Do an inference pass

    autoencoder.output_reconstruction = False
    model.compile(loss='mean_squared_error', optimizer='sgd')
    prefilter_train = model.predict(X_train)
    prefilter_test = model.predict(X_test)
    print("prefilter_train: ", prefilter_train.shape)
    print("prefilter_test: ", prefilter_test.shape)

    # Classify results from Autoencoder
    print("Building classical fully connected layer for classification")
    model = Sequential()
    if autoencoder_type == 'lstm':
        model.add(TimeDistributedDense(8, nb_classes, activation=activation))
        model.add(Flatten())
    elif autoencoder_type == 'classical':
        model.add(Dense(input_dim=prefilter_train.shape[1], output_dim=nb_classes, activation=activation))
    else:
        model.add(Dense(prefilter_train.shape[1], nb_classes, activation=activation))

    model.add(Activation('softmax'))

    #model.get_config(verbose=1)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(prefilter_train, Y_train, validation_data=(prefilter_test, Y_test), batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=False, verbose=0)

    score = model.evaluate(prefilter_test, Y_test, verbose=0, show_accuracy=True)
    print('\nscore:', score)



if __name__=="__main__":
    train()
