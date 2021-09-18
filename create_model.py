# -*- coding: utf-8 -*-
import keras
from keras.layers import *
from keras.models import Model
from utils import *

def RNApar_model(inputX, CNN1D_KER, CNN1D_FIL, LSTM_SIZE, FC_SIZE, DROP_RATE, N_CLASS=2):

    input = Input(shape=(inputX[0].shape[0], inputX[0].shape[1]), name='Input')
    input_mask = Input(shape=([inputX[0].shape[0], 1]), name='Mask', dtype='float32')

    mask_input = []
    mask_input.append(input)
    mask_input.append(input_mask)
    mask_layer = Lambda(mask_func)(mask_input)
    cnn1 = keras.layers.convolutional.Conv1D(filters=CNN1D_FIL, kernel_size=CNN1D_KER, strides=1, padding='same', activation='relu', kernel_initializer='orthogonal')(mask_layer)
    cnn1 = keras.layers.Dropout(DROP_RATE)(cnn1)

    mask_input = []
    mask_input.append(cnn1)
    mask_input.append(input_mask)
    mask_layer = Lambda(mask_func)(mask_input)
    cnn2 = keras.layers.convolutional.Conv1D(filters=CNN1D_FIL, kernel_size=CNN1D_KER, strides=1, padding='same', activation='relu', kernel_initializer='orthogonal')(mask_layer)
    cnn2 = keras.layers.Dropout(DROP_RATE)(cnn2)

    mask_input = []
    mask_input.append(cnn2)
    mask_input.append(input_mask)
    mask_layer = Lambda(mask_func)(mask_input)
    cnn3 = keras.layers.convolutional.Conv1D(filters=CNN1D_FIL, kernel_size=CNN1D_KER, strides=1, padding='same', activation='relu', kernel_initializer='orthogonal')(mask_layer)
    cnn3 = keras.layers.Dropout(DROP_RATE)(cnn3)

    mask_input = []
    mask_input.append(cnn3)
    mask_input.append(input_mask)
    mask_layer = Lambda(mask_func)(mask_input)
    cnn4 = keras.layers.convolutional.Conv1D(filters=CNN1D_FIL, kernel_size=CNN1D_KER, strides=1, padding='same', activation='relu', kernel_initializer='orthogonal')(mask_layer)
    cnn4 = keras.layers.Dropout(DROP_RATE)(cnn4)

    mask_input = []
    mask_input.append(cnn4)
    mask_input.append(input_mask)
    mask_layer = Lambda(mask_func)(mask_input)
    lstm1 = Bidirectional_mask(CuDNNLSTM(LSTM_SIZE, kernel_initializer="orthogonal", recurrent_initializer="orthogonal", return_sequences=True))([mask_layer, input_mask])
    lstm1 = keras.layers.BatchNormalization()(lstm1)
    lstm1 = keras.layers.Dropout(DROP_RATE)(lstm1)

    mask_input = []
    mask_input.append(lstm1)
    mask_input.append(input_mask)
    mask_layer = Lambda(mask_func)(mask_input)
    dnn1 = Dense(FC_SIZE, activation=None, kernel_initializer='orthogonal')(mask_layer)
    dnn1 = BatchNormalization(epsilon=1e-6)(dnn1)
    dnn1 = Activation('relu')(dnn1)
    dnn1 = Dropout(DROP_RATE)(dnn1)

    mask_input = []
    mask_input.append(dnn1)
    mask_input.append(input_mask)
    mask_layer = Lambda(mask_func)(mask_input)
    dnn2 = Dense(FC_SIZE, activation=None, kernel_initializer='orthogonal')(mask_layer)
    dnn2 = BatchNormalization(epsilon=1e-6)(dnn2)
    dnn2 = Activation('relu')(dnn2)
    dnn2 = Dropout(DROP_RATE)(dnn2)

    dnn1_2 = add([dnn1, dnn2])

    mask_input = []
    mask_input.append(dnn1_2)
    mask_input.append(input_mask)
    mask_layer = Lambda(mask_func)(mask_input)
    dnn3 = Dense(FC_SIZE, activation=None, kernel_initializer='orthogonal', )(mask_layer)
    dnn3 = BatchNormalization(epsilon=1e-6)(dnn3)
    dnn3 = Activation('relu')(dnn3)
    dnn3 = Dropout(DROP_RATE)(dnn3)

    mask_input = []
    mask_input.append(dnn3)
    mask_input.append(input_mask)
    mask_layer = Lambda(mask_func)(mask_input)
    dnn4 = Dense(FC_SIZE, activation=None, kernel_initializer='orthogonal', )(mask_layer)
    dnn4 = BatchNormalization(epsilon=1e-6)(dnn4)
    dnn4 = Activation('relu')(dnn4)
    dnn4 = Dropout(DROP_RATE)(dnn4)

    dnn3_4 = add([dnn3, dnn4])


    mask_input = []
    mask_input.append(dnn3_4)
    mask_input.append(input_mask)
    mask_layer = Lambda(mask_func)(mask_input)
    output = Dense(N_CLASS, activation=None, kernel_initializer='orthogonal', )(mask_layer)
    output = Activation('softmax')(output)

    model = Model(inputs=[input, input_mask], outputs=[output])
    return model

