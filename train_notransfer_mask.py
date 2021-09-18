# -*- coding: utf-8 -*-
from keras.callbacks import EarlyStopping, ModelCheckpoint
import argparse
from bayes_opt import BayesianOptimization

from StepLR import *
from create_model import *

def main(CNN1D_KER, CNN1D_FIL, LSTM_SIZE, FC_SIZE):

    EPOCHS = 300
    BATCH_SIZE = 500
    LR = 0.005

    epoch_list = [10, 20, 200, 600, 10000]
    lr_list = [0.005, 0.001, 0.0005, 0.0001, 0.00005]

    CNN1D_KER = int(CNN1D_KER)
    CNN1D_FIL = int(CNN1D_FIL)
    LSTM_SIZE = int(LSTM_SIZE)
    FC_SIZE = int(FC_SIZE)
    DROP_RATE = 0.1

    OUTPUT_FILE_PREFIX = "./results/"
    INPUT_FILE_PREFIX_PRE = "./pre-train/"
    INPUT_FILE_PREFIX_TRANS = "./trans-train/"

    mkdir(OUTPUT_FILE_PREFIX)

    parser = argparse.ArgumentParser(description='RNA single chain prediction tool.')
    parser.add_argument('-seqFile_preTraining', dest='seqFile_preTraining', type=str, help='Sequence', required=False, default=INPUT_FILE_PREFIX_PRE)
    parser.add_argument('-labFile_transTraining', dest='labFile_transTraining', type=str, help='Label', required=False, default=INPUT_FILE_PREFIX_PRE)
    parser.add_argument('-seqFile_preTrain', dest='seqFile_preTrain', type=str, help='Sequence', required=False, default=INPUT_FILE_PREFIX_TRANS)
    parser.add_argument('-labFile_preTrain', dest='labFile_preTrain', type=str, help='Label', required=False, default=INPUT_FILE_PREFIX_TRANS)
    parser.add_argument('-model-prefix', dest='modelPrefix', type=str, help='model prefix', required=False, default=OUTPUT_FILE_PREFIX + "model.h5")

    args = parser.parse_args()

    # pre_training
    seqFile_preTraining = args.seqFile_indepTest
    labFile_transTraining = args.labFile_indepTest

    # training
    seqFile_preTrain = args.seqFile_train
    labFile_preTrain = args.seqFile_train

    modelPrefix = args.modelPrefix

    xx1, yy1, mm1, xx2, yy2, mm2, valX1, valY1, valM1, valX2, valY2, valM2, testX2, testY2, testM2 = sampleSet(seqFile_preTraining, labFile_transTraining, seqFile_preTrain, labFile_preTrain)

    ############################## creat model ####################################
    model = RNApar_model(xx1, CNN1D_KER, CNN1D_FIL, LSTM_SIZE, FC_SIZE, DROP_RATE)

    ############ optimizer ############
    adam = keras.optimizers.Adam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    model.compile(loss=masked_weighted_binary_crossentropy, optimizer=adam, metrics=['acc'])

    ############ call backs ############
    earlyStopping = EarlyStopping(monitor='val_loss', patience=10)
    checkpoint = ModelCheckpoint(filepath=modelPrefix, save_weights_only=True, monitor='val_loss', mode='min', save_best_only='True', period=1)
    tensorboard = keras.callbacks.TensorBoard(log_dir=OUTPUT_FILE_PREFIX)
    stepLR = StepLR(lr_list, epoch_list, verbose=1, info=OUTPUT_FILE_PREFIX)

    ###### pre-training、trans-training ######
    history1 = model.fit(x=[xx1, mm1], y=[yy1], batch_size=BATCH_SIZE, validation_data=([valX1, valM1], [valY1]), shuffle=True, epochs=EPOCHS, callbacks=[earlyStopping, checkpoint, stepLR])
    model.load_weights(modelPrefix)
    history2 = model.fit(x=[xx2, mm2], y=[yy2], batch_size=BATCH_SIZE, validation_data=([valX2, valM2], [valY2]), shuffle=True, epochs=EPOCHS, callbacks=[earlyStopping, checkpoint, tensorboard, stepLR])
    model.load_weights(modelPrefix)

    return -history2.history['val_loss'][-1]


def optimize(times):

    optimizer = BayesianOptimization(
        f=main,
        pbounds={
            "CNN1D_KER": (3, 6),
            "CNN1D_FIL": (32, 64),
            "LSTM_SIZE": (64, 128),
            "FC_SIZE": (32, 128)
        },
        verbose=2
    )

    optimizer.maximize(init_points=3, n_iter=times)
    print("Final result:", optimizer.max)


if __name__ == "__main__":

    os.environ['TF_KERAS'] = '1'
    optimize(times=100)
