import argparse
from create_model import *


def main():

    os.environ['TF_KERAS'] = '1'

    INPUT = './data/test.fasta'
    OUTPUT = './predict/test.data'
    WEIGHT = "./models/weight-1.h5"
    CNN1D_KER = 6
    CNN1D_FIL = 61
    LSTM_SIZE = 115
    FC_SIZE = 53

    parser = argparse.ArgumentParser(description='RNA-par')
    parser.add_argument('-i', dest='inputFile', type=str, help='Input File', required=False, default=INPUT)
    parser.add_argument('-o', dest='outputFile', type=str, help='Output File', required=False, default=OUTPUT)
    parser.add_argument('-w', dest='weightFile', type=str, help='Weight File', required=False, default=WEIGHT)
    parser.add_argument('-K', dest='cnn1d_ker', type=str, help='Length of kernel in 1D-CNN', required=False, default=CNN1D_KER)
    parser.add_argument('-C', dest='cnn1d_fil', type=str, help='Number of channels in 1D-CNN', required=False, default=CNN1D_FIL)
    parser.add_argument('-U', dest='lstm_size', type=str, help='Number of cells in LSTM', required=False, default=LSTM_SIZE)
    parser.add_argument('-N', dest='fc_size', type=str, help='Length of unit in FC', required=False, default=FC_SIZE)

    args = parser.parse_args()

    inputFile = args.inputFile
    outputFile = args.outputFile
    weightFile = args.weightFile
    cnn1d_ker = int(args.cnn1d_ker)
    cnn1d_fil = int(args.cnn1d_fil)
    lstm_size = int(args.lstm_size)
    fc_size = int(args.fc_size)

    ########################## To predict #############################

    mkdir("./predict")

    (xx, mm) = sampleSetPredict(inputFile)

    model = RNApar_model(xx, cnn1d_ker, cnn1d_fil, lstm_size, fc_size)

    model.load_weights(weightFile)

    prediction = model.predict([xx, mm])

    with open(file=outputFile, mode='w') as w:
        for i in range(prediction.shape[0]):
            score = ' '.join([str(j) for j in prediction[i, :, 1]])
            w.write("%s\n" % (score))

    print("******  RNA-par finished.  ******")


if __name__ == "__main__":
    main()
