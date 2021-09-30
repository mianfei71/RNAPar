import os
import numpy as np
from BiLSTM_mask import *
import random


def mask_func(x):
    return x[0] * x[1]


def mkdir(path):
    isExists = os.path.exists(path)

    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False


def seqConverter(seq):
    """
    Convertd the raw data to one-hot code.

    PARAMETER
    seq: "ACGU" one fasta seq,
    probMatr: Probability Matrix for Samples. Shape (nb_samples, 1, nb_length_of_sequence, nb_AA)
    """

    letterDict = {}
    letterDict["A"] = [0, 0, 0, 0, 1]
    letterDict["C"] = [0, 0, 0, 1, 0]
    letterDict["G"] = [0, 0, 1, 0, 0]
    letterDict["U"] = [0, 1, 0, 0, 0]
    letterDict["-"] = [1, 0, 0, 0, 0]

    baseCategory = 5
    baseMatr = np.zeros((len(seq), baseCategory))
    baseNo = 0
    for base in seq:
        if not base in letterDict:
            baseMatr[baseNo] = np.full(baseCategory, 0)
        else:
            baseMatr[baseNo] = letterDict[base]

        baseNo += 1
    return baseMatr


def labelConverter(seq):
    labCategory = 2
    labMatr = np.zeros((len(seq), labCategory))
    masks = np.ones((len(seq)))

    labNo = 0
    for lab in seq:
        if lab == '-':
            labMatr[labNo] = [-1, -1]  # np.full(labCategory, 0) ,    labMatr[labNo] = [1, 0]
            masks[labNo] = 0
        else:
            # [0,1] for 1
            # [1,0] for 0 and -
            labMatr[labNo] = [1 - float(lab), float(lab)]
        labNo += 1

    return (labMatr, masks) # 返回元组


def maskConverter(seq):
    masks = np.ones((len(seq)))

    labNo = 0
    for lab in seq:
        if lab == '-':
            masks[labNo] = 0
        labNo += 1

    return masks


def seqPreprocessor(file):
    """
    This function tures the fasta file to dataframe
    :param file: fasta file
    :return: id and seq
    """
    id = ''
    seq = ''
    seqs = []
    ids = []
    for line in open(file, 'r', encoding="ISO-8859-1"):
        if line[0] == '>':
            if id != '':
                seqs.append(seq.upper())
                ids.append(id)

            id = line.strip()
            seq = ''
        elif line.strip() != '':
            seq = seq + line.strip()

    if id != '':
        seqs.append(seq.upper())
        ids.append(id)
    return (ids, seqs)


def labelPreprocessor(file):

    rna_id = ''
    rna_seq = ''
    labs = []
    ids = []

    for line in open(file, 'r', encoding="ISO-8859-1"):
        if line[0] == '>':
            if rna_id != '':
                labs.append(rna_seq)
                ids.append(rna_id)
            rna_id = line.strip()
            rna_seq = ''
        elif line.strip() != '':
            rna_seq = rna_seq + line.strip()

    if rna_id != '':
        labs.append(rna_seq)
        ids.append(rna_id)

    return (ids, labs)


def masked_binary_crossentropy(y_true, y_pred, weight_zero=1.0, weight_one=1.0):

    masked_y_pred = tf.multiply(y_pred, tf.cast(tf.not_equal(y_true, -1), tf.float32))  # -1 是mask的标记  (?, 200, 2)
    masked_y_true = tf.multiply(y_true, tf.cast(tf.not_equal(y_true, -1), tf.float32))

    weights = masked_y_true * weight_one + (1. - masked_y_true) * weight_zero
    masked_weights = tf.multiply(weights, tf.cast(tf.not_equal(y_true, -1), tf.float32))

    bin_crossentropy = K.binary_crossentropy(masked_y_true, masked_y_pred)

    weighted_bin_crossentropy = masked_weights * bin_crossentropy
    weighted_bin_crossentropy = tf.multiply(weighted_bin_crossentropy, tf.cast(tf.not_equal(y_true, -1.0), tf.float32))

    res = K.sum(weighted_bin_crossentropy, axis=-2)[:, 1:2]
    res = res / (K.sum(tf.cast(tf.not_equal(y_true, -1), tf.float32)) / 2)
    return res


def unduplicate(valX, valY):

    old_dim_X = np.shape(valX)
    old_dim_Y = np.shape(valY)
    print(old_dim_X);  print(old_dim_Y)
    valX = valX.reshape((old_dim_X[0], old_dim_X[1]*old_dim_X[2]))
    valY = valY.reshape((old_dim_Y[0], old_dim_Y[1]*old_dim_Y[2]))
    print(np.shape(valX));  print(np.shape(valY))

    valX_Y = np.concatenate((valX, valY), axis=1)
    valX_Y = np.array(list(set([tuple(row) for row in valX_Y])))
    valX = valX_Y[:, range(0, int(old_dim_X[1]*old_dim_X[2]))]
    valY = valX_Y[:, range(int(old_dim_X[1]*old_dim_X[2]), int(old_dim_X[1]*old_dim_X[2] + old_dim_Y[1]*old_dim_Y[2]))]

    new_dim_X = np.shape(valX)
    new_dim_Y = np.shape(valY)

    valX = valX.reshape((new_dim_X[0], old_dim_X[1], old_dim_X[2]))
    valY = valY.reshape((new_dim_Y[0], old_dim_Y[1], old_dim_Y[2]))
    return valX, valY


def sampleSet(seqFile, labFile, seqFile2, labFile2, N_CLASS=2):

    (ids, seqs) = seqPreprocessor(seqFile)
    (ids, labs) = labelPreprocessor(labFile)
    (ids2, seqs2) = seqPreprocessor(seqFile2)
    (ids2, labs2) = labelPreprocessor(labFile2)

    rawData = list(zip(seqs, labs))  # seqs 和 labels 对应打包成元组（对象），list 强制转换
    rawData2 = list(zip(seqs2, labs2))  # seqs 和 labels 对应打包成元组（对象），list 强制转换

    inputX = [seqConverter(i[0]) for i in rawData]
    inputX2 = [seqConverter(i[0]) for i in rawData2]

    inputY = [labelConverter(i[1]) for i in rawData]
    inputY2 = [labelConverter(i[1]) for i in rawData2]

    data = list(zip(inputX, inputY))
    data2 = list(zip(inputX2, inputY2))
    #    random.seed(4)

    random.shuffle(data)  # Due to step == seq length, so shuffle could not produce overlap between training set and validation set
    random.shuffle(data2)

    # 共17927
    # training : validation : test = 7:2:1
    train_num1 = int(len(inputX) * 0.9)  # 原来是0.8
    val_num1 = int(len(inputX) * 1)
    train = data[0:(train_num1)]
    val1 = data[0:val_num1]  # 原为：(train_num1 + 1):val_num1

    # 共3635
    train_num2 = int(len(inputX2) * 0.8)
    val_num2 = int(len(inputX2) * 0.9)
    train2 = data2[0:(train_num2)]
    val2 = data2[(train_num2 + 1):val_num2]
    test2 = data2[(val_num2 + 1):]

    # training set
    # random.shuffle(train)
    trainX1 = [i[0] for i in train]
    trainY1 = [i[1] for i in train]
    trainX2 = [i[0] for i in train2]
    trainY2 = [i[1] for i in train2]

    xx1 = np.dstack(trainX1)
    xx1 = np.rollaxis(xx1, -1)
    # xx1 = np.array(trainX1)
    yy1 = np.dstack([Tuple[0] for Tuple in trainY1])
    yy1 = np.rollaxis(yy1, -1)
    mm1 = np.dstack([Tuple[1] for Tuple in trainY1])
    mm1 = np.rollaxis(mm1, -1)
    mm1 = np.rollaxis(mm1, 2, 1)

    xx2 = np.dstack(trainX2)
    xx2 = np.rollaxis(xx2, -1)
    # xx2 = np.array(trainX2)
    yy2 = np.dstack([Tuple[0] for Tuple in trainY2])
    yy2 = np.rollaxis(yy2, -1)
    mm2 = np.dstack([Tuple[1] for Tuple in trainY2])
    mm2 = np.rollaxis(mm2, -1)
    mm2 = np.rollaxis(mm2, 2, 1)

    # validation set
    valX1 = [i[0] for i in val1]
    valYM = [i[1] for i in val1]
    valX1 = np.dstack(valX1)
    valX1 = np.rollaxis(valX1, -1)
    # valX1 = np.array(valX1)
    valY1 = np.dstack([Tuple[0] for Tuple in valYM])
    valY1 = np.rollaxis(valY1, -1)
    valM1 = np.dstack([Tuple[1] for Tuple in valYM])
    valM1 = np.rollaxis(valM1, -1)
    valM1 = np.rollaxis(valM1, 2, 1)

    # validation 2 set
    valX2 = [i[0] for i in val2]
    valYM2 = [i[1] for i in val2]
    valX2 = np.dstack(valX2)
    valX2 = np.rollaxis(valX2, -1)
    # valX2 = np.array(valX2)
    valY2 = np.dstack([Tuple[0] for Tuple in valYM2])
    valY2 = np.rollaxis(valY2, -1)
    valM2 = np.dstack([Tuple[1] for Tuple in valYM2])
    valM2 = np.rollaxis(valM2, -1)
    valM2 = np.rollaxis(valM2, 2, 1)

    # test 2 set
    testX2 = [i[0] for i in test2]
    testYM2 = [i[1] for i in test2]
    testX2 = np.dstack(testX2)
    testX2 = np.rollaxis(testX2, -1)
    # testX2 = np.array(testX2)
    testY2 = np.dstack([Tuple[0] for Tuple in testYM2])
    testY2 = np.rollaxis(testY2, -1)
    testM2 = np.dstack([Tuple[1] for Tuple in testYM2])
    testM2 = np.rollaxis(testM2, -1)
    testM2 = np.rollaxis(testM2, 2, 1)

    return (xx1, yy1, mm1, xx2, yy2, mm2, valX1, valY1, valM1, valX2, valY2, valM2, testX2, testY2, testM2)


def sampleSetPredict(seqFile):

    (ids, seqs) = seqPreprocessor(seqFile)
    rawData = list(zip(seqs))  # seqs 和 labels 对应打包成元组（对象），list 强制转换

    inputX = [seqConverter(i[0]) for i in rawData]
    data = list(zip(inputX))
    train = data
    trainX1 = [i[0] for i in train]

    xx1 = np.dstack(trainX1)
    xx1 = np.rollaxis(xx1, -1)

    mm1 = np.dstack([item[:,0] for item in trainX1])
    mm1 = np.rollaxis(mm1, -1)
    mm1 = np.rollaxis(mm1, 2, 1)

    return (xx1, mm1)
