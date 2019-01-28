import math
import os
import time
from argparse import ArgumentParser
from functools import reduce

import configparser
import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, concatenate, BatchNormalization
from keras.layers import Dropout
from keras.models import Model
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from utils.acc_history_plot import AccHistoryPlot
from utils.basic_modules import conjoint_cnn, conjoint_sae
from utils.sequence_encoder import ProEncoder, RNAEncoder
from utils.stacked_auto_encoder import train_auto_encoder

# default program settings
DATA_SET = 'RPI488'
DATA_BASE_PATH = '../data/'
RESULT_BASE_PATH = '../result/'
TIME_FORMAT = "-%y-%m-%d-%H-%M-%S"
INI_PATH = './utils/data_set_settings.ini'
WINDOW_P_UPLIMIT = 3
WINDOW_P_STRUCT_UPLIMIT = 3
WINDOW_R_UPLIMIT = 4
WINDOW_R_STRUCT_UPLIMIT = 4
VECTOR_REPETITION_CNN = 1
RANDOM_SEED = 1
K_FOLD = 5
BATCH_SIZE = 150
FIRST_TRAIN_EPOCHS = [20, 20, 20, 20, 10]
SECOND_TRAIN_EPOCHS = [20, 20, 20, 20, 10]
PATIENCES = [10, 10, 10, 10, 10]
FIRST_OPTIMIZER = 'adam'
SECOND_OPTIMIZER = 'sgd'
CODING_FREQUENCY = True
MONITOR = 'acc'
MIN_DELTA = 0.0
SHUFFLE = True
VERBOSE = 2

metrics_whole = {'RPITER': np.zeros(6),
                 'Conjoint-SAE': np.zeros(6), 'Conjoint-Struct-SAE': np.zeros(6),
                 'Conjoint-CNN': np.zeros(6), 'Conjoint-Struct-CNN': np.zeros(6),
}

parser = ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, help='The dataset you want to process.')
args = parser.parse_args()
if args.dataset != None:
    DATA_SET = args.dataset
print("Dataset: %s" % DATA_SET)

# gpu memory growth for tensorflow
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)


# result save path
result_save_path = RESULT_BASE_PATH + DATA_SET + "/" + DATA_SET + time.strftime(TIME_FORMAT, time.localtime()) + "/"
if not os.path.exists(result_save_path):
    os.mkdir(result_save_path)
out = open(result_save_path + 'result.txt', 'w')


def read_data_pair(path):
    pos_pairs = []
    neg_pairs = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            p, r, label = line.split('\t')
            if label == '1':
                pos_pairs.append((p, r))
            elif label == '0':
                neg_pairs.append((p, r))
    return pos_pairs, neg_pairs


def read_data_seq(path):
    seq_dict = {}
    with open(path, 'r') as f:
        name = ''
        for line in f:
            line = line.strip()
            if line[0] == '>':
                name = line[1:]
                seq_dict[name] = ''
            else:
                if line.startswith('XXX'):
                    seq_dict.pop(name)
                else:
                    seq_dict[name] = line
    return seq_dict


# calculate the 6 metrics of Acc, Sn, Sp, Precision, MCC and AUC
def calc_metrics(y_label, y_proba):
    con_matrix = confusion_matrix(y_label, [1 if x >= 0.5 else 0 for x in y_proba])
    TN = float(con_matrix[0][0])
    FP = float(con_matrix[0][1])
    FN = float(con_matrix[1][0])
    TP = float(con_matrix[1][1])
    P = TP + FN
    N = TN + FP
    Sn = TP / P
    Sp = TN / N
    Acc = (TP + TN) / (P + N)
    Pre = (TP) / (TP + FP)
    MCC = 0
    tmp = math.sqrt((TP + FP) * (TP + FN)) * math.sqrt((TN + FP) * (TN + FN))
    if tmp != 0:
        MCC = (TP * TN - FP * FN) / tmp
    fpr, tpr, thresholds = roc_curve(y_label, y_proba)
    AUC = auc(fpr, tpr)
    return Acc, Sn, Sp, Pre, MCC, AUC


def load_data(data_set):
    pro_seqs = read_data_seq(DATA_BASE_PATH + "sequence/" + data_set + '_protein_seq.fa')
    rna_seqs = read_data_seq(DATA_BASE_PATH + "sequence/" + data_set + '_rna_seq.fa')
    pro_structs = read_data_seq(DATA_BASE_PATH + "structure/" + data_set + '_protein_struct.fa')
    rna_structs = read_data_seq(DATA_BASE_PATH + "structure/" + data_set + '_rna_struct.fa')
    pos_pairs, neg_pairs = read_data_pair(DATA_BASE_PATH + data_set + '_pairs.txt')

    return pos_pairs, neg_pairs, pro_seqs, rna_seqs, pro_structs, rna_structs


def coding_pairs(pairs, pro_seqs, rna_seqs, pro_structs, rna_structs, PE, RE, kind):
    samples = []
    for pr in pairs:
        if pr[0] in pro_seqs and pr[1] in rna_seqs and pr[0] in pro_structs and pr[1] in rna_structs:
            p_seq = pro_seqs[pr[0]]  # protein sequence
            r_seq = rna_seqs[pr[1]]  # rna sequence
            p_struct = pro_structs[pr[0]]  # protein structure
            r_struct = rna_structs[pr[1]]  # rna structure

            p_conjoint = PE.encode_conjoint(p_seq)
            r_conjoint = RE.encode_conjoint(r_seq)
            p_conjoint_struct = PE.encode_conjoint_struct(p_seq, p_struct)
            r_conjoint_struct = RE.encode_conjoint_struct(r_seq, r_struct)

            if p_conjoint is 'Error':
                print('Skip {} in pair {} according to conjoint coding process.'.format(pr[0], pr))
            elif r_conjoint is 'Error':
                print('Skip {} in pair {} according to conjoint coding process.'.format(pr[1], pr))
            elif p_conjoint_struct is 'Error':
                print('Skip {} in pair {} according to conjoint_struct coding process.'.format(pr[0], pr))
            elif r_conjoint_struct is 'Error':
                print('Skip {} in pair {} according to conjoint_struct coding process.'.format(pr[1], pr))

            else:
                samples.append([[p_conjoint, r_conjoint],
                                [p_conjoint_struct, r_conjoint_struct],
                                kind])
        else:
            print('Skip pair {} according to sequence dictionary.'.format(pr))
    return samples


def standardization(X):
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler


def pre_process_data(samples, samples_pred=None):
    # np.random.shuffle(samples)

    p_conjoint = np.array([x[0][0] for x in samples])
    r_conjoint = np.array([x[0][1] for x in samples])
    p_conjoint_struct = np.array([x[1][0] for x in samples])
    r_conjoint_struct = np.array([x[1][1] for x in samples])
    y_samples = np.array([x[2] for x in samples])

    p_conjoint, scaler_p = standardization(p_conjoint)
    r_conjoint, scaler_r = standardization(r_conjoint)
    p_conjoint_struct, scaler_p_struct = standardization(p_conjoint_struct)
    r_conjoint_struct, scaler_r_struct = standardization(r_conjoint_struct)

    p_conjoint_cnn = np.array([list(map(lambda e: [e] * VECTOR_REPETITION_CNN, x)) for x in p_conjoint])
    r_conjoint_cnn = np.array([list(map(lambda e: [e] * VECTOR_REPETITION_CNN, x)) for x in r_conjoint])
    p_conjoint_struct_cnn = np.array([list(map(lambda e: [e] * VECTOR_REPETITION_CNN, x)) for x in p_conjoint_struct])
    r_conjoint_struct_cnn = np.array([list(map(lambda e: [e] * VECTOR_REPETITION_CNN, x)) for x in r_conjoint_struct])

    p_ctf_len = 7 ** WINDOW_P_UPLIMIT
    r_ctf_len = 4 ** WINDOW_R_UPLIMIT
    p_conjoint_previous = np.array([x[-p_ctf_len:] for x in p_conjoint])
    r_conjoint_previous = np.array([x[-r_ctf_len:] for x in r_conjoint])

    X_samples = [[p_conjoint, r_conjoint],
                 [p_conjoint_struct, r_conjoint_struct],
                 [p_conjoint_cnn, r_conjoint_cnn],
                 [p_conjoint_struct_cnn, r_conjoint_struct_cnn],
                 [p_conjoint_previous, r_conjoint_previous]
                 ]

    if samples_pred:
        # np.random.shuffle(samples_pred)

        p_conjoint_pred = np.array([x[0][0] for x in samples_pred])
        r_conjoint_pred = np.array([x[0][1] for x in samples_pred])
        p_conjoint_struct_pred = np.array([x[1][0] for x in samples_pred])
        r_conjoint_struct_pred = np.array([x[1][1] for x in samples_pred])
        y_samples_pred = np.array([x[2] for x in samples_pred])

        p_conjoint_pred = scaler_p.transform(p_conjoint_pred)
        r_conjoint_pred = scaler_r.transform(r_conjoint_pred)
        p_conjoint_struct_pred = scaler_p_struct.transform(p_conjoint_struct_pred)
        r_conjoint_struct_pred = scaler_r_struct.transform(r_conjoint_struct_pred)

        p_conjoint_cnn_pred = np.array([list(map(lambda e: [e] * VECTOR_REPETITION_CNN, x)) for x in p_conjoint_pred])
        r_conjoint_cnn_pred = np.array([list(map(lambda e: [e] * VECTOR_REPETITION_CNN, x)) for x in r_conjoint_pred])
        p_conjoint_struct_cnn_pred = np.array(
            [list(map(lambda e: [e] * VECTOR_REPETITION_CNN, x)) for x in p_conjoint_struct_pred])
        r_conjoint_struct_cnn_pred = np.array(
            [list(map(lambda e: [e] * VECTOR_REPETITION_CNN, x)) for x in r_conjoint_struct_pred])

        p_conjoint_previous_pred = np.array([x[-p_ctf_len:] for x in p_conjoint_pred])
        r_conjoint_previous_pred = np.array([x[-r_ctf_len:] for x in r_conjoint_pred])

        X_samples_pred = [[p_conjoint_pred, r_conjoint_pred],
                          [p_conjoint_struct_pred, r_conjoint_struct_pred],
                          [p_conjoint_cnn_pred, r_conjoint_cnn_pred],
                          [p_conjoint_struct_cnn_pred, r_conjoint_struct_cnn_pred],
                          [p_conjoint_previous_pred, r_conjoint_previous_pred]
                          ]

        return X_samples, y_samples, X_samples_pred, y_samples_pred

    else:
        return X_samples, y_samples


def sum_power(num, bottom, top):
    return reduce(lambda x, y: x + y, map(lambda x: num ** x, range(bottom, top + 1)))


def get_callback_list(patience, weight_path, result_path, stage, fold, X_test, y_test):
    earlystopping = EarlyStopping(monitor='acc', min_delta=MIN_DELTA, patience=patience, verbose=1,
                                  mode='auto')
    checkpoint = ModelCheckpoint(weight_path, monitor=MONITOR, verbose=1, save_best_only=True,
                                 save_weights_only=True, mode='auto', period=1)
    acchistory = AccHistoryPlot([stage, fold], [X_test, y_test], data_name=DATA_SET,
                                result_save_path=result_path, validate=0, plot_epoch_gap=5)

    return [acchistory, earlystopping, checkpoint]


def get_auto_encoders(X_train, X_test, batch_size=BATCH_SIZE):
    encoders_protein, decoders_protein, train_tmp_p, test_tmp_p = train_auto_encoder(
        X_train=X_train[0],
        X_test=X_test[0],
        layers=[X_train[0].shape[1], 256, 128, 64], batch_size=batch_size)
    encoders_rna, decoders_rna, train_tmp_r, test_tmp_r = train_auto_encoder(
        X_train=X_train[1],
        X_test=X_test[1],
        layers=[X_train[1].shape[1], 256, 128, 64], batch_size=batch_size)
    return encoders_protein, encoders_rna


# load data settings
if DATA_SET in ['RPI369', 'RPI488', 'RPI1807', 'RPI2241', 'NPInter']:
    config = configparser.ConfigParser()
    config.read(INI_PATH)
    WINDOW_P_UPLIMIT = config.getint(DATA_SET, 'WINDOW_P_UPLIMIT')
    WINDOW_P_STRUCT_UPLIMIT = config.getint(DATA_SET, 'WINDOW_P_STRUCT_UPLIMIT')
    WINDOW_R_UPLIMIT = config.getint(DATA_SET, 'WINDOW_R_UPLIMIT')
    WINDOW_R_STRUCT_UPLIMIT = config.getint(DATA_SET, 'WINDOW_R_STRUCT_UPLIMIT')
    VECTOR_REPETITION_CNN = config.getint(DATA_SET, 'VECTOR_REPETITION_CNN')
    RANDOM_SEED = config.getint(DATA_SET, 'RANDOM_SEED')
    K_FOLD = config.getint(DATA_SET, 'K_FOLD')
    BATCH_SIZE = config.getint(DATA_SET, 'BATCH_SIZE')
    PATIENCES = [int(x) for x in config.get(DATA_SET, 'PATIENCES').replace('[', '').replace(']', '').split(',')]
    FIRST_TRAIN_EPOCHS = [int(x) for x in
                          config.get(DATA_SET, 'FIRST_TRAIN_EPOCHS').replace('[', '').replace(']', '').split(',')]
    SECOND_TRAIN_EPOCHS = [int(x) for x in
                           config.get(DATA_SET, 'SECOND_TRAIN_EPOCHS').replace('[', '').replace(']', '').split(',')]
    FIRST_OPTIMIZER = config.get(DATA_SET, 'FIRST_OPTIMIZER')
    SECOND_OPTIMIZER = config.get(DATA_SET, 'SECOND_OPTIMIZER')
    CODING_FREQUENCY = config.getboolean(DATA_SET, 'CODING_FREQUENCY')
    MONITOR = config.get(DATA_SET, 'MONITOR')
    MIN_DELTA = config.getfloat(DATA_SET, 'MIN_DELTA')

# write program parameter settings to  result file
settings = (
    """# Analyze data set {}\n
Program parameters:
WINDOW_P_UPLIMIT = {},
WINDOW_R_UPLIMIT = {},
WINDOW_P_STRUCT_UPLIMIT = {},
WINDOW_R_STRUCT_UPLIMIT = {},
VECTOR_REPETITION_CNN = {},
RANDOM_SEED = {},
K_FOLD = {},
BATCH_SIZE = {},
FIRST_TRAIN_EPOCHS = {},
SECOND_TRAIN_EPOCHS = {},
PATIENCES = {},
FIRST_OPTIMIZER = {},
SECOND_OPTIMIZER = {},
CODING_FREQUENCY = {},
MONITOR = {},
MIN_DELTA = {},
    """.format(DATA_SET, WINDOW_P_UPLIMIT, WINDOW_R_UPLIMIT, WINDOW_P_STRUCT_UPLIMIT,
               WINDOW_R_STRUCT_UPLIMIT, VECTOR_REPETITION_CNN,
               RANDOM_SEED, K_FOLD, BATCH_SIZE, FIRST_TRAIN_EPOCHS, SECOND_TRAIN_EPOCHS, PATIENCES, FIRST_OPTIMIZER,
               SECOND_OPTIMIZER,
               CODING_FREQUENCY, MONITOR, MIN_DELTA)
)
out.write(settings)

PRO_CODING_LENGTH = sum_power(7, 1, WINDOW_P_UPLIMIT)
PRO_STRUCT_CODING_LENGTH = sum_power(7, 1, WINDOW_P_UPLIMIT) + sum_power(3, 1, WINDOW_P_STRUCT_UPLIMIT)
RNA_CODING_LENGTH = sum_power(4, 1, WINDOW_R_UPLIMIT)
RNA_STRUCT_CODING_LENGTH = sum_power(4, 1, WINDOW_R_UPLIMIT) + sum_power(2, 1, WINDOW_R_STRUCT_UPLIMIT)

# read rna-protein pairs and sequences from data files
pos_pairs, neg_pairs, pro_seqs, rna_seqs, pro_structs, rna_structs = load_data(DATA_SET)

# sequence encoder instance
PE = ProEncoder(WINDOW_P_UPLIMIT, WINDOW_P_STRUCT_UPLIMIT, CODING_FREQUENCY, VECTOR_REPETITION_CNN)
RE = RNAEncoder(WINDOW_R_UPLIMIT, WINDOW_R_STRUCT_UPLIMIT, CODING_FREQUENCY, VECTOR_REPETITION_CNN)

print("Coding positive protein-rna pairs.\n")
samples = coding_pairs(pos_pairs, pro_seqs, rna_seqs, pro_structs, rna_structs, PE, RE, kind=1)
positive_sample_number = len(samples)
print("Coding negative protein-rna pairs.\n")
samples += coding_pairs(neg_pairs, pro_seqs, rna_seqs, pro_structs, rna_structs, PE, RE, kind=0)
negative_sample_number = len(samples) - positive_sample_number
sample_num = len(samples)

# sample numbers for the positive and the negative
print('\nPos samples: {}, Neg samples: {}.\n'.format(positive_sample_number, negative_sample_number))
out.write('\nPos samples: {}, Neg samples: {}.\n'.format(positive_sample_number, negative_sample_number))

X, y = pre_process_data(samples=samples)

# K-fold CV processes

# skf = StratifiedKFold(n_splits=K_FOLD, random_state=RANDOM_SEED, shuffle=True)
# fold = 0
print('\n\nK-fold cross validation processes:\n')
out.write('\n\nK-fold cross validation processes:\n')
for fold in range(K_FOLD):
    train = [i for i in range(sample_num) if i%K_FOLD !=fold]
    test = [i for i in range(sample_num) if i%K_FOLD ==fold]
    print(test)
    # generate train and test data
    X_train_conjoint = [X[0][0][train], X[0][1][train]]
    X_train_conjoint_struct = [X[1][0][train], X[1][1][train]]
    X_train_conjoint_cnn = [X[2][0][train], X[2][1][train]]
    X_train_conjoint_struct_cnn = [X[3][0][train], X[3][1][train]]
    X_train_conjoint_previous = [X[4][0][train], X[4][1][train]]

    X_test_conjoint = [X[0][0][test], X[0][1][test]]
    X_test_conjoint_struct = [X[1][0][test], X[1][1][test]]
    X_test_conjoint_cnn = [X[2][0][test], X[2][1][test]]
    X_test_conjoint_struct_cnn = [X[3][0][test], X[3][1][test]]
    X_test_conjoint_previous = [X[4][0][test], X[4][1][test]]

    y_train_mono = y[train]
    y_train = np_utils.to_categorical(y_train_mono, 2)
    y_test_mono = y[test]
    y_test = np_utils.to_categorical(y_test_mono, 2)

    X_ensemble_train = X_train_conjoint + X_train_conjoint_struct + X_train_conjoint_cnn + X_train_conjoint_struct_cnn
    X_ensemble_test = X_test_conjoint + X_test_conjoint_struct + X_test_conjoint_cnn + X_test_conjoint_struct_cnn


    print(str('\n# ' + '=' * 10 + " Fold {} " + "=" * 10 + '\n').format(fold))
    out.write(str('\n# ' + '=' * 10 + " Fold {} " + "=" * 10 + '\n').format(fold))
    model_metrics = {'RPITER': np.zeros(6),
                     'Conjoint-SAE': np.zeros(6), 'Conjoint-Struct-SAE': np.zeros(6),
                     'Conjoint-CNN': np.zeros(6), 'Conjoint-Struct-CNN': np.zeros(6),
                     }
    model_weight_path = result_save_path + 'weights.hdf5'

    module_index = 0
    # =================================================================
    # Conjoint-CNN module

    stage = 'Conjoint-CNN'
    print("\n# Module Conjoint-CNN part #\n")

    # create model
    model_conjoint_cnn = conjoint_cnn(PRO_CODING_LENGTH, RNA_CODING_LENGTH, VECTOR_REPETITION_CNN)
    callbacks = get_callback_list(PATIENCES[0], model_weight_path, result_save_path, stage, fold, X_test_conjoint_cnn,
                                  y_test)

    # first train
    model_conjoint_cnn.compile(loss='categorical_crossentropy', optimizer=FIRST_OPTIMIZER, metrics=['accuracy'])
    callbacks[0].close_plt_on_train_end = False
    model_conjoint_cnn.fit(x=X_train_conjoint_cnn,
                           y=y_train,
                           epochs=FIRST_TRAIN_EPOCHS[0],
                           batch_size=BATCH_SIZE,
                           verbose=VERBOSE,
                           shuffle=SHUFFLE,
                           callbacks=callbacks)
    model_conjoint_cnn.load_weights(model_weight_path)

    # second train
    model_conjoint_cnn.compile(loss='categorical_crossentropy', optimizer=SECOND_OPTIMIZER, metrics=['accuracy'])
    callbacks[0].close_plt_on_train_end = True
    model_conjoint_cnn.fit(x=X_train_conjoint_cnn,
                           y=y_train,
                           epochs=SECOND_TRAIN_EPOCHS[0],
                           batch_size=BATCH_SIZE,
                           verbose=VERBOSE,
                           shuffle=SHUFFLE,
                           callbacks=callbacks)
    model_conjoint_cnn.load_weights(model_weight_path)

    # test
    y_test_predict = model_conjoint_cnn.predict(X_test_conjoint_cnn)
    model_metrics['Conjoint-CNN'] = np.array(calc_metrics(y_test[:, 1], y_test_predict[:, 1]))
    print('Best performance for module Conjoint-CNN:\n {}\n'.format(model_metrics['Conjoint-CNN'].tolist()))
    # =================================================================


    # =================================================================
    # Conjoint-Struct-CNN module

    stage = "Conjoint-Struct-CNN"
    print("\n# Module Conjoint-Struct-CNN part #\n")
    module_index += 1

    # create model
    model_conjoint_struct_cnn = conjoint_cnn(PRO_STRUCT_CODING_LENGTH, RNA_STRUCT_CODING_LENGTH, VECTOR_REPETITION_CNN)
    callbacks = get_callback_list(PATIENCES[1], model_weight_path, result_save_path, stage, fold,
                                  X_test_conjoint_struct_cnn, y_test)

    # first train
    model_conjoint_struct_cnn.compile(loss='categorical_crossentropy', optimizer=FIRST_OPTIMIZER, metrics=['accuracy'])
    callbacks[0].close_plt_on_train_end = False
    model_conjoint_struct_cnn.fit(x=X_train_conjoint_struct_cnn,
                                  y=y_train,
                                  epochs=FIRST_TRAIN_EPOCHS[1],
                                  batch_size=BATCH_SIZE,
                                  verbose=VERBOSE,
                                  shuffle=SHUFFLE,
                                  callbacks=callbacks)
    model_conjoint_struct_cnn.load_weights(model_weight_path)

    # second train
    model_conjoint_struct_cnn.compile(loss='categorical_crossentropy', optimizer=SECOND_OPTIMIZER, metrics=['accuracy'])
    callbacks[0].close_plt_on_train_end = True
    model_conjoint_struct_cnn.fit(x=X_train_conjoint_struct_cnn,
                                  y=y_train,
                                  epochs=SECOND_TRAIN_EPOCHS[1],
                                  batch_size=BATCH_SIZE,
                                  verbose=VERBOSE,
                                  shuffle=SHUFFLE,
                                  callbacks=callbacks)
    model_conjoint_struct_cnn.load_weights(model_weight_path)

    # test
    y_test_predict = model_conjoint_struct_cnn.predict(X_test_conjoint_struct_cnn)
    model_metrics['Conjoint-Struct-CNN'] = np.array(calc_metrics(y_test[:, 1], y_test_predict[:, 1]))
    print(
        'Best performance for module Conjoint-Struct-CNN:\n {}\n'.format(model_metrics['Conjoint-Struct-CNN'].tolist()))
    # =================================================================


    # =================================================================
    # Conjoint-SAE module

    stage = 'Conjoint-SAE'
    print("\n# Module Conjoint-SAE part #\n")
    module_index += 1

    # create model
    encoders_pro, encoders_rna = get_auto_encoders(X_train_conjoint, X_test_conjoint)
    model_conjoint_sae = conjoint_sae(encoders_pro, encoders_rna, PRO_CODING_LENGTH, RNA_CODING_LENGTH)
    callbacks = get_callback_list(PATIENCES[2], model_weight_path, result_save_path, stage, fold, X_test_conjoint,
                                  y_test)

    # first train
    model_conjoint_sae.compile(loss='categorical_crossentropy', optimizer=FIRST_OPTIMIZER, metrics=['accuracy'])
    callbacks[0].close_plt_on_train_end = False
    model_conjoint_sae.fit(x=X_train_conjoint,
                           y=y_train,
                           epochs=FIRST_TRAIN_EPOCHS[2],
                           batch_size=BATCH_SIZE,
                           verbose=VERBOSE,
                           shuffle=SHUFFLE,
                           callbacks=callbacks)
    model_conjoint_sae.load_weights(model_weight_path)

    # second train
    model_conjoint_sae.compile(loss='categorical_crossentropy', optimizer=SECOND_OPTIMIZER, metrics=['accuracy'])
    callbacks[0].close_plt_on_train_end = True
    model_conjoint_sae.fit(x=X_train_conjoint,
                           y=y_train,
                           epochs=SECOND_TRAIN_EPOCHS[2],
                           batch_size=BATCH_SIZE,
                           verbose=VERBOSE,
                           shuffle=SHUFFLE,
                           callbacks=callbacks)
    model_conjoint_sae.load_weights(model_weight_path)

    # test
    y_test_predict = model_conjoint_sae.predict(X_test_conjoint)
    model_metrics['Conjoint-SAE'] = np.array(calc_metrics(y_test[:, 1], y_test_predict[:, 1]))
    print('Best performance for module Conjoint-SAE:\n {}\n'.format(model_metrics['Conjoint-SAE'].tolist()))
    # =================================================================


    # =================================================================
    # Conjoint-Struct-SAE module

    stage = 'Conjoint-Struct-SAE'
    print("\n# Module Conjoint-Struct-SAE part #\n")
    module_index += 1

    # create model
    encoders_pro, encoders_rna = get_auto_encoders(X_train_conjoint_struct, X_test_conjoint_struct)
    model_conjoint_struct_sae = conjoint_sae(encoders_pro, encoders_rna, PRO_STRUCT_CODING_LENGTH,
                                             RNA_STRUCT_CODING_LENGTH)
    callbacks = get_callback_list(PATIENCES[3], model_weight_path, result_save_path, stage, fold,
                                  X_test_conjoint_struct,
                                  y_test)

    # first train
    model_conjoint_struct_sae.compile(loss='categorical_crossentropy', optimizer=FIRST_OPTIMIZER, metrics=['accuracy'])
    callbacks[0].close_plt_on_train_end = False
    model_conjoint_struct_sae.fit(x=X_train_conjoint_struct,
                                  y=y_train,
                                  epochs=FIRST_TRAIN_EPOCHS[3],
                                  batch_size=BATCH_SIZE,
                                  verbose=VERBOSE,
                                  shuffle=SHUFFLE,
                                  callbacks=callbacks)
    model_conjoint_struct_sae.load_weights(model_weight_path)
    # second train
    model_conjoint_struct_sae.compile(loss='categorical_crossentropy', optimizer=SECOND_OPTIMIZER, metrics=['accuracy'])
    callbacks[0].close_plt_on_train_end = True
    model_conjoint_struct_sae.fit(x=X_train_conjoint_struct,
                                  y=y_train,
                                  epochs=SECOND_TRAIN_EPOCHS[3],
                                  batch_size=BATCH_SIZE,
                                  verbose=VERBOSE,
                                  shuffle=SHUFFLE,
                                  callbacks=callbacks)
    model_conjoint_struct_sae.load_weights(model_weight_path)

    # test
    y_test_predict = model_conjoint_struct_sae.predict(X_test_conjoint_struct)
    model_metrics['Conjoint-Struct-SAE'] = np.array(calc_metrics(y_test[:, 1], y_test_predict[:, 1]))
    print(
        'Best performance for module Conjoint-Struct-SAE:\n {}\n'.format(model_metrics['Conjoint-Struct-SAE'].tolist()))
    # =================================================================



    # =================================================================
    # module ensemble

    stage = 'RPITER'
    print("\n# Module Ensemble part #\n")
    module_index += 1

    # create model
    ensemble_in = concatenate([model_conjoint_sae.output, model_conjoint_struct_sae.output,
                               model_conjoint_cnn.output, model_conjoint_struct_cnn.output])
    ensemble_in = Dropout(0.25)(ensemble_in)
    ensemble = Dense(16, kernel_initializer='random_uniform', activation='relu')(ensemble_in)
    ensemble = BatchNormalization()(ensemble)
    # ensemble = Dropout(0.15)(ensemble)
    ensemble = Dense(8, kernel_initializer='random_uniform', activation='relu')(ensemble)
    ensemble = BatchNormalization()(ensemble)
    # ensemble = Dropout(0.2)(ensemble)
    ensemble_out = Dense(2, activation='softmax')(ensemble)
    model_ensemble = Model(
        inputs=model_conjoint_sae.input + model_conjoint_struct_sae.input + model_conjoint_cnn.input + model_conjoint_struct_cnn.input,
        outputs=ensemble_out)

    callbacks = get_callback_list(PATIENCES[4], model_weight_path, result_save_path, stage, fold, X_ensemble_test,
                                  y_test)

    # first train
    model_ensemble.compile(loss='categorical_crossentropy', optimizer=FIRST_OPTIMIZER, metrics=['accuracy'])
    callbacks[0].close_plt_on_train_end = False
    model_ensemble.fit(x=X_ensemble_train,
                       y=y_train,
                       epochs=FIRST_TRAIN_EPOCHS[4],
                       batch_size=BATCH_SIZE,
                       verbose=VERBOSE,
                       shuffle=SHUFFLE,
                       callbacks=callbacks)
    model_ensemble.load_weights(model_weight_path)

    # second train
    model_ensemble.compile(loss='categorical_crossentropy', optimizer=SECOND_OPTIMIZER, metrics=['accuracy'])
    callbacks[0].close_plt_on_train_end = True
    model_ensemble.fit(x=X_ensemble_train,
                       y=y_train,
                       epochs=SECOND_TRAIN_EPOCHS[4],
                       batch_size=BATCH_SIZE,
                       verbose=VERBOSE,
                       shuffle=SHUFFLE,
                       callbacks=callbacks)
    model_ensemble.load_weights(model_weight_path)

    # test
    y_test_predict = model_ensemble.predict(X_ensemble_test)
    model_metrics['RPITER'] = np.array(calc_metrics(y_test[:, 1], y_test_predict[:, 1]))
    print('Best performance for module ensemble:\n {}\n'.format(model_metrics['RPITER'].tolist()))

    # =================================================================
    for key in model_metrics:
        print(key + " : " + str(model_metrics[key].tolist()) + "\n")
        out.write(key + " : " + str(model_metrics[key].tolist()) + "\n")
    for key in model_metrics:
        metrics_whole[key] += model_metrics[key]

    # get rid of the model weights file
    if os.path.exists(model_weight_path):
        os.remove(model_weight_path)

print('\nMean metrics in {} fold:\n'.format(K_FOLD))
out.write('\nMean metrics in {} fold:\n'.format(K_FOLD))
for key in metrics_whole.keys():
    metrics_whole[key] /= K_FOLD
    print(key + " : " + str(metrics_whole[key].tolist()) + "\n")
    out.write(key + " : " + str(metrics_whole[key].tolist()) + "\n")
out.flush()
out.close()
