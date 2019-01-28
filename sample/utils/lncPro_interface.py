import os
import sys
import math
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc

dataset = sys.argv[1]
data_path = './data/'

def read_data_pair(path):
    print("Reading RPI pairs...")
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

def create_seq_file(path, seq_name, seq):
    with open(path, 'w') as f:
        f.write('>{}\n{}\n'.format(seq_name, seq))

def exec_cmd(cmd):
    r = os.popen(cmd)
    text = r.read()
    r.close()
    return text

# calculate the 6 metrics of Acc, Sn, Sp, Precision, MCC and AUC
def calc_metrics(y_label, y_proba):
    con_matrix = confusion_matrix(y_label, [1 if x >= 0.5 else 0 for x in y_proba])
    TN = con_matrix[0][0]
    FP = con_matrix[0][1]
    FN = con_matrix[1][0]
    TP = con_matrix[1][1]
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

def predict(pair, pro_seq, rna_seq):
    create_seq_file('./pro.txt', pair[0], pro_seq)
    create_seq_file('./rna.txt', pair[1], rna_seq)
    result = exec_cmd('./RNAScore pro.txt rna.txt')
    return float(result)

print("Procesing dataset %s" % (dataset))

# load data
print("Loading data...")
pos_pairs, neg_pairs = read_data_pair(data_path + dataset +'_pairs.txt')
pro_seq_dict = read_data_seq(data_path + 'sequence/' + dataset +'_protein_seq.fa')
rna_seq_dict = read_data_seq(data_path + 'sequence/' + dataset +'_rna_seq.fa')

print('Predicting...')
predictions = []
for pr in pos_pairs:
    result = predict(pr, pro_seq_dict[pr[0]], rna_seq_dict[pr[1]])
    print('{}\t{}\t{}'.format(pr, 1, result))
    predictions.append(result)
for pr in neg_pairs:
    result = predict(pr, pro_seq_dict[pr[0]], rna_seq_dict[pr[1]])
    print('{}\t{}\t{}'.format(pr, 0, result))
    predictions.append(result)

metrics = calc_metrics([1]*len(pos_pairs)+[0]*len(neg_pairs), predictions)
print(metrics)

with open('./%s_result.txt' % dataset, 'w') as r:
    r.write("# lncPro performances on %s (Acc, Sn, Sp, Pre, MCC, AUC):\n" % dataset)
    r.write(str(metrics)+'\n')
    r.write('\n\nDetailed prediction results (protein, RNA, label, prediction):\n')
    pos_num = len(pos_pairs)
    neg_num = len(neg_pairs)
    for i in range(pos_num):
        r.write('{}\t{}\t{}\t{}'.format(pr[0],pr[1],1,predictions[i]))
    for i in range(neg_num):
        r.write('{}\t{}\t{}\t{}'.format(pr[0],pr[1],1,predictions[i+pos_num]))