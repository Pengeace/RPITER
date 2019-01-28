import math
from functools import reduce

import numpy as np


# encoder for protein sequence
class ProEncoder:
    elements = 'AIYHRDC'
    structs = 'hec'

    element_number = 7
    # number of structure kind
    struct_kind = 3

    # clusters: {A,G,V}, {I,L,F,P}, {Y,M,T,S}, {H,N,Q,W}, {R,K}, {D,E}, {C}
    pro_intab = 'AGVILFPYMTSHNQWRKDEC'
    pro_outtab = 'AAAIIIIYYYYHHHHRRDDC'

    def __init__(self, WINDOW_P_UPLIMIT, WINDOW_P_STRUCT_UPLIMIT, CODING_FREQUENCY, VECTOR_REPETITION_CNN,
                 TRUNCATION_LEN=None, PERIOD_EXTENDED=None):

        self.WINDOW_P_UPLIMIT = WINDOW_P_UPLIMIT
        self.WINDOW_P_STRUCT_UPLIMIT = WINDOW_P_STRUCT_UPLIMIT
        self.CODING_FREQUENCY = CODING_FREQUENCY
        self.VECTOR_REPETITION_CNN = VECTOR_REPETITION_CNN

        self.TRUNCATION_LEN = TRUNCATION_LEN
        self.PERIOD_EXTENDED = PERIOD_EXTENDED

        # list and position map for k_mer
        k_mers = ['']
        self.k_mer_list = []
        self.k_mer_map = {}
        for T in range(self.WINDOW_P_UPLIMIT):
            temp_list = []
            for k_mer in k_mers:
                for x in self.elements:
                    temp_list.append(k_mer + x)
            k_mers = temp_list
            self.k_mer_list += temp_list
        for i in range(len(self.k_mer_list)):
            self.k_mer_map[self.k_mer_list[i]] = i

        # list and position map for k_mer with structure
        k_mers = ['']
        self.k_mer_struct_list = []
        self.k_mer_struct_map = {}
        for T in range(self.WINDOW_P_STRUCT_UPLIMIT):
            temp_list = []
            for k_mer in k_mers:
                for x in self.elements:
                    for s in self.structs:
                        temp_list.append(k_mer[0:T] + x + k_mer[T:2 * T] + s)
            k_mers = temp_list
            self.k_mer_struct_list += temp_list
        for i in range(len(self.k_mer_struct_list)):
            self.k_mer_struct_map[self.k_mer_struct_list[i]] = i

        # table for amino acid clusters
        self.transtable = str.maketrans(self.pro_intab, self.pro_outtab)

        # print(len(self.k_mer_list))
        # print(self.k_mer_list)
        # print(len(self.k_mer_struct_list))
        # print(self.k_mer_struct_list)

    def encode_conjoint_previous(self, seq):
        seq = seq.translate(self.transtable)
        seq = ''.join([x for x in seq if x in self.elements])
        seq_len = len(seq)
        if seq_len == 0:
            return 'Error'
        result = []
        K = self.WINDOW_P_UPLIMIT
        offset = reduce(lambda x, y: x + y, map(lambda x: self.element_number ** x, range(1, K)))
        vec = [0.0] * (self.element_number ** K)
        counter = seq_len - K + 1
        for i in range(seq_len - K + 1):
            k_mer = seq[i:i + K]
            vec[self.k_mer_map[k_mer] - offset] += 1
        vec = np.array(vec)
        if self.CODING_FREQUENCY:
            vec = vec / vec.max()
        result += list(vec)
        return np.array(result)

    def encode_conjoint(self, seq):
        seq = seq.translate(self.transtable)
        seq = ''.join([x for x in seq if x in self.elements])
        seq_len = len(seq)
        if seq_len == 0:
            return 'Error'
        result = []
        offset = 0
        for K in range(1, self.WINDOW_P_UPLIMIT + 1):
            vec = [0.0] * (self.element_number ** K)
            counter = seq_len - K + 1
            for i in range(seq_len - K + 1):
                k_mer = seq[i:i + K]
                vec[self.k_mer_map[k_mer] - offset] += 1
            vec = np.array(vec)
            offset += vec.size
            if self.CODING_FREQUENCY:
                vec = vec / vec.max()
            result += list(vec)
        return np.array(result)

    def encode_conjoint_struct(self, seq, struct):
        seq = seq.translate(self.transtable)
        seq_temp = []
        struct_temp = []
        for i in range(len(seq)):
            if seq[i] in self.elements:
                seq_temp.append(seq[i])
                struct_temp.append(struct[i])
        seq = ''.join(seq_temp)
        struct = ''.join(struct_temp)
        seq_len = len(seq)
        if seq_len == 0:
            return 'Error'
        result = []
        offset = 0
        for K in range(1, self.WINDOW_P_STRUCT_UPLIMIT + 1):
            vec = [0.0] * ((self.element_number * self.struct_kind) ** K)
            counter = seq_len - K + 1
            for i in range(seq_len - K + 1):
                k_mer = seq[i:i + K] + struct[i:i + K]
                vec[self.k_mer_struct_map[k_mer] - offset] += 1
            vec = np.array(vec)
            offset += vec.size
            if self.CODING_FREQUENCY:
                vec = vec / vec.max()
            result += list(vec)
        return np.array(result)

    def encode_conjoint_cnn(self, seq):
        seq = seq.translate(self.transtable)
        seq = ''.join([x for x in seq if x in self.elements])
        seq_len = len(seq)
        if seq_len == 0:
            return 'Error'
        result_t = []
        offset = 0
        for K in range(1, self.WINDOW_P_UPLIMIT + 1):
            vec = [0.0] * (self.element_number ** K)
            counter = seq_len - K + 1
            for i in range(seq_len - K + 1):
                k_mer = seq[i:i + K]
                vec[self.k_mer_map[k_mer] - offset] += 1
            vec = np.array(vec)
            offset += vec.size
            if self.CODING_FREQUENCY:
                vec = vec / vec.max()
            result_t += list(vec)
        result = np.array([[x] * self.VECTOR_REPETITION_CNN for x in result_t])
        return result

    def encode_conjoint_struct_cnn(self, seq, struct):
        seq = seq.translate(self.transtable)
        seq_temp = []
        struct_temp = []
        for i in range(len(seq)):
            if seq[i] in self.elements:
                seq_temp.append(seq[i])
                struct_temp.append(struct[i])
        seq = ''.join(seq_temp)
        struct = ''.join(struct_temp)
        seq_len = len(seq)
        if seq_len == 0:
            return 'Error'
        result_t = []
        offset = 0
        for K in range(1, self.WINDOW_P_STRUCT_UPLIMIT + 1):
            vec = [0.0] * ((self.element_number * self.struct_kind) ** K)
            counter = seq_len - K + 1
            for i in range(seq_len - K + 1):
                k_mer = seq[i:i + K] + struct[i:i + K]
                vec[self.k_mer_struct_map[k_mer] - offset] += 1
            vec = np.array(vec)
            offset += vec.size
            if self.CODING_FREQUENCY:
                vec = vec / vec.max()
            result_t += list(vec)
        result = np.array([[x] * self.VECTOR_REPETITION_CNN for x in result_t])
        return result

    def encode_onehot(self, seq):
        seq = seq.translate(self.transtable)
        seq = ''.join([x for x in seq if x in self.elements])
        seq_len = len(seq)
        if seq_len == 0:
            return 'Error'
        coding_len = min(len(seq), self.TRUNCATION_LEN)
        vec = [0] * coding_len
        for i in range(coding_len):
            pos = self.elements.index(seq[i])
            vec[i] = [0] * self.element_number
            vec[i][pos] = 1
        if coding_len < self.TRUNCATION_LEN:
            gap_len = (self.TRUNCATION_LEN - coding_len)
            if self.PERIOD_EXTENDED:
                gap_term = math.ceil(gap_len / coding_len)
                gap_data = vec * gap_term
                vec += gap_data[0:gap_len]
            else:
                vec += [[0] * self.element_number] * gap_len
        return np.array(vec)

    def encode_onehot_struct(self, seq, struct):
        seq = seq.translate(self.transtable)
        seq_temp = []
        struct_temp = []
        for i in range(len(seq)):
            if seq[i] in self.elements:
                seq_temp.append(seq[i])
                struct_temp.append(struct[i])
        seq = ''.join(seq_temp)
        struct = ''.join(struct_temp)
        seq_len = len(seq)
        if seq_len == 0:
            return 'Error'
        coding_len = min(len(seq), self.TRUNCATION_LEN)
        vec = [0] * coding_len
        for i in range(coding_len):
            pos = self.elements.index(seq[i])
            pos = pos + self.element_number * self.structs.index(struct[i])
            vec[i] = [0] * (self.element_number * self.struct_kind)
            vec[i][pos] = 1
        if coding_len < self.TRUNCATION_LEN:
            gap_len = (self.TRUNCATION_LEN - coding_len)
            if self.PERIOD_EXTENDED:
                gap_term = math.ceil(gap_len / coding_len)
                gap_data = vec * gap_term
                vec += gap_data[0:gap_len]
            else:
                vec += [[0] * self.element_number * self.struct_kind] * gap_len
        return np.array(vec)

    def encode_word2vec(self, seq, pro_word2vec, window_size, stride):
        seq = seq.translate(self.transtable)
        seq_len = len(seq)
        if seq_len == 0:
            return 'Error'
        words = pro_word2vec.keys()
        VEC_LEN_W2V = len(list(words)[0])
        MAX_PRO_W2V_LEN = self.TRUNCATION_LEN // stride
        vec = []
        p = 0
        while (len(vec) < MAX_PRO_W2V_LEN) and (p + window_size <= seq_len):
            word = seq[p:p + window_size]
            if word in words:
                vec.append(pro_word2vec[word])
            p += stride
        encoded_len = len(vec)
        if encoded_len == 0:
            return 'Error'
        elif encoded_len < MAX_PRO_W2V_LEN:
            gap_len = (MAX_PRO_W2V_LEN - encoded_len)
            if self.PERIOD_EXTENDED:
                gap_term = math.ceil(gap_len / encoded_len)
                gap_data = vec * gap_term
                vec += gap_data[0:gap_len]
            else:
                vec += [[0.0] * VEC_LEN_W2V] * gap_len
        return np.array(vec)

    def encode_word2vec_struct(self, seq, struct, pro_word2vec, window_size, stride):
        seq = seq.translate(self.transtable)
        seq_temp = []
        struct_temp = []
        for i in range(len(seq)):
            if seq[i] in self.elements:
                seq_temp.append(seq[i])
                struct_temp.append(struct[i])
        seq = ''.join(seq_temp)
        struct = ''.join(struct_temp)
        seq_len = len(seq)
        if seq_len == 0:
            return 'Error'

        words = pro_word2vec.keys()
        VEC_LEN_W2V = len(list(words)[0])
        MAX_PRO_W2V_LEN = self.TRUNCATION_LEN // stride
        vec = []
        p = 0
        while (len(vec) < MAX_PRO_W2V_LEN) and (p + window_size <= seq_len):
            word = seq[p:p + window_size] + struct[p:p + window_size]
            if word in words:
                vec.append(pro_word2vec[word])
            p += stride
        encoded_len = len(vec)
        if encoded_len == 0:
            return 'Error'
        elif encoded_len < MAX_PRO_W2V_LEN:
            gap_len = (MAX_PRO_W2V_LEN - encoded_len)
            if self.PERIOD_EXTENDED:
                gap_term = math.ceil(gap_len / encoded_len)
                gap_data = vec * gap_term
                vec += gap_data[0:gap_len]
            else:
                vec += [[0.0] * VEC_LEN_W2V] * gap_len
        return np.array(vec)


# encoder for RNA sequence
class RNAEncoder:
    elements = 'AUCG'
    structs = '.('

    element_number = 4
    struct_kind = 2

    def __init__(self, WINDOW_R_UPLIMIT, WINDOW_R_STRUCT_UPLIMIT, CODING_FREQUENCY, VECTOR_REPETITION_CNN,
                 TRUNCATION_LEN=None, PERIOD_EXTENDED=None):

        self.WINDOW_R_UPLIMIT = WINDOW_R_UPLIMIT
        self.WINDOW_R_STRUCT_UPLIMIT = WINDOW_R_STRUCT_UPLIMIT
        self.CODING_FREQUENCY = CODING_FREQUENCY
        self.VECTOR_REPETITION_CNN = VECTOR_REPETITION_CNN

        self.TRUNCATION_LEN = TRUNCATION_LEN
        self.PERIOD_EXTENDED = PERIOD_EXTENDED

        k_mers = ['']
        self.k_mer_list = []
        self.k_mer_map = {}
        for T in range(self.WINDOW_R_UPLIMIT):
            temp_list = []
            for k_mer in k_mers:
                for x in self.elements:
                    temp_list.append(k_mer + x)
            k_mers = temp_list
            self.k_mer_list += temp_list
        for i in range(len(self.k_mer_list)):
            self.k_mer_map[self.k_mer_list[i]] = i

        k_mers = ['']
        self.k_mer_struct_list = []
        self.k_mer_struct_map = {}
        for T in range(self.WINDOW_R_STRUCT_UPLIMIT):
            temp_list = []
            for k_mer in k_mers:
                for x in self.elements:
                    for s in self.structs:
                        temp_list.append(k_mer[0:T] + x + k_mer[T:2 * T] + s)
            k_mers = temp_list
            self.k_mer_struct_list += temp_list
        for i in range(len(self.k_mer_struct_list)):
            self.k_mer_struct_map[self.k_mer_struct_list[i]] = i

            # print(len(self.k_mer_list))
            # print(self.k_mer_list)
            # print(len(self.k_mer_struct_list))
            # print(self.k_mer_struct_list)

    def encode_conjoint_previous(self, seq):
        seq = seq.replace('T', 'U')
        seq = ''.join([x for x in seq if x in self.elements])
        seq_len = len(seq)
        if seq_len == 0:
            return 'Error'
        result = []
        K = self.WINDOW_R_UPLIMIT
        offset = reduce(lambda x, y: x + y, map(lambda x: self.element_number ** x, range(1, K)))
        vec = [0.0] * (self.element_number ** K)
        counter = seq_len - K + 1
        for i in range(seq_len - K + 1):
            k_mer = seq[i:i + K]
            vec[self.k_mer_map[k_mer] - offset] += 1
        vec = np.array(vec)
        if self.CODING_FREQUENCY:
            vec = vec / vec.max()
        result += list(vec)
        return np.array(result)

    def encode_conjoint(self, seq):
        seq = seq.replace('T', 'U')
        seq = ''.join([x for x in seq if x in self.elements])
        seq_len = len(seq)
        if seq_len == 0:
            return 'Error'
        result = []
        offset = 0
        for K in range(1, self.WINDOW_R_UPLIMIT + 1):
            vec = [0.0] * (self.element_number ** K)
            counter = seq_len - K + 1
            for i in range(seq_len - K + 1):
                k_mer = seq[i:i + K]
                vec[self.k_mer_map[k_mer] - offset] += 1
            vec = np.array(vec)
            offset += vec.size
            if self.CODING_FREQUENCY:
                vec = vec / vec.max()
            result += list(vec)
        return np.array(result)

    def encode_conjoint_struct(self, seq, struct):
        seq = seq.replace('T', 'U')
        struct = struct.replace(')', '(')
        seq_temp = []
        struct_temp = []
        for i in range(len(seq)):
            if seq[i] in self.elements:
                seq_temp.append(seq[i])
                struct_temp.append(struct[i])
        seq = ''.join(seq_temp)
        struct = ''.join(struct_temp)
        seq_len = len(seq)
        if seq_len == 0:
            return 'Error'
        result = []
        offset = 0
        for K in range(1, self.WINDOW_R_STRUCT_UPLIMIT + 1):
            vec = [0.0] * ((self.element_number * self.struct_kind) ** K)
            counter = seq_len - K + 1
            for i in range(seq_len - K + 1):
                k_mer = seq[i:i + K] + struct[i:i + K]
                vec[self.k_mer_struct_map[k_mer] - offset] += 1
            vec = np.array(vec)
            offset += vec.size
            if self.CODING_FREQUENCY:
                vec = vec / vec.max()
            result += list(vec)
        return np.array(result)

    def encode_conjoint_cnn(self, seq):
        seq = seq.replace('T', 'U')
        seq = ''.join([x for x in seq if x in self.elements])
        seq_len = len(seq)
        if seq_len == 0:
            return 'Error'
        result_t = []
        offset = 0
        for K in range(1, self.WINDOW_R_UPLIMIT + 1):
            vec = [0.0] * (self.element_number ** K)
            counter = seq_len - K + 1
            for i in range(seq_len - K + 1):
                k_mer = seq[i:i + K]
                vec[self.k_mer_map[k_mer] - offset] += 1
            vec = np.array(vec)
            offset += vec.size
            if self.CODING_FREQUENCY:
                vec = vec / vec.max()
            result_t += list(vec)
        result = np.array([[x] * self.VECTOR_REPETITION_CNN for x in result_t])
        return result

    def encode_conjoint_struct_cnn(self, seq, struct):
        seq = seq.replace('T', 'U')
        struct = struct.replace(')', '(')
        seq_temp = []
        struct_temp = []
        for i in range(len(seq)):
            if seq[i] in self.elements:
                seq_temp.append(seq[i])
                struct_temp.append(struct[i])
        seq = ''.join(seq_temp)
        struct = ''.join(struct_temp)
        seq_len = len(seq)
        if seq_len == 0:
            return 'Error'
        result_t = []
        offset = 0
        for K in range(1, self.WINDOW_R_STRUCT_UPLIMIT + 1):
            vec = [0.0] * ((self.element_number * self.struct_kind) ** K)
            counter = seq_len - K + 1
            for i in range(seq_len - K + 1):
                k_mer = seq[i:i + K] + struct[i:i + K]
                vec[self.k_mer_struct_map[k_mer] - offset] += 1
            vec = np.array(vec)
            offset += vec.size
            if self.CODING_FREQUENCY:
                vec = vec / vec.max()
            result_t += list(vec)
        result = np.array([[x] * self.VECTOR_REPETITION_CNN for x in result_t])
        return result

    def encode_onehot(self, seq):
        seq = seq.replace('T', 'U')
        seq = ''.join([x for x in seq if x in self.elements])
        seq_len = len(seq)
        if seq_len == 0:
            return 'Error'
        coding_len = min(len(seq), self.TRUNCATION_LEN)
        vec = [0] * coding_len
        for i in range(coding_len):
            pos = self.elements.index(seq[i])
            vec[i] = [0] * self.element_number
            vec[i][pos] = 1
        if coding_len < self.TRUNCATION_LEN:
            gap_len = (self.TRUNCATION_LEN - coding_len)
            if self.PERIOD_EXTENDED:
                gap_term = math.ceil(gap_len / coding_len)
                gap_data = vec * gap_term
                vec += gap_data[0:gap_len]
            else:
                vec += [[0] * self.element_number] * gap_len
        return np.array(vec)

    def encode_onehot_struct(self, seq, struct):
        seq = seq.replace('T', 'U')
        struct = struct.replace(')', '(')
        seq_temp = []
        struct_temp = []
        for i in range(len(seq)):
            if seq[i] in self.elements:
                seq_temp.append(seq[i])
                struct_temp.append(struct[i])
        seq = ''.join(seq_temp)
        struct = ''.join(struct_temp)
        seq_len = len(seq)
        if seq_len == 0:
            return 'Error'
        coding_len = min(len(seq), self.TRUNCATION_LEN)
        vec = [0] * coding_len
        for i in range(coding_len):
            pos = self.elements.index(seq[i])
            pos = pos + self.element_number * self.structs.index(struct[i])
            vec[i] = [0] * (self.element_number * self.struct_kind)
            vec[i][pos] = 1
        if coding_len < self.TRUNCATION_LEN:
            gap_len = (self.TRUNCATION_LEN - coding_len)
            if self.PERIOD_EXTENDED:
                gap_term = math.ceil(gap_len / coding_len)
                gap_data = vec * gap_term
                vec += gap_data[0:gap_len]
            else:
                vec += [[0] * self.element_number * self.struct_kind] * gap_len
        return np.array(vec)

    def encode_word2vec(self, seq, rna_word2vec, window_size, stride):
        seq = seq.replace('T', 'U')
        seq = ''.join([x for x in seq if x in self.elements])
        seq_len = len(seq)
        if seq_len == 0:
            return 'Error'
        words = rna_word2vec.keys()
        VEC_LEN_W2V = len(list(words)[0])
        MAX_RNA_W2V_LEN = self.TRUNCATION_LEN // stride
        vec = []
        p = 0
        while (len(vec) < MAX_RNA_W2V_LEN) and (p + window_size <= seq_len):
            word = seq[p:p + window_size]
            if word in words:
                vec.append(rna_word2vec[word])
            p += stride
        encoded_len = len(vec)
        if encoded_len == 0:
            return 'Error'
        elif encoded_len < MAX_RNA_W2V_LEN:
            gap_len = (MAX_RNA_W2V_LEN - encoded_len)
            if self.PERIOD_EXTENDED:
                gap_term = math.ceil(gap_len / encoded_len)
                gap_data = vec * gap_term
                vec += gap_data[0:gap_len]
            else:
                vec += [[0.0] * VEC_LEN_W2V] * gap_len
        return np.array(vec)

    def encode_word2vec_struct(self, seq, struct, rna_word2vec, window_size, stride):
        seq = seq.replace('T', 'U')
        struct = struct.replace(')', '(')
        seq_temp = []
        struct_temp = []
        for i in range(len(seq)):
            if seq[i] in self.elements:
                seq_temp.append(seq[i])
                struct_temp.append(struct[i])
        seq = ''.join(seq_temp)
        struct = ''.join(struct_temp)
        seq_len = len(seq)
        if seq_len == 0:
            return 'Error'

        words = rna_word2vec.keys()
        VEC_LEN_W2V = len(list(words)[0])
        MAX_RNA_W2V_LEN = self.TRUNCATION_LEN // stride
        vec = []
        p = 0
        while (len(vec) < MAX_RNA_W2V_LEN) and (p + window_size <= seq_len):
            word = seq[p:p + window_size] + struct[p:p + window_size]
            if word in words:
                vec.append(rna_word2vec[word])
            p += stride
        encoded_len = len(vec)
        if encoded_len == 0:
            return 'Error'
        elif encoded_len < MAX_RNA_W2V_LEN:
            gap_len = (MAX_RNA_W2V_LEN - encoded_len)
            if self.PERIOD_EXTENDED:
                gap_term = math.ceil(gap_len / encoded_len)
                gap_data = vec * gap_term
                vec += gap_data[0:gap_len]
            else:
                vec += [[0.0] * VEC_LEN_W2V] * gap_len
        return np.array(vec)
