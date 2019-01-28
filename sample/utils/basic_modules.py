from keras.layers import Dense, Conv1D, concatenate, BatchNormalization, MaxPooling1D
from keras.layers import Dropout, Flatten, Input
from keras.models import Model


def conjoint_cnn(pro_coding_length, rna_coding_length, vector_repeatition_cnn):

    if type(vector_repeatition_cnn)==int:
        vec_len_p = vector_repeatition_cnn
        vec_len_r = vector_repeatition_cnn
    else:
        vec_len_p = vector_repeatition_cnn[0]
        vec_len_r = vector_repeatition_cnn[1]

    # NN for protein feature analysis by one hot encoding
    xp_in_conjoint_cnn = Input(shape=(pro_coding_length, vec_len_p))
    xp_cnn = Conv1D(filters=45, kernel_size=6, strides=1, activation='relu')(xp_in_conjoint_cnn)
    xp_cnn = MaxPooling1D(pool_size=2)(xp_cnn)
    xp_cnn = BatchNormalization()(xp_cnn)
    xp_cnn = Dropout(0.2)(xp_cnn)
    xp_cnn = Conv1D(filters=64, kernel_size=5, strides=1, activation='relu')(xp_cnn)
    xp_cnn = MaxPooling1D(pool_size=2)(xp_cnn)
    xp_cnn = BatchNormalization()(xp_cnn)
    xp_cnn = Dropout(0.2)(xp_cnn)
    xp_cnn = Conv1D(filters=45, kernel_size=4, strides=1, activation='relu')(xp_cnn)
    xp_cnn = BatchNormalization()(xp_cnn)
    xp_cnn = Dropout(0.2)(xp_cnn)
    # xp_cnn = Bidirectional(LSTM(32,return_sequences=True))(xp_cnn)
    # xp_cnn = LSTM(32,return_sequences=True)(xp_cnn)
    xp_cnn = Flatten()(xp_cnn)
    xp_out_conjoint_cnn = Dense(64)(xp_cnn)
    xp_out_conjoint_cnn = Dropout(0.2)(xp_out_conjoint_cnn)

    # NN for RNA feature analysis  by one hot encoding
    xr_in_conjoint_cnn = Input(shape=(rna_coding_length, vec_len_r))
    xr_cnn = Conv1D(filters=45, kernel_size=6, strides=1, activation='relu')(xr_in_conjoint_cnn)
    xr_cnn = MaxPooling1D(pool_size=2)(xr_cnn)
    xr_cnn = BatchNormalization()(xr_cnn)
    xr_cnn = Dropout(0.2)(xr_cnn)
    xr_cnn = Conv1D(filters=64, kernel_size=5, strides=1, activation='relu')(xr_cnn)
    xr_cnn = MaxPooling1D(pool_size=2)(xr_cnn)
    xr_cnn = BatchNormalization()(xr_cnn)
    xr_cnn = Dropout(0.2)(xr_cnn)
    xr_cnn = Conv1D(filters=45, kernel_size=4, strides=1, activation='relu')(xr_cnn)
    xr_cnn = BatchNormalization()(xr_cnn)
    xr_cnn = Dropout(0.2)(xr_cnn)
    # xr_cnn = Bidirectional(LSTM(32,return_sequences=True))(xr_cnn)
    # xr_cnn = LSTM(32,return_sequences=True)(xr_cnn)
    xr_cnn = Flatten()(xr_cnn)
    xr_out_conjoint_cnn = Dense(64)(xr_cnn)
    xr_out_conjoint_cnn = Dropout(0.2)(xr_out_conjoint_cnn)

    x_out_conjoint_cnn = concatenate([xp_out_conjoint_cnn, xr_out_conjoint_cnn])
    x_out_conjoint_cnn = Dense(128, kernel_initializer='random_uniform', activation='relu')(x_out_conjoint_cnn)
    # x_out_conjoint_cnn = Dropout(0.25)(x_out_conjoint_cnn)
    x_out_conjoint_cnn = BatchNormalization()(x_out_conjoint_cnn)
    # x_out_conjoint_cnn = Dropout(0.3)(x_out_conjoint_cnn)
    x_out_conjoint_cnn = Dense(64, kernel_initializer='random_uniform', activation='relu')(x_out_conjoint_cnn)
    y_conjoint_cnn = Dense(2, activation='softmax')(x_out_conjoint_cnn)

    model_conjoint_cnn = Model(inputs=[xp_in_conjoint_cnn, xr_in_conjoint_cnn], outputs=y_conjoint_cnn)



    return model_conjoint_cnn


def conjoint_sae(encoders_protein, encoders_rna, pro_coding_length, rna_coding_length):

    # NN for protein feature analysis
    xp_in_conjoint = Input(shape=(pro_coding_length,))
    xp_encoded = encoders_protein[0](xp_in_conjoint)
    xp_encoded = Dropout(0.2)(xp_encoded)
    xp_encoded = encoders_protein[1](xp_encoded)
    xp_encoded = Dropout(0.2)(xp_encoded)
    xp_encoder = encoders_protein[2](xp_encoded)
    xp_encoder = Dropout(0.2)(xp_encoder)
    xp_encoder = BatchNormalization()(xp_encoder)
    # xp_encoder = PReLU()(xp_encoder)
    xp_encoder = Dropout(0.2)(xp_encoder)

    # NN for RNA feature analysis
    xr_in_conjoint = Input(shape=(rna_coding_length,))
    xr_encoded = encoders_rna[0](xr_in_conjoint)
    xr_encoded = Dropout(0.2)(xr_encoded)
    xr_encoded = encoders_rna[1](xr_encoded)
    xr_encoded = Dropout(0.2)(xr_encoded)
    xr_encoded = encoders_rna[2](xr_encoded)
    xr_encoder = Dropout(0.2)(xr_encoded)
    xr_encoder = BatchNormalization()(xr_encoder)
    # xr_encoder = PReLU()(xr_encoder)
    xr_encoder = Dropout(0.2)(xr_encoder)

    x_out_conjoint = concatenate([xp_encoder, xr_encoder])
    x_out_conjoint = Dense(128, kernel_initializer='random_uniform', activation='relu')(x_out_conjoint)
    x_out_conjoint = BatchNormalization()(x_out_conjoint)
    # x_out_conjoint = Dropout(0.3)(x_out_conjoint)
    # x_out_conjoint = Dense(128, activation='relu')(x_out_conjoint)
    # x_out_conjoint = BatchNormalization()(x_out_conjoint)
    # x_out_conjoint = Dropout(0.2)(x_out_conjoint)
    # x_out_conjoint = Dense(64, activation='relu')(x_out_conjoint)
    # x_out_conjoint = BatchNormalization()(x_out_conjoint)
    # x_out_conjoint = Dropout(0.2)(x_out_conjoint)
    x_out_conjoint = Dense(64, kernel_initializer='random_uniform', activation='relu')(x_out_conjoint)
    y_conjoint = Dense(2, activation='softmax')(x_out_conjoint)

    model_conjoint = Model(inputs=[xp_in_conjoint, xr_in_conjoint], outputs=y_conjoint)


    return model_conjoint