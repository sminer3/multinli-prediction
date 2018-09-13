from keras.activations import softmax
from keras.layers import *
from keras.optimizers import Adam
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint

#NN Functions
def create_pretrained_embedding(embedding_matrix):
    "Create embedding layer from a pretrained weights array"
    in_dim, out_dim = embedding_matrix.shape
    embedding = Embedding(in_dim, out_dim, weights=[embedding_matrix], trainable=False)
    return embedding


def unchanged_shape(input_shape):
    "Function for Lambda layer"
    return input_shape


def substract(input_1, input_2):
    "Substract element-wise"
    neg_input_2 = Lambda(lambda x: -x, output_shape=unchanged_shape)(input_2)
    out_ = Add()([input_1, neg_input_2])
    return out_


def submult(input_1, input_2):
    "Get multiplication and subtraction then concatenate results"
    mult = Multiply()([input_1, input_2])
    sub = substract(input_1, input_2)
    out_= Concatenate()([sub, mult])
    return out_


def apply_multiple(input_, layers):
    "Apply layers to input then concatenate result"
    if not len(layers) > 1:
        raise ValueError('Layers list should contain more than 1 layer')
    else:
        agg_ = []
        for layer in layers:
            agg_.append(layer(input_))
        out_ = Concatenate()(agg_)
    return out_


def time_distributed(input_, layers):
    "Apply a list of layers in TimeDistributed mode"
    out_ = []
    node_ = input_
    for layer_ in layers:
        node_ = TimeDistributed(layer_)(node_)
    out_ = node_
    return out_


def soft_attention_alignment(input_1, input_2):
    "Align text representation with neural soft attention"
    attention = Dot(axes=-1)([input_1, input_2])
    w_att_1 = Lambda(lambda x: softmax(x, axis=1),
                        output_shape=unchanged_shape)(attention)
    w_att_2 = Permute((2,1))(Lambda(lambda x: softmax(x, axis=2),
                                output_shape=unchanged_shape)(attention))
    in1_aligned = Dot(axes=1)([w_att_1, input_1])
    in2_aligned = Dot(axes=1)([w_att_2, input_2])
    return in1_aligned, in2_aligned


def decomposable_attention(embedding_matrix, projection_dim=300, projection_hidden=0, projection_dropout=0.2,
                           compare_dim=500, compare_dropout=0.2,
                           dense_dim=300, dense_dropout=0.2,
                           lr=1e-3, activation='elu', maxlen=15):
    # Based on: https://arxiv.org/abs/1606.01933
    
    s1 = Input(name='s1',shape=(maxlen,))
    s2 = Input(name='s2',shape=(maxlen,))
    
    # Embedding
    embedding = create_pretrained_embedding(embedding_matrix)
    s1_embed = embedding(s1)
    s2_embed = embedding(s2)
    
    # Projection
    projection_layers = []
    if projection_hidden > 0:
        projection_layers.extend([
                Dense(projection_hidden, activation=activation),
                Dropout(rate=projection_dropout),
            ])
    projection_layers.extend([
            Dense(projection_dim, activation=None),
            Dropout(rate=projection_dropout),
        ])
    s1_encoded = time_distributed(s1_embed, projection_layers)
    s2_encoded = time_distributed(s2_embed, projection_layers)
    
    # Attention
    s1_aligned, s2_aligned = soft_attention_alignment(s1_encoded, s2_encoded)    
    
    # Compare
    s1_combined = Concatenate()([s1_encoded, s2_aligned, submult(s1_encoded, s2_aligned)])	
    s2_combined = Concatenate()([s2_encoded, s1_aligned, submult(s2_encoded, s1_aligned)]) 
    compare_layers = [
        Dense(compare_dim, activation=activation),
        Dropout(compare_dropout),
        Dense(compare_dim, activation=activation),
        Dropout(compare_dropout),
    ]
    s1_compare = time_distributed(s1_combined, compare_layers)
    s2_compare = time_distributed(s2_combined, compare_layers)
    
    # Aggregate
    s1_rep = apply_multiple(s1_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])
    s2_rep = apply_multiple(s2_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])

    # Classifier
    merged = Concatenate()([s1_rep, s2_rep])
    dense = BatchNormalization()(merged)
    dense = Dense(dense_dim, activation=activation)(dense)
    dense = Dropout(dense_dropout)(dense)
    dense = BatchNormalization()(dense)
    dense = Dense(dense_dim, activation=activation)(dense)
    dense = Dropout(dense_dropout)(dense)
    out_ = Dense(3, activation='softmax')(dense)
    
    model = Model(inputs=[s1, s2], outputs=out_)
    model.compile(optimizer=Adam(lr=lr), loss='categorical_crossentropy', 
                  metrics=['categorical_crossentropy','accuracy'])
    return model


# def esim(embedding_matrix, maxlen=15, 
#          lstm_dim=300, 
#          dense_dim=300, 
#          dense_dropout=0.5):
             
#     # Based on arXiv:1609.06038
#     s1 = Input(name='s1',shape=(maxlen,))
#     s2 = Input(name='s2',shape=(maxlen,))
    
#     # Embedding
#     embedding = create_pretrained_embedding(embedding_matrix)
#     bn = BatchNormalization(axis=2)
#     s1_embed = bn(embedding(s1))
#     s2_embed = bn(embedding(s2))

#     # Encode
#     encode = Bidirectional(LSTM(lstm_dim, return_sequences=True))
#     s1_encoded = encode(s1_embed)
#     s2_encoded = encode(s2_embed)
    
#     # Attention
#     s1_aligned, s2_aligned = soft_attention_alignment(s1_encoded, s2_encoded)
    
#     # Compose
#     s1_combined = Concatenate()([s1_encoded, s2_aligned, submult(s1_encoded, s2_aligned)])
#     s2_combined = Concatenate()([s2_encoded, s1_aligned, submult(s2_encoded, s1_aligned)]) 
       
#     compose = Bidirectional(LSTM(lstm_dim, return_sequences=True))
#     s1_compare = compose(s1_combined)
#     s2_compare = compose(s2_combined)
    
#     # Aggregate
#     s1_rep = apply_multiple(s1_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])
#     s2_rep = apply_multiple(s2_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])
    
#     # Classifier
#     merged = Concatenate()([s1_rep, s2_rep])
    
#     dense = BatchNormalization()(merged)
#     dense = Dense(dense_dim, activation='elu')(dense)
#     dense = BatchNormalization()(dense)
#     dense = Dropout(dense_dropout)(dense)
#     dense = Dense(dense_dim, activation='elu')(dense)
#     dense = BatchNormalization()(dense)
#     dense = Dropout(dense_dropout)(dense)
#     out_ = Dense(3, activation='softmax')(dense)
    
#     model = Model(inputs=[s1, s2], outputs=out_)
#     model.compile(optimizer=Adam(lr=1e-3), loss='categorical_crossentropy', metrics=['categorical_crossentropy','accuracy'])
#     return model
