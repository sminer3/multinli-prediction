print("Loading libraries")
import os
import sys
#os.chdir(os.getcwd() + '/source')
sys.path.insert(0, '../scripts/')
import functions as fn
import nn_functions as nn
import pandas as pd
import numpy as np
import pickle
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder	
from sklearn.metrics import roc_auc_score, log_loss

#Hyperparameters
MAX_SEQUENCE_LENGTH = 30
BATCH_SIZE = 256
n_randomized_vectors = 100
randomized_vectors_mean = 0
randomized_vectors_std = .18
projection_dim = 300
projection_hidden = 0
projection_dropout=0.2
compare_dim=500 
compare_dropout=0.2
dense_dim=300
dense_dropout=0.2
lr=1e-3 
activation='elu' 

print("Loading Data")
with open('../data/embeddings.p', 'rb') as fp:
    embeddings = pickle.load(fp)
train = pd.read_csv('../data/multinli_train.csv')

print("Preprocessing")
train['sentence1'] = train.sentence1.apply(str).apply(fn.preprocess)
train['sentence2'] = train.sentence2.apply(str).apply(fn.preprocess)

print("Tokenizing")
tokenizer = Tokenizer(filters="")
tokenizer.fit_on_texts(np.append(train.sentence1, train.sentence2))
word_index = tokenizer.word_index

#To deal with words not found, we try including randomized vectors that will be used in the embedding section
words_not_found = [w for w in word_index.keys() if w not in embeddings.keys()]
#print(words_not_found[0:100])
random_vectors = np.random.normal(loc = randomized_vectors_mean, scale = randomized_vectors_std, size = (n_randomized_vectors, 300))

print("Padding Sequences")
s1 = pad_sequences(tokenizer.texts_to_sequences(train.sentence1), maxlen=MAX_SEQUENCE_LENGTH)
s2 = pad_sequences(tokenizer.texts_to_sequences(train.sentence2), maxlen=MAX_SEQUENCE_LENGTH)

print("Creating word embedding matrix, encoding labels and splitting data")
nb_words = len(word_index) + 1
embedding_matrix = np.zeros((nb_words, 300))

for word, i in word_index.items():
    if word in embeddings.keys():
        embedding_matrix[i] = embeddings[word]
    else:
        embedding_matrix[i] = random_vectors[np.random.randint(n_randomized_vectors-1),]

#One hot encode labels
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(train.gold_label).reshape(-1,1)
one_hot_encoder = OneHotEncoder(sparse=False)
label = one_hot_encoder.fit_transform(integer_encoded)

#Split data
s1_train, s1_test, s2_train, s2_test, label_train, label_test = train_test_split(s1,s2,label, test_size = 20000, stratify = label, random_state=1)
s1_dev, s1_test, s2_dev, s2_test, label_dev, label_test = train_test_split(s1_test, s2_test, label_test, test_size = .5, stratify = label_test, random_state = 1)
training = [s1_train, s2_train]
dev = [s1_dev, s2_dev]
testing = [s1_test, s2_test]

model = nn.decomposable_attention(embedding_matrix, projection_dim, projection_hidden, projection_dropout,compare_dim, compare_dropout,dense_dim, dense_dropout, lr, activation, MAX_SEQUENCE_LENGTH)
model.load_weights('NN_test_model.h5')
model.predict([s1[1:10],s2[1:10]])
model.evaluate(x=testing, y=label_test)


early_stopping = EarlyStopping(monitor="val_loss", patience=5)
model_checkpoint = ModelCheckpoint('NN_test_model.h5',save_best_only=True,save_weights_only=True) #, questions_train, f_train , questions_val, f_val
hist = model.fit(training, label_train, validation_data=(dev, label_dev), 
                        epochs=18, batch_size=BATCH_SIZE, shuffle=True,
                        callbacks=[early_stopping, model_checkpoint], verbose=1)
print("validation loss:", min(hist.history["val_loss"]))\

model.load_weights('NN_test_model.h5')
preds = model.predict(testing, batch_size=BATCH_SIZE, verbose=2)
test_loss = log_loss(label_test,preds)
print("Test Loss:",test_loss)


#To Do:
	#Remove stop words?
	#Figure out why google prepended a NULL token to each sentence
	#Add helper functions for Neural networks
	#Vocabulary should only come from training data?
	#Come up with a list of hyperparameters to tune
	#Consider using efficient batching like Google did
	#Save results with hyperparameters, architecture, and test metrics
	#Save predictions to conduct error analysis

#Embeddings
	#Google did a number of things including:
	#normalizing each embedding to have l2 norm of 1 and projected down to less dimensions (hyperparameter)
	#OOV words hashed to 1 of one hundred random embeddings of mean 0 and sd 1.
	#projection matrix was trained


