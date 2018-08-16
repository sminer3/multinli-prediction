print("Loading libraries")
import os
import sys
#os.chdir(os.getcwd() + '/source')
sys.path.insert(0, '../scripts/')
import functions as fn
import pandas as pd
import pickle


print("Loading Data")
with open('../data/embeddings.p', 'rb') as fp:
    embeddings = pickle.load(fp)
train = pd.read_csv('../data/multinli_train.csv')

print("Preprocessing")
train['sentence1'] = train.sentence1.apply(str).apply(fn.preprocess)
train['sentence2'] = train.sentence1.apply(str).apply(fn.preprocess)

