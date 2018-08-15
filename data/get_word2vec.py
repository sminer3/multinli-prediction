import gensim
import pandas as pd

path_to_google_word2vec = 'C:/Users/sminer/Downloads/GoogleNews-vectors-negative300.bin.gz'
model = gensim.models.KeyedVectors.load_word2vec_format(path_to_google_word2vec, binary=True)

dat = pd.read_csv('multinli_train.csv')

