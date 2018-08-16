from sklearn.feature_extraction.text import CountVectorizer
import re

def preprocess(string):
    #Remove punctuation (except for apostrophe's) and make lower case
    string = string.lower()
    replace = str.maketrans('/\\"#$%&!()*+,-.:;<=>?@[]^_`{|}~', ' '*len('/\\"#$%&!()*+,-.:;<=>?@[]^_`{|}~'))
    string = string.translate(replace).strip()
    string = re.sub(' +',' ',string)
    #string = ' '.join([w for w in string.split() if w != 'a'])
    return string


def get_words(sentences,with_preprocessing=True):
	#Input: list of sentences
	#Output: tokens with counts
	sentences = set(sentences + list(map(preprocess, sentences)))
	vectorizer = CountVectorizer()
	vectorizer.fit(sentences)
	return list(vectorizer.vocabulary_.keys())