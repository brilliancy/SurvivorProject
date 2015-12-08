import pandas as pd
import numpy as np
import string
import nltk
import pdb
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer

from collections import Counter
from nltk.corpus import stopwords

def get_tokens():

	TESTFILE = '/Users/alanju/Documents/GADS_Project/Data_Preprocessing/DataFrames/MasterDf.csv'
	MasterDf = pd.read_csv(TESTFILE)
	MasterDf = MasterDf.drop(MasterDf.columns[0], axis=1) 

	MasterDf=MasterDf.dropna()
	MasterDf['subtitles_text'] = MasterDf.subtitles_text.apply(lambda k: k + ' ')
	text = ''.join(MasterDf.subtitles_text) 
	lowers = text.lower()

	no_punctuation = re.sub('[\W_]+', ' ', lowers)
	#no_punctuation = lowers.translate(None, string.punctuation)
	tokens = nltk.word_tokenize(no_punctuation)
	return tokens
tokens = get_tokens()
'''
fileObject = open("pickle_EDA",'wb') 
pickle.dump(tokens,fileObject)   
fileObject.close()
fileObject = open("pickle_EDA",'rb') #changed to rb
pickle_obj = pickle.load(fileObject)  
'''

filtered = [w for w in tokens if not w in stopwords.words('english')]
count = Counter(filtered)

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

stemmer = PorterStemmer()
stemmed = stem_tokens(filtered, stemmer)
count = Counter(stemmed)
print(count.most_common(100))


token_dict = {}
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed
def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems
pdb.set_trace() # figure how to tfidf
tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
tfs = tfidf.fit_transform(tokens)#changed to tokens, pretty sure i'm not doing this right
fileObject = open("pickle_EDA",'wb') 
pickle.dump(tfs,fileObject)   
fileObject.close()
