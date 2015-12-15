import pdb
import pandas as pd
import pickle
import re
import os
import datetime

#token = word
#ngram =character or word
def main():
	'''
	Before Model:
	1. Clean up dataset to fit this format:
		SentenceId,EpisodeId,Season,Episode,begin,endSentence
		https://raw.githubusercontent.com/mneedham/neo4j-himym/master/data/import/sentences.csv
		******#PROBLEM: makes an assumption that the dictionary is in order, however it isn't****
		
	2. Have a list of labels: who was voted out each episode

	During Model:
	1. Feed in list of names of survivor contestents
	2. For each episode look at the ngrams around each 'name'
		2.1
	3. gather the sentiment of positive and negative words around the 'name'
		3.1 I could hand code it and look up the word and its corresponding positive or negative value
			then average the values

	4. can cross val by taking out some episodes
	
	Ways of improving Model(scores):
	5. Look at high information features instead of using all the words as features
		http://streamhacker.com/2010/06/16/text-classification-sentiment-analysis-eliminate-low-information-features/

	****random forest
	dont need to do sentiment scores since it looks at the words itself
	could build 3 models: random forest,bayes
	  * with sentiment scores
	  * without sentiment scores
	  * with sentiment scores and looking at only EXTREME sentiment

	base classifiers n_estimators = several hundred up to 1000
	njobs =-1
	records = episode
	features = ngrams start with ngrams =5 then move up
	target variable = bucket their where they placed

	Current job list:
	1.convert begin and end columns into date time 
	1.1 subset 00:35:55,00:36:14
	1.1 subest 00:28:34,00:34:50
	2.look up when people are planning the voting phase, subset based on that phase
	3. turn target variable into binary
	4. look at ngrams per person for an episode (NLTK/sklearnlade)
	'''
	def cleaning_MasterDf():
		masterDf = pd.read_csv("Data_Preprocessing/DataFrames/MasterDf.csv")
		return(masterDf)
	#script_dir = os.path.dirname(__file__)
	#pdb.set_trace()

	def cleaning_labelDf():
		
		fileObject = open("Data_Preprocessing/cleaned_label_soup",'rb') #changed to rb
		pickle_obj = pickle.load(fileObject)
		
		clean_pickle = pickle_obj.replace('Runner-UP','2',regex=True)
		clean_pickle = clean_pickle.replace('Winner','1',regex=True)
		clean_pickle = clean_pickle.replace('2nd','2',regex=True)

		return(clean_pickle)
	masterDf = cleaning_MasterDf()	
	labelDf = cleaning_labelDf()

	def sub_datetime(k):
		#sub_time = re.split(':|\.',k)
		#sub_time = map(int, sub_time)
	
		#time = datetime.datetime(1,1,1,*map(int, sub_time))
		#date_object = datetime.strptime("00:00:03.73"', '%02d:%02d:%02d%s')
		#format = "%H:%M:%S.%f"
		format = "%H:%M:%S"
		time = datetime.datetime.strptime(k[:8],format)
		#return('{0:0>2}:{1:0>2}:{2:0>2}'.format(time.hour,time.minute,time.second))
		return time
	#masterDf.end = masterDf.end.apply(lambda k:sub_datetime(re.split(':|\.',k)))
	masterDf.end = masterDf.end.apply(sub_datetime)
	masterDf.begin = masterDf.begin.apply(sub_datetime)

	subset_1 = masterDf[masterDf.begin > sub_datetime("00:28:34")]
	subset_2 = subset_1[subset_1.begin < sub_datetime("00:36:14")]
	#can't do this apparently but i can just subset it twice
	#masterDf[masterDf.begin >= sub_datetime("00:35:55") and masterDf.begin <= sub_datetime("00:36:14")]
	

	'''
	 for line in lines:
	        if subtime.findall(line):
	           time = datetime.datetime(1,1,1,*map(int, line[:8].split(':')))
	'''
	pdb.set_trace()









main()