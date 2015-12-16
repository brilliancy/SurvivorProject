import pdb
import pandas as pd
import numpy
import pickle
import re
import os
import datetime

from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.ensemble import RandomForestClassifier 

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
		format = "%H:%M:%S"
		time = datetime.datetime.strptime(k[:8],format)
		return time
	
	#changing columns to datetime type
	masterDf.end = masterDf.end.apply(sub_datetime)
	masterDf.begin = masterDf.begin.apply(sub_datetime)

	#removing noise by selecting features where they are in a 'voting phase'
	# Note: I need to further subset the specials episode where the time stamps for voting will be different
	subset_1 = masterDf[masterDf.begin > sub_datetime("00:28:34")]
	subset_2 = subset_1[subset_1.begin < sub_datetime("00:36:14")]
	#can't do this apparently but i can just subset it twice
	#masterDf[masterDf.begin >= sub_datetime("00:35:55") and masterDf.begin <= sub_datetime("00:36:14")]
	

	#subset some seasons just to see if it works at least
	subset_3 = subset_2[subset_2["Season"].isin([28,29,30])]

	labelDf = labelDf[:550]
	# cleaning  2 2', ' N/A' '1st
	def cleaning_Finish_col():
		#labelDf = labelDf[:550]
		labelDf.Finish.unique() # find unique values

		labelDf.Finish = labelDf.Finish.replace(' N/A','20')
		labelDf.Finish = labelDf.Finish.replace(' 2 2','2')
		labelDf.Finish = labelDf.Finish.replace('st','',regex=True)

		labelDf.Finish = labelDf.Finish.apply(lambda k: int(k))
		return(labelDf)

	labelDf = cleaning_Finish_col()

	#Starting Model1

	def cleaning_Indiv_col():
		labelDf.Indiv_Challg_Wins.unique()
		labelDf.Indiv_Challg_Wins = labelDf.Indiv_Challg_Wins.replace([' N/A','N/A'],'0',regex=True)
		labelDf.Indiv_Challg_Wins = labelDf.Indiv_Challg_Wins.apply(lambda k: int(k))
		return(labelDf)
	
	labelDf = cleaning_Indiv_col()

	def cleaning_Tribal_col():
		labelDf.Tribal_Challg_Wins.unique()
		labelDf.Tribal_Challg_Wins = labelDf.Tribal_Challg_Wins.replace(' N/A','0',regex=True)
		labelDf.Tribal_Challg_Wins = labelDf.Tribal_Challg_Wins.apply(lambda k: int(k))
		return(labelDf)
	labelDf = cleaning_Tribal_col()

	def cleaning_Season_label():
		labelDf.Season.unique()
		labelDf.Season = labelDf.Season.replace(' Sou Pacific','23',regex=True)
		labelDf.Season = labelDf.Season.apply(lambda k: int(k))
		return(labelDf)
	labelDf = cleaning_Season_label()

	#Let's start Model 1 

	#bucketing target variable
	bins = numpy.linspace(0,16,3)
	labelDf["Finish_bucket"] = numpy.digitize(labelDf.Finish,bins)

	def fit_model_1():
		models = [LogisticRegression(),GradientBoostingClassifier(),RandomForestClassifier()]

		features = labelDf[["Indiv_Challg_Wins","Tribal_Challg_Wins"]]
		target = labelDf["Finish_bucket"]

		scores = [cross_val_score(i,features,target,cv = 10) for i in models]

		print('\nLogistic Regression acc: ', '\n', scores[0])
		print('\nAVG Logistic Regression acc: ', scores[0].mean())

		print('\nGradient Boosting Classifier: ', '\n', scores[1])
		print('\nAVG Gradient Boosting Classifier acc: ', scores[1].mean())

		print('\nRandomForestClassifier: ', '\n', scores[2])
		print('\nAVG RandomForestClassifier acc: ', scores[2].mean())

	#Let's start Model 2

	#relevant target variables for Model 2
	labelDf = labelDf[["Contestant_names","Finish"]]


	'''
	 for line in lines:
	        if subtime.findall(line):
	           time = datetime.datetime(1,1,1,*map(int, line[:8].split(':')))
	'''









main()