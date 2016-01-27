import pdb
import pandas as pd
import numpy as np
import pickle
import re
import os
import datetime
import nltk
import collections
from textblob import TextBlob


from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.ensemble import RandomForestClassifier 

from nltk.tokenize import RegexpTokenizer
from nltk.text import Text

#token = word
#ngram =character or word

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

def cleaning_labelDf():
	
	fileObject = open("Data_Preprocessing/cleaned_label_soup",'rb') #changed to rb
	pickle_obj = pickle.load(fileObject)
	
	clean_pickle = pickle_obj.replace('Runner-UP','2',regex=True)
	clean_pickle = clean_pickle.replace('Winner','1',regex=True)
	clean_pickle = clean_pickle.replace('2nd','2',regex=True)

	return(clean_pickle)
#masterDf = cleaning_MasterDf()	
#labelDf = cleaning_labelDf()

def sub_datetime(k):
	format = "%H:%M:%S"
	time = datetime.datetime.strptime(k[:8],format)
	return time

#changing columns to datetime type

#masterDf.end = masterDf.end.apply(sub_datetime)
#masterDf.begin = masterDf.begin.apply(sub_datetime)

#labelDf = labelDf[:550]
# cleaning  2 2', ' N/A' '1st
def cleaning_Finish_col(labelDf):
	#labelDf = labelDf[:550]
	labelDf.Finish.unique() # find unique values

	labelDf.Finish = labelDf.Finish.replace(' N/A','20',regex=True)#reg
	labelDf.Finish = labelDf.Finish.replace(' 2 2','2', regex=True)
	labelDf.Finish = labelDf.Finish.replace('st','',regex=True)

	labelDf.Finish = labelDf.Finish.apply(lambda k: int(k))
	return(labelDf)

#labelDf = cleaning_Finish_col()

#Starting Model1

def cleaning_Indiv_col(labelDf):
	labelDf.Indiv_Challg_Wins.unique()
	labelDf.Indiv_Challg_Wins = labelDf.Indiv_Challg_Wins.replace([' N/A','N/A'],'0',regex=True)
	labelDf.Indiv_Challg_Wins = labelDf.Indiv_Challg_Wins.apply(lambda k: int(k))
	return(labelDf)

#labelDf = cleaning_Indiv_col()

def cleaning_Tribal_col(labelDf):
	labelDf.Tribal_Challg_Wins.unique()
	labelDf.Tribal_Challg_Wins = labelDf.Tribal_Challg_Wins.replace(' N/A','0',regex=True)
	labelDf.Tribal_Challg_Wins = labelDf.Tribal_Challg_Wins.apply(lambda k: int(k))
	return(labelDf)
#labelDf = cleaning_Tribal_col()

def cleaning_Season_label(labelDf):
	labelDf.Season.unique()
	labelDf.Season = labelDf.Season.replace(' Sou Pacific','23',regex=True)
	labelDf.Season = labelDf.Season.apply(lambda k: int(k))
	return(labelDf)
#labelDf = cleaning_Season_label()
'''
****************************************************************************
Let's start Model 1
****************************************************************************
'''
#bucketing target variable
#bins = numpy.linspace(0,16,3)
#labelDf["Finish_bucket"] = numpy.digitize(labelDf.Finish,bins)

def fit_model_1(labelDf):
	#bucketing target variable
	bins = np.linspace(0,16,3)
	labelDf["Finish_bucket"] = np.digitize(labelDf.Finish,bins)

	# can't add parameter n_jobs = -1 for some reason
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

#fit_model_1()

'''
****************************************************************************
Let's start Model 2
****************************************************************************
'''
#removing noise by selecting features where they are in a 'voting phase'
# Note: I need to further subset the specials episode where the time stamps for voting will be different

#subset_1 = masterDf[(masterDf.begin > sub_datetime("00:26:00")) & (masterDf.end < sub_datetime("00:36:14")) ]

#relevant target variables for Model 2

#labelDf = labelDf[["Contestant_names","Finish","Season"]]

def Label_subset_by_season(Season_begin,Season_end,labelDf):
	#subset label by season
	label_sub_season = labelDf[(labelDf.Season >= Season_begin) & (labelDf.Season <= Season_end)]
	return label_sub_season
#label_sub_season = Label_subset_by_season(28,30)

#subset some seasons in MasterDf just to see if it works at least
def Master_subset_by_season(Season_begin,Season_end,subset_1):
	#temporarily subset_2 instead of masterDf
	subset_2 = subset_1[(subset_1.Season >= Season_begin) & (subset_1.Season <= Season_end)]
	return subset_2

#master_sub_season = Master_subset_by_season(28,30)

def labelDf_target_variable(label_sub_season):
	S28 = label_sub_season[label_sub_season.Season == 28].copy()
	S28_target_cov  = [1,1,2,3,4,5,5,6,7,8,9,10,11,12,13,13,13,100]
	S28.loc[:,'Episode_voted_out'] = S28_target_cov #Season 28,  finale has two people, 100 is to be truncated since they don't ever get voted out

	S29 = label_sub_season[label_sub_season.Season == 29].copy()
	S29_target_cov = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,14,14,14,100]
	S29.loc[:,'Episode_voted_out'] = S29_target_cov #Season 29, finale has three people so the last three of it

	S30 = label_sub_season[label_sub_season.Season == 30].copy()
	S30_target_cov = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,14,14,14,100] #Season 29, finale has three people so the last three of it
	S30.loc[:,'Episode_voted_out'] = S30_target_cov #Season 29, finale has three people so the last three of it

	frames = [S28,S29,S30]
	labelDf = pd.concat(frames)
	return labelDf

#labelDf = labelDf_target_variable()


def Label_subset_by_finale():
	pass
def Master_subset_by_finale():
	pass

#get token episodes from subset by season

def get_ep_tokens(EpID,master_sub_season):
	subset_episode = master_sub_season[master_sub_season.EpisodeId == EpID].copy()

	text = ' '.join(subset_episode.subtitles_text) 
	lowers = text.lower()
	no_punctuation = re.sub('[\W_]+', ' ', lowers)

	
	#tokens = nltk.word_tokenize(no_punctuation)
	tokenizer = RegexpTokenizer('\s+', gaps=True)
	#tokens = tokenizer.tokenize(no_punctuation)
	blob = TextBlob(no_punctuation,tokenizer=tokenizer) 
	
	return blob

#EpUnique = sorted(master_sub_season.EpisodeId.unique())

#tokens_list = []

#for EpID in EpUnique: 
	#tokens_list.append(get_ep_tokens(EpID)) # each item in list is an episode

# name concordance

#the list of names fed into concordance *important*
#list_of_names = label_sub_season.Contestant_names.apply(lambda k: k.split(" "))
def Sentiment_score_FN(tokenized_ep,indiv_name,minrange,maxrange):
	'''
	Pulled from http://www.clips.ua.ac.be/pages/pattern-en#sentiment
	Input in tokenized subtitles from an episode, Input in a name, grabs the concordance around the name
	'''
	agg_firstname_concordance_dic = {}
	agg_lastname_concordance_dic = {}
	first_name_temp = []
	last_name_temp = []
	
	first_name, last_name = indiv_name[0], indiv_name[1]
	
	#print(first_name,last_name)
	first_name_index = [ i for i, j in enumerate(tokenized_ep) if j == first_name.lower()] 
	last_name_index = [ i for i, j in enumerate(tokenized_ep) if j == last_name.lower()] 

	for name_loc in first_name_index:
		first_name_temp.append(tokenized_ep[name_loc-minrange:name_loc+maxrange])
	agg_firstname_concordance_dic[first_name] = first_name_temp

	text = first_name_temp
	agg_sentiment = []

	for i in text: 
		combine = ' '.join(i)
		blob = TextBlob(combine)
		sentiment = blob.sentiment.polarity
		agg_sentiment.append(sentiment)


	try:
		avg_sentiment = sum(agg_sentiment)/len(agg_sentiment)
	except ZeroDivisionError:
		avg_sentiment = 0
	
	#some names like 'So' kim are refercing to the word 'so' instead of name 'So', and will
	#print(agg_firstname_concordance_dic)
	first_name_temp = []
	'''
	for name_loc in last_name_index:
		last_name_temp.append(tokenized_ep[name_loc-minrange:name_loc+maxrange])
	agg_lastname_concordance_dic[last_name] = last_name_temp
	return(agg_firstname_concordance_dic)
	'''

	return(avg_sentiment)

def last_concordance(tokenized_ep,indiv_name,minrange,maxrange):

	agg_firstname_concordance_dic = {}
	agg_lastname_concordance_dic = {}
	first_name_temp = []
	last_name_temp = []
	
	first_name, last_name = indiv_name[0], indiv_name[1]
	
	#print(first_name,last_name)
	first_name_index = [ i for i, j in enumerate(tokenized_ep) if j == first_name.lower()] 
	last_name_index = [ i for i, j in enumerate(tokenized_ep) if j == last_name.lower()] 

	for name_loc in first_name_index:
		first_name_temp.append(tokenized_ep[name_loc-minrange:name_loc+maxrange])
	agg_firstname_concordance_dic[first_name] = first_name_temp
	#some names like 'So' kim are refercing to the word 'so' instead of name 'So', and will
	#print(agg_firstname_concordance_dic)
	first_name_temp = []

	for name_loc in last_name_index:
		last_name_temp.append(tokenized_ep[name_loc-minrange:name_loc+maxrange])
	agg_lastname_concordance_dic[last_name] = last_name_temp	
	return(agg_lastname_concordance_dic)

def concordance(tokenized_ep,indiv_name,minrange,maxrange):

	agg_firstname_concordance_dic = {}
	agg_lastname_concordance_dic = {}
	first_name_temp = []
	last_name_temp = []
	
	first_name, last_name = indiv_name[0], indiv_name[1]
	
	#print(first_name,last_name)
	first_name_index = [ i for i, j in enumerate(tokenized_ep) if j == first_name.lower()] 
	last_name_index = [ i for i, j in enumerate(tokenized_ep) if j == last_name.lower()] 

	for name_loc in first_name_index:
		first_name_temp.append(tokenized_ep[name_loc-minrange:name_loc+maxrange])
	agg_firstname_concordance_dic[first_name] = first_name_temp
	#some names like 'So' kim are refercing to the word 'so' instead of name 'So', and will
	#print(agg_firstname_concordance_dic)
	first_name_temp = []

	for name_loc in last_name_index:
		last_name_temp.append(tokenized_ep[name_loc-minrange:name_loc+maxrange])
	agg_lastname_concordance_dic[last_name] = last_name_temp	
	

	'''
	for indiv_name in list_of_names:
		pdb.set_trace()
		first_name, last_name = indiv_name[0], indiv_name[1]
		#print(first_name,last_name)
		first_name_index = [ i for i, j in enumerate(tokenized_ep) if j == first_name.lower()] 
		last_name_index = [ i for i, j in enumerate(tokenized_ep) if j == last_name.lower()] 

		for name_loc in first_name_index:
			first_name_temp.append(tokenized_ep[name_loc-minrange:name_loc+maxrange])
		agg_firstname_concordance_dic[first_name] = first_name_temp
		#some names like 'So' kim are refercing to the word 'so' instead of name 'So', and will
		#print(agg_firstname_concordance_dic)
		first_name_temp = []

		for name_loc in last_name_index:
			last_name_temp.append(tokenized_ep[name_loc-minrange:name_loc+maxrange])
		agg_lastname_concordance_dic[last_name] = last_name_temp	
		last_name_temp = []
	'''
	return(agg_firstname_concordance_dic)#agg_lastname_concordance_dic

def concordance_packager(labelDf,tokens_list):
	firstname_concordance = []
	lastname_concordance = []

	columns = ["Season","Episode","Firstname_concordance","Lastname_concordance", "Sentiment_score"]
	ConcordDf = pd.DataFrame(columns = columns)

	TopDf = pd.DataFrame(columns = ["Firstname_concordance"])
	
	episode_list = [13, 15, 15, 14,14,14,14,16,14,14,14,15,15,14,14,14,13,14,15,15,15,13,15,14,14,14,14,13,14,14,11]
	season_list = [i for i in range(1,32)]
	episode_ord_dict = collections.OrderedDict()
	for x,y in zip(season_list,episode_list):
		episode_ord_dict[int(x)] = y 

	row = []
	aggrow = []
	for season in labelDf.Season.unique():
		contestantsPerSeason = labelDf.Contestant_names[labelDf.Season == season] 
		episode_length = episode_ord_dict[season]

		for episode in range(1,episode_length+1):
			row.append(episode)
			aggrow.append(season)

	ConcordDf['Season'] = aggrow
	ConcordDf['Episode'] = row
	
	#the list of names fed into concordance *important*
	#label_sub_season.Season.unique()
	
	#S_unique
	list_of_names = labelDf.Contestant_names.apply(lambda k: k.split(" "))

	for textblob in tokens_list: #tokens_list is a list of tokens. Each one is a episode subsetted by time and season 
		first_con, last_con = concordance(textblob.tokens,list_of_names,minrange=5,maxrange=5)
		firstname_concordance.append(first_con) 
		lastname_concordance.append(last_con) 
		
		'''
		BottomDf = pd.DataFrame(columns = ["Firstname_concordance"])

		con = np.array(first_con.items())
		BottomDf['Firstname_concordance'] = con
		#i can get it out with con.item()
		#.toarray().tolist()
		frames = [TopDf,BottomDf]
		TopDf = pd.concat(frames)
		
		'''
		#pdb.set_trace()
		#dictionary obj with key = contestant names, value = concordances around name
	#panda df with season,episode,firstname_concordance,lastname_concordance, sentiment score
	#then I want to rank the sentiment scores
	ConcordDf['Season'] = aggrow
	ConcordDf['Episode'] = row
	# ******** commented out ******
	#pdb.set_trace()
	ConcordDf['Firstname_concordance'] = firstname_concordance
	ConcordDf['Lastname_concordance'] = lastname_concordance
	
	'''
	frames = [TopDf,MiddleDf]
	TopDf = pd.concat(frames)
	'''

	ConcordDf = ConcordDf[["Season","Episode","Firstname_concordance","Lastname_concordance", "Sentiment_score"]]
	#ConcordDf['Sentiment_score'] = [40*"Sentiment_score"]
	return ConcordDf

def concordance_sentiment_packager(labelDf,tokens_list):
	firstname_concordance = []
	lastname_concordance = []

	columns = ["Season","Episode","Contestant","Firstname_concordance","Lastname_concordance", "Sentiment_score_FN"]
	ConcordDf = pd.DataFrame(columns = columns)

	TopDf = pd.DataFrame(columns = ["Firstname_concordance"])
	
	episode_list = [13, 15, 15, 14,14,14,14,16,14,14,14,15,15,14,14,14,13,14,15,15,15,13,15,14,14,14,14,13,14,14,11]
	season_list = [i for i in range(1,32)]
	episode_ord_dict = collections.OrderedDict()
	for x,y in zip(season_list,episode_list):
		episode_ord_dict[int(x)] = y 

	#the list of names fed into concordance *important*
	#label_sub_season.Season.unique()
	
	#S_unique

	#labelDf.Season.unique()
	#len(list_of_names)
	def contestant_index(labelDf):
		pass
	'''
	Contestant_index = []
	for i in labelDf.Season.unique(): #subsets contestants based on season
		subset = labelDf.Contestant_names[labelDf.Season == i]
		list_of_names = subset.apply(lambda k: k.split(" "))
		#Contestant_index = [list_of_names for i in range(1,18)]
		episode_length = 17
		for i in range(1,len(episode_length)+1):
			Contestant_index.append(list_of_names)
	pdb.set_trace()	
	'''
	def season_episode_contestant_index(labelDf):
		row = []
		aggrow = []
		Contestant_index = []
		a = 0
		index = []
		for season in labelDf.Season.unique():
			contestantsPerSeason = labelDf.Contestant_names[labelDf.Season == season] 
			episode_length = episode_ord_dict[season]

			for episode in range(1,episode_length+1):
				#row.append(episode)
				subset_contestant = labelDf.Contestant_names[labelDf.Season == season]
				list_of_names = labelDf.Contestant_names.apply(lambda k: k.split(" "))
				row.extend([episode for i in range(1,len(subset_contestant)+1)])
				index.extend([a for i in range(1,len(subset_contestant)+1)])
				
				#[episode for i in range(1,len(list_of_names)+1)] 
				#aggrow.append(season)
				aggrow.extend([season for i in range(1,len(subset_contestant)+1)])

				Contestant_index.extend(subset_contestant)
				a += 1
		return (row,aggrow,Contestant_index,index)

	episode, season,Contestant_index,index = season_episode_contestant_index(labelDf)
	ConcordDf['Season'] = season
	ConcordDf['Episode'] = episode
	ConcordDf['Contestant']= Contestant_index
	ConcordDf['Index'] = index

	#ConcordDf['Index'] = ConcordDf.Index.apply(lambda k: int(k))#.astype(int)
	#ConcordDf['Index'] = ConcordDf.Index.apply(lambda k: k.astype(int))
	#ConcordDf['Index'] = ConcordDf.Index.apply(int())
	ConcordDf["Contestant"] = ConcordDf.Contestant.apply(lambda k: k.split(" "))
	#ConcordDf["Firstname_concordance"] = ConcordDf.apply(lambda row: row,axis=1)
	#ConcordDf["Firstname_concordance"] = ConcordDf.Contestant.apply(lambda row: )
	#ConcordDf["Firstname_concordance"] = concordance(tokens_list[ConcordDf.Index[0]],ConcordDf.Contestant,minrange=5,maxrange=5)

	def search(row):
		#pdb.set_trace()
		return concordance(tokens_list[row.Index].tokens,row.Contestant,minrange=5,maxrange=5)

	ConcordDf["Firstname_concordance"] = ConcordDf.apply(search, axis=1)
	
	def searchL(row):
		#pdb.set_trace()
		return last_concordance(tokens_list[row.Index].tokens,row.Contestant,minrange=5,maxrange=5)
	
	ConcordDf["Lastname_concordance"] = ConcordDf.apply(searchL, axis=1)

	def sent_search(row):
		
		return Sentiment_score_FN(tokens_list[row.Index].tokens,row.Contestant,minrange=5,maxrange=5)
	ConcordDf["Sentiment_score"] = ConcordDf.apply(sent_search,axis=1)
	pdb.set_trace()
	labelDf


	#def concordance():
	#	ConcordDf["Firstname_concordance"] = concordance(tokens_list[ConcordDf.Index[0]],ConcordDf.Contestant,minrange=5,maxrange=5)

	#concordance(tokens_list[0],current_name,minrange=5,maxrange=5)
	'''
	
	#for each row in contestant column, that becomes list of names. get the concordance for that specific episode and save it in firstname_concordance
	#having difficulting subsetting through textblob 
	#have a key that corresponds with Episode number and tokenblob index
	ConcordDf["Firstname_concordance"] = ConcordDf.Firstname.apply(lambda current_name: concordance(tokens_list[0],current_name,minrange=5,maxrange=5))

	for textblob in tokens_list: #tokens_list is a list of tokens. Each one is a episode subsetted by time and season 
		
		ConcordDf["Firstname_concordance"] = ConcordDf.Firstname.apply(lambda current_name: concordance(textblob.tokens,current_name,minrange=5,maxrange=5))
		#first_con, last_con = concordance(textblob.tokens,current_name,minrange=5,maxrange=5)
		#firstname_concordance.append(first_con) 
		#lastname_concordance.append(last_con) 
	'''
	#aggrow.extend([list_of_names for i in range(1,len(episode_length)+1)])



	#****** each episode_blob is an episode tokenized, for each row feed in the contestant in each row into the concordance function
	#fill in the concordance


	
		
		#dictionary obj with key = contestant names, value = concordances around name
	#panda df with season,episode,firstname_concordance,lastname_concordance, sentiment score
	#then I want to rank the sentiment scores

	# ******** commented out ******
	
	#ConcordDf['Firstname_concordance'] = firstname_concordance
	#ConcordDf['Lastname_concordance'] = lastname_concordance
	
	'''
	frames = [TopDf,MiddleDf]
	TopDf = pd.concat(frames)
	'''

	ConcordDf = ConcordDf[["Season","Episode","Contestant","Firstname_concordance","Lastname_concordance", "Sentiment_score"]]

	#ConcordDf['Sentiment_score'] = [40*"Sentiment_score"]
	return ConcordDf

#ngrams of concordances

#row names contestname name and episode number

'''
subset by unique episode
get the Contestant_names for that episode
#sync up Contestant_name to that particular episode
problem: during special they might vote out more than one person

note: possibly also need to add metadata in the form of episode to the concordance so i can subset by episode as well
note: Do I tokensize before or AFTER the concordance
'''
'''
ModelDf_col = []
ModelDf_col.append()

labelDf.Contestant_names
master_sub_season
ModelDf = ModelDf[ModelDf]

Contestant_names, Episode_voted_out 

need to truncate the last portion of votes
'''

def featureDf_index(labelDf):
	row = []

	episode_list = [13, 15, 15, 14,14,14,14,16,14,14,14,15,15,14,14,14,13,14,15,15,15,13,15,14,14,14,14,13,14,14,11]
	season_list = [i for i in range(1,32)]
	episode_ord_dict = collections.OrderedDict()
	for x,y in zip(season_list,episode_list):
		episode_ord_dict[int(x)] = y 

	for season in labelDf.Season.unique():
		contestantsPerSeason = labelDf.Contestant_names[labelDf.Season == season] 
		episode_length = episode_ord_dict[season]
		

		for episode in range(1,episode_length+1):
			for name in contestantsPerSeason:
				row.append([name,episode])
				#print("row\n",row)
			#do i actually want to remove contestant names after they been voted out on the rows? I don't think so since the target variable would be somewhat embedded into the model
			#Contestant_names_per_season = Contestant_names_per_season[Contestant_names_per_season.Episode_voted_out != ]
			
	return(row)
def main():

	masterDf = cleaning_MasterDf()	
	labelDf = cleaning_labelDf()

	#changing columns to datetime type
	masterDf.end = masterDf.end.apply(sub_datetime)
	masterDf.begin = masterDf.begin.apply(sub_datetime)

	#getting rid of contestents in new season 
	# ****remove this bit later on when updated new data*****
	labelDf = labelDf[:550]

	#clean some various columns
	labelDf = cleaning_Finish_col(labelDf)
	labelDf = cleaning_Indiv_col(labelDf)
	labelDf = cleaning_Tribal_col(labelDf)
	labelDf = cleaning_Season_label(labelDf)

	#model 1:
	#fit_model_1(labelDf)
	

	#model 2:(WIP)
	
	#remove noise

	#******removed timestamps, add back later*******
	#***************************************
	subset_1 = masterDf[(masterDf.begin > sub_datetime("00:00:00")) & (masterDf.end < sub_datetime("00:36:14")) ]
	#subset_1 = masterDf
	labelDf = labelDf[["Contestant_names","Finish","Season"]]
	
	#subset of data to work on at first
	label_sub_season = Label_subset_by_season(28,30,labelDf)
	master_sub_season = Master_subset_by_season(28,30,subset_1)
	

	labelDf = labelDf_target_variable(label_sub_season)

	EpUnique = sorted(master_sub_season.EpisodeId.unique())

	tokens_list = []

	for EpID in EpUnique: 
		tokens_list.append(get_ep_tokens(EpID,master_sub_season))

	#list_of_names = label_sub_season.Contestant_names.apply(lambda k: k.split(" "))

	rows = featureDf_index(labelDf)
	
	'''
	model 3: sentiment score
	'''
	#ConcordDf = concordance_packager(labelDf,tokens_list)
	sentimentDf = concordance_sentiment_packager(labelDf,tokens_list)
	#sentimentDf[sentimentDf.Episode == 1].sort('Sentiment_score',ascending = True)
	pdb.set_trace()
	
	

main()

