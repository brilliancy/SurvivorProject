import csv
import pandas as pd
import numpy as np
import collections
import sys 
import os
import pdb

def main():
	'''
	combine all dataframe csv's into one large dataframe with an additional column that specifies what 
	season
	and episode it is
	SentenceId,EpisodeId,Season,Episode,begin,end,Sentence

	******PROBLEM: makes an assumption that the dictionary is in order, however it isn't****
	'''
	#episode_list = ['1':13,'2':15,'3':15,'4':14,'5':14,'6':14,'7':14,'8':16,'9':14,'10':14,'11':14,'12':15,'13':15,'14':14,'15':14,'16':14,'17':13,'18':14,'19':15,'20':15,'21':15,'22':13,'23':15,'24':14,'25':14,'26':14,'27':14,'28':13,'29':14,'30':14,'31':11]
	#episode_lib_dic = {'1':13,'2':15,'3':15,'4':14,'5':14,'6':14,'7':14,'8':16,'9':14,'10':14,'11':14,'12':15,'13':15,'14':14,'15':14,'16':14,'17':13,'18':14,'19':15,'20':15,'21':15,'22':13,'23':15,'24':14,'25':14,'26':14,'27':14,'28':13,'29':14,'30':14,'31':11}
	
	episode_list = [13, 15, 15, 14,14,14,14,16,14,14,14,15,15,14,14,14,13,14,15,15,15,13,15,14,14,14,14,13,14,14,11]
	season_list = [i for i in range(1,32)]
	
	episode_ord_dict = collections.OrderedDict()
	
	for x,y in zip(season_list,episode_list):
		episode_ord_dict[str(x)] = y 


	first_file = 'DataFrames/101.csv'
	TopDf = pd.read_csv(first_file)
	TopDf['Season'] = 1
	TopDf['Episode'] = 1

	EpisodeID_num = 1
	TopDf['EpisodeId'] = EpisodeID_num

	for season,episode in episode_ord_dict.items():
		for episode_num in range(2,episode+1):
			print('Season:',season,'Episode:',episode_num)
			script_dir = os.path.dirname(__file__)
			rel_path = 'DataFrames/{0}{1:0>2}.csv'.format(season,episode_num)
			filename = os.path.join(script_dir, rel_path)
			print('filename:',filename)
			BottomDf = pd.read_csv(filename)
			# adding season/episode col
			BottomDf['Season'] = season 
			BottomDf['Episode'] = episode 
			#adding episode id
			EpisodeID_num += 1
			BottomDf['EpisodeId'] = EpisodeID_num
			frames = [TopDf,BottomDf]
			TopDf = pd.concat(frames)

	MasterDf = TopDf
	MasterDf = MasterDf.drop(MasterDf.columns[0], axis=1)  
	
	#adding SentenceId
	sentence_list = [i for i in range(0,len(MasterDf)) ]
	MasterDf['SentenceId'] = sentence_list
	# reordering in a better way
	MasterDf = MasterDf[['SentenceId','EpisodeId', 'Season','Episode','begin','end','subtitles_text']]




	MasterDf.to_csv('DataFrames/MasterDf.csv')
	





main()