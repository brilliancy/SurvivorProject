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
	'''
	episode_lib_dic = {'1':13,'2':15,'3':15,'4':14,'5':14,'6':14,'7':14,'8':16,'9':14,'10':14,'11':14,'12':15,'13':15,'14':14,'15':14,'16':14,'17':13,'18':14,'19':15,'20':15,'21':15,'22':13,'23':15,'24':14,'25':14,'26':14,'27':14,'28':13,'29':14,'30':14,'31':11}
	
	first_file = 'DataFrames/101.csv'
	TopDf = pd.read_csv(first_file)
	TopDf['season'] = 1
	for season,episode in sorted(episode_lib_dic.items()):
		for episode_num in range(2,episode+1):
			print('Season:',season,'Episode:',episode_num)
			script_dir = os.path.dirname(__file__)
			rel_path = 'DataFrames/{0}{1:0>2}.csv'.format(season,episode_num)
			filename = os.path.join(script_dir, rel_path)
			print('filename:',filename)
			BottomDf = pd.read_csv(filename)
			BottomDf['season'] = season # this is changing the whole column, 
			frames = [TopDf,BottomDf]
			TopDf = pd.concat(frames)
	MasterDf = TopDf
	MasterDf = MasterDf.drop(MasterDf.columns[0], axis=1)  
	MasterDf.to_csv('DataFrames/MasterDf.csv')
	





main()