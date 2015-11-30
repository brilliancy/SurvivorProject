import urllib, http.cookiejar
from pprint import pprint
from bs4 import BeautifulSoup
import pdb#pdb.set_trace()
import csv
import pandas as pd
import numpy as np
import sys
import os
import collections
from Data_Preprocessing.xml_scrape_loop import *

def main():	
	class run_scraper(): 
		''' library scraper up to episode 19'''
		episode_dic = {'28':13,'29':14,'30':14,'31':10}
		episode_lib_dic = {'1':13,'2':15,'3':15,'4':14,'5':14,'6':14,'7':14,'8':16,'9':14,'10':14,'11':14,'12':15,'13':15,'14':14,'15':14,'16':14,'17':13,'18':14,'19':15,'20':15,'21':15,'22':13,'23':15,'24':14,'25':14,'26':14,'27':14}
		
		episode_dic = collections.OrderedDict(sorted(episode_dic.items()))
		episode_lib_dic = collections.OrderedDict(sorted(episode_lib_dic.items()))
		for key in episode_lib_dic:
			Season = key
			num_episodes = episode_lib_dic[key]
			for episode_num in range(1,num_episodes+1):
				print('season:',Season,'episode:',episode_num)
				
				UrlName = 'http://www.cbsstatic.com/closedcaption/Library/SURVIVOR/DFXP/CBS_SURVIVOR_{0}{1:0>2}_CIAN_caption_DFXP.xml'.format(Season,episode_num)
				
				script_dir = os.path.dirname(os.path.abspath(__file__)) #changed slightly
				rel_path = 'Data_Preprocessing/DataFrames/{0}{1:0>2}.csv'.format(Season,episode_num)
				abs_file_path = os.path.join(script_dir, rel_path)
				
				print(Season == 17 ,episode_num == 1)
				if os.path.isfile(abs_file_path):
					print('found file for ', abs_file_path)
					continue
				elif Season == '17' and episode_num == 1:
					print(Season,episode_num)
					UrlName = 'http://www.cbsstatic.com/closedcaption/Library/SURVIVOR/DFXP/CBS_SURVIVOR_1701_1702_CIAN_caption_DFXP.xml'					
					soup = load_data(Season,episode_num,UrlName)
					clean_data()
					subtitles_df = create_dataframe(soup)
					save_pandas(Season,episode_num,subtitles_df)
					continue
				elif Season == '17':
					print(Season,episode_num)
					UrlName = 'http://www.cbsstatic.com/closedcaption/Library/SURVIVOR/DFXP/CBS_SURVIVOR_{0}{1:0>2}_CIAN_caption_DFXP.xml'.format(Season,episode_num+1) #adds 1
					soup = load_data(Season,episode_num,UrlName)
					clean_data()
					subtitles_df = create_dataframe(soup)
					save_pandas(Season,episode_num,subtitles_df)
					continue
				else: 

					#if Season == 11 and episode_num == 10: 
					#	continue 
					#else: 
					#UrlName = 'http://www.cbsstatic.com/closedcaption/Library/SURVIVOR/DFXP/CBS_SURVIVOR_{0}{1:0>2}_CIAN_caption_DFXP.xml'.format(Season,episode_num)
					soup = load_data(Season,episode_num,UrlName)
					clean_data()
					subtitles_df = create_dataframe(soup)
					save_pandas(Season,episode_num,subtitles_df)
		for key in episode_dic:
			Season = key
			num_episodes = episode_dic[key]
			for episode_num in range(1,num_episodes+1):
				print('Season:', Season, 'Episode:',episode_num)

				UrlName = 'http://www.cbsstatic.com/closedcaption/Current/Primetime/DFXP/CBS_SURVIVOR_{0}{1:0>2}_CONTENT_CIAN_caption_DFXP.xml'.format(Season,episode_num)
				
				script_dir = os.path.dirname(os.path.abspath(__file__)) #changed slightly
				rel_path = 'Data_Preprocessing/DataFrames/{0}{1:0>2}.csv'.format(Season,episode_num)
				abs_file_path = os.path.join(script_dir, rel_path)

				if os.path.isfile(abs_file_path):
					print('found file for ', abs_file_path)
					continue
				elif Season == '28' and episode_num == 1:
					print(Season,episode_num)
					UrlName = 'http://www.cbsstatic.com/closedcaption/Library/SURVIVOR/DFXP/CBS_SURVIVOR_2801_2802_CIAN_caption_DFXP.xml'					
					soup = load_data(Season,episode_num,UrlName)
					clean_data()
					subtitles_df = create_dataframe(soup)
					save_pandas(Season,episode_num,subtitles_df)
				elif Season == '28':
					print(Season,episode_num)
					UrlName = 'http://www.cbsstatic.com/closedcaption/Current/SURVIVOR/DFXP/CBS_SURVIVOR_{0}{1:0>2}_CIAN_caption_DFXP.xml'.format(Season,episode_num+1) #adds 1
					soup = load_data(Season,episode_num,UrlName)
					clean_data()
					subtitles_df = create_dataframe(soup)
					save_pandas(Season,episode_num,subtitles_df)
					continue
				elif Season == '29' and episode_num == 11:
					print(Season,episode_num)
					UrlName = 'http://www.cbsstatic.com/closedcaption/Current/Primetime/DFXP/CBS_SURVIVOR_2911_2912_CONTENT_CIAN_caption_DFXP.xml'					
					soup = load_data(Season,episode_num,UrlName)
					clean_data()
					subtitles_df = create_dataframe(soup)
					save_pandas(Season,episode_num,subtitles_df)
				elif Season == '29' and episode_num == 13:
					print(Season,episode_num)
					UrlName = 'http://www.cbsstatic.com/closedcaption/Current/Primetime/DFXP/CBS_SURVIVOR_2913_CONTENT_CIAN_caption_DFXP.xml'					
					soup = load_data(Season,episode_num,UrlName)
					clean_data()
					subtitles_df = create_dataframe(soup)
					save_pandas(Season,episode_num,subtitles_df)
				elif Season == '29' and episode_num == 14:
					print(Season,episode_num)
					UrlName = 'http://www.cbsstatic.com/closedcaption/Current/Primetime/DFXP/CBS_SURVIVOR_2916_CONTENT_CIAN_caption_DFXP.xml'					
					soup = load_data(Season,episode_num,UrlName)
					clean_data()
					subtitles_df = create_dataframe(soup)
					save_pandas(Season,episode_num,subtitles_df)
				elif Season == '29':
					print(Season,episode_num)
					UrlName = 'http://www.cbsstatic.com/closedcaption/Current/Primetime/DFXP/CBS_SURVIVOR_{0}{1:0>2}_CONTENT_CIAN_caption_DFXP.xml'.format(Season,episode_num+1) #adds 1
					soup = load_data(Season,episode_num,UrlName)
					clean_data()
					subtitles_df = create_dataframe(soup)
					save_pandas(Season,episode_num,subtitles_df)
				elif Season == '30' and episode_num == 4:
					print(Season,episode_num)
					UrlName = 'http://www.cbsstatic.com/closedcaption/Current/Primetime/DFXP/CBS_SURVIVOR_3004_3005_CONTENT_CIAN_caption_DFXP.xml'					
					soup = load_data(Season,episode_num,UrlName)
					clean_data()
					subtitles_df = create_dataframe(soup)
					save_pandas(Season,episode_num,subtitles_df)
				elif Season == '30':
					print(Season,episode_num)
					UrlName = 'http://www.cbsstatic.com/closedcaption/Current/Primetime/DFXP/CBS_SURVIVOR_{0}{1:0>2}_CONTENT_CIAN_caption_DFXP.xml'.format(Season,episode_num+1) #adds 1
					soup = load_data(Season,episode_num,UrlName)
					clean_data()
					subtitles_df = create_dataframe(soup)
					save_pandas(Season,episode_num,subtitles_df)
				elif Season == '31' and episode_num == 10:
					print(Season,episode_num)
					UrlName = 'http://www.cbsstatic.com/closedcaption/Current/Primetime/DFXP/CBS_SURVIVOR_3110_3111_CONTENT_CIAN_caption_DFXP.xml'					
					soup = load_data(Season,episode_num,UrlName)
					clean_data()
					subtitles_df = create_dataframe(soup)
					save_pandas(Season,episode_num,subtitles_df)
				elif Season == '31':
					print(Season,episode_num)
					UrlName = 'http://www.cbsstatic.com/closedcaption/Current/Primetime/DFXP/CBS_SURVIVOR_{0}{1:0>2}_CONTENT_CIAN_caption_DFXP.xml'.format(Season,episode_num+1) #adds 1
					soup = load_data(Season,episode_num,UrlName)
					clean_data()
					subtitles_df = create_dataframe(soup)
					save_pandas(Season,episode_num,subtitles_df)
				else: 

					#if Season == 11 and episode_num == 10: 
					#	continue 
					#else: 
					#UrlName = 'http://www.cbsstatic.com/closedcaption/Library/SURVIVOR/DFXP/CBS_SURVIVOR_{0}{1:0>2}_CIAN_caption_DFXP.xml'.format(Season,episode_num)
					soup = load_data(Season,episode_num,UrlName)
					clean_data()
					subtitles_df = create_dataframe(soup)
					save_pandas(Season,episode_num,subtitles_df)
main()