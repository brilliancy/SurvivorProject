#def scraper():
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

cj = http.cookiejar.CookieJar()


def load_data(Season,episode_num,UrlName):
	#UrlMainName = 'http://www.cbs.com/shows/survivor/video/647453A3-F802-2BA1-8C90-BF50EC692B75/survivor-second-chance-play-to-win/'
	#opener = urllib.request.urlopen(UrlMainName)
	#print(opener.read())

	#Season = '31' # future string formmating loops
	#Episode = '7' # library url changes at episode 19,20
	#UrlSecondaryName = 'http://www.cbsstatic.com/closedcaption/Current/Primetime/DFXP/CBS_SURVIVOR_{0}{1:0>2}_CONTENT_CIAN_caption_DFXP.xml'.format(Season,Episode)
	#UrlLibraryName = 'http://www.cbsstatic.com/closedcaption/Library/SURVIVOR/DFXP/CBS_SURVIVOR_{0}{1:0>2}_CIAN_caption_DFXP.xml'.format(Season,Episode)
	print ('The URL is:',UrlName)
	req = urllib.request.Request(UrlName)
	response = urllib.request.urlopen(req)
	the_page = response.read()

	soup = BeautifulSoup(the_page, 'xml') 
	return soup
def clean_data():
	pass
'''
note: will have to update the data frame in save_pandas to the clean one
Specification: 
	if length of text is longer than a certain amount then insert a space at a certain point
	cons no longer works since the concatenation is not at a fixed point
Plan B_Specification:
	See how BeautifulSoup is getting the text and if it grabs to lines then to split them
	http://stackoverflow.com/questions/28385881/modifying-a-beautifulsoup-string-with-line-breaks
	http://stackoverflow.com/questions/31958517/beautifulsoup-how-to-extract-text-after-br-tag
Plan C_specification: 
	Use norvig's spellcheck http://norvig.com/spell-correct.html
	cons: might cause



Plan B:
for row in soup.find_all('p'): br_edit = soup.find_all('br')
for i in br_edit: ''.join(i.next_siblings) #concatenates the generator obj

'''


def create_dataframe(soup):
	#pdb.set_trace()
	subtitles_text = []
	begin = []
	end = []

	for item in soup.find_all('p'): 
		subtitles_text.append(item.text) #note: Season 11, episode 10

		begin_val = item.get('begin', 'bs val')
		begin.append(begin_val)
		end_val = item.get('begin', 'bs val')
		end.append(end_val)

	#if 'bs val' in begin or 'bs val' in end:
	#	print (begin, end)

	subtitles_df = pd.DataFrame({
		'subtitles_text':subtitles_text,
		'begin':begin,
		'end':end
		})
	return subtitles_df

def save_pandas(Season,episode_num,subtitles_df):
	script_dir = os.path.dirname(__file__)
	rel_path = 'DataFrames/{0}{1:0>2}.csv'.format(Season,episode_num)
	abs_file_path = os.path.join(script_dir, rel_path)
	subtitles_df.to_csv(abs_file_path)

#ISSUE TO SOLVE: getting some funky text from original xml document along with combined words, going to need to spell

#scraper()
