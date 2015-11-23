def main():
	import urllib, http.cookiejar
	from pprint import pprint
	from bs4 import BeautifulSoup
	import pdb#pdb.set_trace()
	import csv
	import pandas as pd
	import numpy as np
	import sys
	import os

	cj = http.cookiejar.CookieJar()

	class load_data():
		#UrlMainName = 'http://www.cbs.com/shows/survivor/video/647453A3-F802-2BA1-8C90-BF50EC692B75/survivor-second-chance-play-to-win/'
		#opener = urllib.request.urlopen(UrlMainName)
		#print(opener.read())
		
		Season = '31' # future string formmating loops
		Episode = '7' # library url changes at episode 19,20
		UrlSecondaryName = 'http://www.cbsstatic.com/closedcaption/Current/Primetime/DFXP/CBS_SURVIVOR_{0}{1:0>2}_CONTENT_CIAN_caption_DFXP.xml'.format(Season,Episode)
		UrlLibraryName = 'http://www.cbsstatic.com/closedcaption/Library/SURVIVOR/DFXP/CBS_SURVIVOR_{0}{1:0>2}_CIAN_caption_DFXP.xml'.format(Season,Episode)
		print ('The Second URL is:',UrlSecondaryName)
		req = urllib.request.Request(UrlSecondaryName)
		response = urllib.request.urlopen(req)
		the_page = response.read()

		soup = BeautifulSoup(the_page, 'xml') 
	class clean_data():
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

	
	class create_dataframe():
		#pdb.set_trace()
		subtitles_text = []
		begin = []
		end = []

		for item in load_data.soup.find_all('p'): 
			subtitles_text.append(item.text), begin.append(item['begin']), end.append(item['end'])

		subtitles_df = pd.DataFrame({
			'subtitles_text':subtitles_text,
			'begin':begin,
			'end':end
			})
	
	class save_pandas():
		script_dir = os.path.dirname(__file__)
		rel_path = 'DataFrames/{0}{1:0>2}.csv'.format(load_data.Season,load_data.Episode)
		abs_file_path = os.path.join(script_dir, rel_path)
		create_dataframe.subtitles_df.to_csv(abs_file_path)

	#ISSUE TO SOLVE: getting some funky text from original xml document along with combined words, going to need to spell

	
main()
'''
for the season and episode string formmating url
series = {'1':13,2:13}
for k,v in series.teims():
	print k,v
'''