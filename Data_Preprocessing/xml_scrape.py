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
		UrlMainName = 'http://www.cbs.com/shows/survivor/video/647453A3-F802-2BA1-8C90-BF50EC692B75/survivor-second-chance-play-to-win/'
		opener = urllib.request.urlopen(UrlMainName)
		#print(opener.read())
		
		Season = '31' # future string formmating loops
		Episode = '7'
		UrlSecondaryName = 'http://www.cbsstatic.com/closedcaption/Current/Primetime/DFXP/CBS_SURVIVOR_{0}{1:0>2}_CONTENT_CIAN_caption_DFXP.xml'.format(Season,Episode)
		print ('The Second URL is:',UrlSecondaryName)
		req = urllib.request.Request(UrlSecondaryName)
		response = urllib.request.urlopen(req)
		the_page = response.read()

		soup = BeautifulSoup(the_page, 'xml') #should I set parser to xml 
	
	
	
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
	pdb.set_trace()

	
main()
'''
for the season and episode string formmating url
series = {'1':13,2:13}
for k,v in series.teims():
	print k,v
'''