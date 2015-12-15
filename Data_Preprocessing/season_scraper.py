from bs4 import BeautifulSoup
import urllib.request
import pdb
import pandas as pd
import pickle

def main():
	'''
	Goal: to get some labels for the dataset
	Problem: Only says what place they finished.
		Solutions:
			1. match it up to each episode (harder but cleaner)
			2. change the model and predict a list of contestants over a reason rather than episode
	'''
	def get_soup():
		wiki = 'https://en.wikipedia.org/wiki/Survivor_(U.S._TV_series)'
		header = {'User-Agent': 'Mozilla/5.0'}
		#post request
		data = urllib.parse.urlencode(header)
		data = data.encode('ascii') # data should be bytes
		req = urllib.request.Request(wiki, data)
		with urllib.request.urlopen(req) as response:
			the_page = response.read()

		#pdb.set_trace()
		#req = urllib.request(wiki,headers=header)
		#page = urllib.urlopen(req)
		soup = BeautifulSoup(the_page)
		return soup 
	soup = get_soup()
	table = soup.find("tbody")
	i_text = [i.text for i in soup.find_all("i")]
	season_labels = i_text[13:44]

	fileObject = open("season_labels",'wb') 
	
	pickle.dump(season_labels,fileObject)   
	fileObject.close()

main()