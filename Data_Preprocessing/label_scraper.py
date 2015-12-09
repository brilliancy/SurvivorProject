from bs4 import BeautifulSoup
import urllib.request
import pdb



def main():
	'''
	Goal: to get some labels for the dataset
	Problem: Only says what place they finished.
		Solutions:
			1. match it up to each episode (harder but cleaner)
			2. change the model and predict a list of contestents over a reason rather than episode
	'''
	wiki = 'http://survivor.wikia.com/wiki/List_of_Survivor_contestants'
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
	pdb.set_trace()

	table = soup.find("table")
	print(table)


main()