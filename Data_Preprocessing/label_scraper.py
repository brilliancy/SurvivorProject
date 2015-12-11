from bs4 import BeautifulSoup
import urllib.request
import pdb
import pandas as pd


def main():
	'''
	Goal: to get some labels for the dataset
	Problem: Only says what place they finished.
		Solutions:
			1. match it up to each episode (harder but cleaner)
			2. change the model and predict a list of contestants over a reason rather than episode
	'''
	def get_soup():
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
		return soup 
	soup = get_soup()
	'''
	fileObject = open("pickle_soup_label",'wb') 
	
	pickle.dump(tokens,fileObject)   
	fileObject.close()
	fileObject = open("pickle_EDA",'rb') #changed to rb
	pickle_obj = pickle.load(fileObject)  
	'''
	table = soup.find("table")

	Contestants = []
	Age = []
	Hometown = []
	Season = []
	Finish = []
	Tribal_Challg_Wins = []
	Indiv_Challg_Wins = []
	Total_Wins = []
	Days_Lasted = []
	Votes_Against = []

	for row in table.findAll("tr"):
		cells = [x.text for x in row.findAll("td")]
	#For each "tr", assign each "td" to a variable.
		if len(cells) == 10:
			# pdb.set_trace()
			'''
			[i.text for i in table.findAll("a")]
			a_tags = soup.find_all("a")
			a_text = [i.text for i in table.findAll("a")]
			contestant_names = [a_text[i] for i in range(0,len(a_text)+1) if i%2 == 0]


			'''
			Age.append(cells[1])
			Hometown.append(cells[2])
			Season.append(cells[3])
			Finish.append(cells[4])
			Tribal_Challg_Wins.append(cells[5])
			Indiv_Challg_Wins.append(cells[6])
			Total_Wins.append(cells[7])
			Days_Lasted.append(cells[8])
			Votes_Against.append(cells[9])

	
	a_text = [i.text for i in table.findAll("a")]
	contestant_names = a_text[1::3] # contestant_names has one more record than the rest of the table
	'''
	labels_df = pd.DataFrame({
		#'Contestent':contestant_names,
		'Age':Age,
		'Hometown':Hometown,
		'Season':Season,
		'Finish':Finish,
		'Tribal_Challg_Wins':Tribal_Challg_Wins,
		'Indiv_Challg_Wins':Indiv_Challg_Wins,
		'Total_Wins':Total_Wins,
		'Votes_Against':Votes_Against
		})
	'''

	labels_df = pd.DataFrame({
		#'Contestent':contestant_names,
		'Contestant_names':contestant_names[:200],
		'Age':Age[:200],
		'Hometown':Hometown[:200],
		'Season':Season[:200],
		'Finish':Finish[:200],
		'Tribal_Challg_Wins':Tribal_Challg_Wins[:200],
		'Indiv_Challg_Wins':Indiv_Challg_Wins[:200],
		'Total_Wins':Total_Wins[:200],
		'Votes_Against':Votes_Against[:200]
		})
	print(Age)
	pdb.set_trace()
	# len(contestant_names) - >558
	# two ways: contestant_names = [i.text for i in table.find_all("th")]
	#len(contestant_names) -- > 569
	#  len(contestant_names[11:])--->558

	

main()