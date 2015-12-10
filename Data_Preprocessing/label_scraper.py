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


	table = soup.find("table")
	# print(table)

	Contestents = []
	Age = ""
	Hometown = ""
	Season = ""
	Finish = ""
	Tribal_Challg_Wins = ""
	Indiv_Challg_Wins = ""
	Total_Wins = ""
	Days_Lasted = ""
	Votes_Against = ""

	for row in table.findAll("tr"):
		cells = [x.text for x in row.findAll("td")]
	#For each "tr", assign each "td" to a variable.
		if len(cells) == 10:
			# pdb.set_trace()

			Contestents.append(cells[0])

			# Contestents.append(contestents)
			#Age = append(cells[1])
			# Hometown = append(cells[2])
			# Season = cells[3].find(text=True)
			# Finish = cells[4].find(text=True)
			# Tribal_Challg_Wins = cells[5].find(text=True)
			# Indiv_Challg_Wins= cells[6].find(text=True)
			# Total_Wins= cells[7].find(text=True)
			# Days_Lasted= cells[8].find(text=True)
			# Votes_Against= cells[9].find(text=True)
	# pdb.set_trace()
	print(Contestents)

main()