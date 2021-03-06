from bs4 import BeautifulSoup
import urllib.request
import pdb
import pandas as pd
import pickle
import re


def main():
    '''
    Goal: to get some labels for the dataset
    '''
    def get_soup():
        wiki = 'http://survivor.wikia.com/wiki/List_of_Survivor_contestants'
        header = {'User-Agent': 'Mozilla/5.0'}
        # post request
        data = urllib.parse.urlencode(header)
        data = data.encode('ascii')  # data should be bytes
        req = urllib.request.Request(wiki, data)
        with urllib.request.urlopen(req) as response:
            the_page = response.read()

        soup = BeautifulSoup(the_page)
        return soup
    soup = get_soup()

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
    contestant_names = a_text[1::3]  # contestant_names has one more record than the rest of the table

    # beautiful soup doesn't grab Alexis Jones 272, therefore I'm inserting it.
    Age.insert(272, 24)
    Hometown.insert(272, 'Los Angeles,  CA')
    Season.insert(272, 'Micronesia')
    Finish.insert(272, ' 6th')
    Tribal_Challg_Wins.insert(272, '8')
    Indiv_Challg_Wins.insert(272, '1')
    Total_Wins.insert(272, '9')
    Days_Lasted.insert(272, '33')
    Votes_Against.insert(272, '2')

    labels_df = pd.DataFrame({
        # 'Contestent':contestant_names,
        'Contestant_names': contestant_names,
        'Age': Age,
        'Hometown': Hometown,
        'Season': Season,
        'Finish': Finish,
        'Tribal_Challg_Wins': Tribal_Challg_Wins,
        'Indiv_Challg_Wins': Indiv_Challg_Wins,
        'Total_Wins': Total_Wins,
        'Votes_Against': Votes_Against})

    c_labels_df = labels_df.replace('\n', '', regex=True)
    c_labels_df.Finish = c_labels_df.Finish.replace('th', '', regex=True)
    c_labels_df.Finish = c_labels_df.Finish.replace('rd', '', regex=True)
    c_labels_df.Finish = c_labels_df.Finish.replace('Runner-Up', '2', regex=True)
    c_labels_df.Finish = c_labels_df.Finish.replace('Sole Survivor', '1', regex=True)

    file_object = open("season_labels", 'rb')  # changed to rb
    season_labels = pickle.load(file_object)
    season_labels.insert(0, '')
    for i in range(0, len(season_labels)):
        season_labels[i] = re.sub('Survivor: ', '', season_labels[i])
    for i in range(0, len(season_labels)):
        c_labels_df = c_labels_df.replace(season_labels[i], i, regex=True)

    file_object = open("cleaned_label_soup", 'wb')

    pickle.dump(c_labels_df, file_object)
    file_object.close()
    '''
    filenamex = 'test.txt'
    with open('test.txt', 'w') as f:
        for a,b in zip(Age,contestant_names): f.write('{0}{1}'.format(a,b)+ '\n')
    '''
    # len(contestant_names) - >558
    # two ways: contestant_names = [i.text for i in table.find_all("th")]
    #len(contestant_names) -- > 569
    #  len(contestant_names[11:])--->558



main()
