import urllib, http.cookiejar
from pprint import pprint
from bs4 import BeautifulSoup
import pdb
import csv
import pandas as pd
import numpy as np
import sys
import os
import collections
import re

cj = http.cookiejar.CookieJar()


def load_data(Season, episode_num, UrlName):
    '''
    UrlMainName = 'http://www.cbs.com/shows/survivor/video/647453A3-F802-2BA1-8C90-BF50EC692B75/survivor-second-chance-play-to-win/'
    opener = urllib.request.urlopen(UrlMainName)
    print(opener.read())

    Season = '31' # future string formmating loops
    Episode = '7' # library url changes at episode 19,20
    UrlSecondaryName = 'http://www.cbsstatic.com/closedcaption/Current/Primetime/DFXP/CBS_SURVIVOR_{0}{1:0>2}_CONTENT_CIAN_caption_DFXP.xml'.format(Season,Episode)
    UrlLibraryName = 'http://www.cbsstatic.com/closedcaption/Library/SURVIVOR/DFXP/CBS_SURVIVOR_{0}{1:0>2}_CIAN_caption_DFXP.xml'.format(Season,Episode)
    '''
    print ('The URL is:', UrlName)
    req = urllib.request.Request(UrlName)
    response = urllib.request.urlopen(req)
    the_page = response.read()

    new_text = re.sub('<br></br>', ' ', str(the_page))  # replace br tags with spaces
    soup = BeautifulSoup(new_text, )

    return soup


def clean_data():
    pass


def create_dataframe(soup):

    subtitles_text = []
    begin = []
    end = []

    for item in soup.find_all('p'):
        subtitles_text.append(item.text)
        begin_val = item.get('begin', 'bs val')
        begin.append(begin_val)
        end_val = item.get('begin', 'bs val')
        end.append(end_val)

    subtitles_df = pd.DataFrame({
        'subtitles_text': subtitles_text,
        'begin': begin,
        'end': end})
    subtitles_df = subtitles_df.ix[1:]  # removing the first row of the data frame
    return subtitles_df


def save_pandas(Season, episode_num, subtitles_df):
    script_dir = os.path.dirname(__file__)
    rel_path = 'DataFrames/{0}{1:0>2}.csv'.format(Season, episode_num)
    abs_file_path = os.path.join(script_dir, rel_path)
    subtitles_df.to_csv(abs_file_path)
