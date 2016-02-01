import pdb
import pandas as pd
import numpy as np
import pickle
import re
import datetime
import collections
from textblob import TextBlob

from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

from nltk.tokenize import RegexpTokenizer
import Levenshtein


def cleaning_masterdf():
    master_df = pd.read_csv("Data_Preprocessing/DataFrames/MasterDf.csv")
    return(master_df)


def cleaning_labels():
    file_object = open("Data_Preprocessing/cleaned_label_soup", 'rb')
    pickle_obj = pickle.load(file_object)

    clean_pickle = pickle_obj.replace('Runner-UP', '2', regex=True)
    clean_pickle = clean_pickle.replace('Winner', '1', regex=True)
    clean_pickle = clean_pickle.replace('2nd', '2', regex=True)

    return(clean_pickle)


def sub_datetime(k):
    format = "%H:%M:%S"
    time = datetime.datetime.strptime(k[:8], format)
    return time


def cleaning_finish_col(label_df):
    label_df.Finish.unique()

    label_df.Finish = label_df.Finish.replace(' N/A', '20', regex=True)
    label_df.Finish = label_df.Finish.replace(' 2 2', '2', regex=True)
    label_df.Finish = label_df.Finish.replace('st', '', regex=True)

    label_df.Finish = label_df.Finish.apply(lambda k: int(k))
    return(label_df)


def cleaning_indiv_col(label_df):
    label_df.Indiv_Challg_Wins.unique()
    label_df.Indiv_Challg_Wins = label_df.Indiv_Challg_Wins.replace([' N/A', 'N/A'], '0', regex=True)
    label_df.Indiv_Challg_Wins = label_df.Indiv_Challg_Wins.apply(lambda k: int(k))
    return(label_df)


def cleaning_tribal_col(label_df):
    label_df.Tribal_Challg_Wins.unique()
    label_df.Tribal_Challg_Wins = label_df.Tribal_Challg_Wins.replace(' N/A', '0', regex=True)
    label_df.Tribal_Challg_Wins = label_df.Tribal_Challg_Wins.apply(lambda k: int(k))
    return(label_df)


def cleaning_season_label(label_df):

    label_df.Season.unique()
    label_df.Season = label_df.Season.replace(' sou Pacific', '23', regex=True)
    label_df.Season = label_df.Season.apply(lambda k: int(k))
    return(label_df)

'''
****************************************************************************
Let's start Model 1
****************************************************************************
'''


def fit_model_1(label_df):
    # bucketing target variable
    bins = np.linspace(0, 16, 3)
    label_df["Finish_bucket"] = np.digitize(label_df.Finish, bins)

    # can't add parameter n_jobs = -1 for some reason
    models = [LogisticRegression(), GradientBoostingClassifier(), RandomForestClassifier()]

    features = label_df[["Indiv_Challg_Wins", "Tribal_Challg_Wins"]]
    target = label_df["Finish_bucket"]

    scores = [cross_val_score(i, features, target, cv=10) for i in models]

    print('\nLogistic Regression acc: ', '\n', scores[0])
    print('\nAVG Logistic Regression acc: ', scores[0].mean())

    print('\nGradient Boosting Classifier: ', '\n', scores[1])
    print('\nAVG Gradient Boosting Classifier acc: ', scores[1].mean())

    print('\nRandomForestClassifier: ', '\n', scores[2])
    print('\nAVG RandomForestClassifier acc: ', scores[2].mean())

'''
****************************************************************************
Let's start Model 2
****************************************************************************

removing noise by selecting features where they are in a 'voting phase'
Note: I need to further subset the specials episode where the time stamps for voting will be different'''


def label_subset_by_season(season_begin, season_end, label_df):
    label_sub_season = label_df[(label_df.Season >= season_begin) & (label_df.Season <= season_end)]
    return label_sub_season


def master_subset_by_season(season_begin, season_end, subset_1):
    # temporarily subset_2 instead of master_df
    subset_2 = subset_1[(subset_1.Season >= season_begin) & (subset_1.Season <= season_end)]
    return subset_2


def label_df_target_variable(label_sub_season):
    s28 = label_sub_season[label_sub_season.Season == 28].copy()
    s28_target_cov = [1, 1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 10, 11, 12, 13, 13, 13, 100]
    s28.loc[:, 'episode_voted_out'] = s28_target_cov  # season 28,  finale has two people, 100 is to be truncated since they don't ever get voted out

    s29 = label_sub_season[label_sub_season.Season == 29].copy()
    s29_target_cov = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 14, 14, 14, 100]
    s29.loc[:, 'episode_voted_out'] = s29_target_cov  # season 29, finale has three people so the last three of it

    s30 = label_sub_season[label_sub_season.Season == 30].copy()
    s30_target_cov = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 14, 14, 14, 100]  # season 29, finale has three people so the last three of it
    s30.loc[:, 'episode_voted_out'] = s30_target_cov  # season 29, finale has three people so the last three of it

    frames = [s28, s29, s30]
    label_df = pd.concat(frames)
    return label_df

# get token episodes from subset by season


def get_ep_tokens(ep_id, master_sub_season):

    subset_episode = master_sub_season[master_sub_season.EpisodeId == ep_id].copy()

    text = ' '.join(subset_episode.subtitles_text)
    lowers = text.lower()
    no_punctuation = re.sub('[\W_]+', ' ', lowers)
    tokenizer = RegexpTokenizer('\s+', gaps=True)
    blob = TextBlob(no_punctuation, tokenizer=tokenizer)

    return blob


def sentiment_score_fn(tokenized_ep, indiv_name, minrange, maxrange):
    '''
    Pulled from http://www.clips.ua.ac.be/pages/pattern-en#sentiment
    Input in tokenized subtitles from an episode, Input in a name, grabs the concordance around the name
    '''
    agg_firstname_concordance_dic = {}
    agg_lastname_concordance_dic = {}
    first_name_temp = []
    last_name_temp = []

    first_name, last_name = indiv_name[0], indiv_name[1]

    first_name_index = [i for i, j in enumerate(tokenized_ep) if j == first_name.lower()]
    last_name_index = [i for i, j in enumerate(tokenized_ep) if j == last_name.lower()]

    for name_loc in first_name_index:
        first_name_temp.append(tokenized_ep[name_loc - minrange:name_loc + maxrange])

    for name_loc in last_name_index:
        last_name_temp.append(tokenized_ep[name_loc - minrange:name_loc + maxrange])

    agg_firstname_concordance_dic[first_name] = first_name_temp
    agg_lastname_concordance_dic[last_name] = last_name_temp

    for x in last_name_temp:
        for y in first_name_temp:

            x1 = ' '.join(x)
            y1 = ' '.join(y)
            dist = Levenshtein.distance(x1, y1)
            if dist > 20:  # low 15
                first_name_temp.append(x)
            else:
                continue

    text = first_name_temp

    agg_sentiment = []
    for i in text:
        combine = ' '.join(i)
        blob = TextBlob(combine)
        sentiment = blob.sentiment.polarity
        agg_sentiment.append(sentiment)

    try:
        avg_sentiment = sum(agg_sentiment) / len(agg_sentiment)
    except ZeroDivisionError:
        avg_sentiment = 0

    # some names like 'so' kim are refercing to the word 'so' instead of name 'so', and will
    first_name_temp = []
    '''
    for name_loc in last_name_index:
        last_name_temp.append(tokenized_ep[name_loc-minrange:name_loc+maxrange])
    agg_lastname_concordance_dic[last_name] = last_name_temp
    return(agg_firstname_concordance_dic)
    '''

    return(avg_sentiment)


def last_concordance(tokenized_ep, indiv_name, minrange, maxrange):

    agg_firstname_concordance_dic = {}
    agg_lastname_concordance_dic = {}
    first_name_temp = []
    last_name_temp = []

    first_name, last_name = indiv_name[0], indiv_name[1]

    first_name_index = [i for i, j in enumerate(tokenized_ep) if j == first_name.lower()]
    last_name_index = [i for i, j in enumerate(tokenized_ep) if j == last_name.lower()]

    for name_loc in first_name_index:
        first_name_temp.append(tokenized_ep[name_loc - minrange:name_loc + maxrange])
    agg_firstname_concordance_dic[first_name] = first_name_temp
    # some names like 'so' kim are refercing to the word 'so' instead of name 'so', and will
    first_name_temp = []

    for name_loc in last_name_index:
        last_name_temp.append(tokenized_ep[name_loc - minrange:name_loc + maxrange])
    agg_lastname_concordance_dic[last_name] = last_name_temp
    return(agg_lastname_concordance_dic)


def concordance(tokenized_ep, indiv_name, minrange, maxrange):

    agg_firstname_concordance_dic = {}
    agg_lastname_concordance_dic = {}
    first_name_temp = []
    last_name_temp = []

    first_name, last_name = indiv_name[0], indiv_name[1]

    first_name_index = [i for i, j in enumerate(tokenized_ep) if j == first_name.lower()]
    last_name_index = [i for i, j in enumerate(tokenized_ep) if j == last_name.lower()]

    for name_loc in first_name_index:
        first_name_temp.append(tokenized_ep[name_loc - minrange:name_loc + maxrange])
    agg_firstname_concordance_dic[first_name] = first_name_temp
    # some names like 'so' kim are refercing to the word 'so' instead of name 'so', and will
    first_name_temp = []

    for name_loc in last_name_index:
        last_name_temp.append(tokenized_ep[name_loc - minrange:name_loc + maxrange])
    agg_lastname_concordance_dic[last_name] = last_name_temp

    '''
    # last name concordance is not inluded in former
    for indiv_name in list_of_names:
        pdb.set_trace()
        first_name, last_name = indiv_name[0], indiv_name[1]
        #print(first_name,last_name)
        first_name_index = [ i for i, j in enumerate(tokenized_ep) if j == first_name.lower()]
        last_name_index = [ i for i, j in enumerate(tokenized_ep) if j == last_name.lower()]

        for name_loc in first_name_index:
            first_name_temp.append(tokenized_ep[name_loc-minrange:name_loc+maxrange])
        agg_firstname_concordance_dic[first_name] = first_name_temp
        #some names like 'so' kim are refercing to the word 'so' instead of name 'so', and will
        #print(agg_firstname_concordance_dic)
        first_name_temp = []

        for name_loc in last_name_index:
            last_name_temp.append(tokenized_ep[name_loc-minrange:name_loc+maxrange])
        agg_lastname_concordance_dic[last_name] = last_name_temp
        last_name_temp = []
    '''
    return(agg_firstname_concordance_dic)  # agg_lastname_concordance_dic


def concordance_packager(label_df, tokens_list):
    firstname_concordance = []
    lastname_concordance = []

    columns = ["season", "episode", "Firstname_concordance", "Lastname_concordance", "sentiment_score"]
    concord_df = pd.DataFrame(columns=columns)

    top_df = pd.DataFrame(columns=["Firstname_concordance"])

    episode_list = [13, 15, 15, 14, 14, 14, 14, 16, 14, 14, 14, 15, 15, 14, 14, 14, 13, 14, 15, 15, 15, 13, 15, 14, 14, 14, 14, 13, 14, 14, 11]
    season_list = [i for i in range(1, 32)]
    episode_ord_dict = collections.OrderedDict()
    for x, y in zip(season_list, episode_list):
        episode_ord_dict[int(x)] = y

    row = []
    aggrow = []
    for season in label_df.Season.unique():
        episode_length = episode_ord_dict[season]

        for episode in range(1, episode_length + 1):
            row.append(episode)
            aggrow.append(season)

    concord_df['season'] = aggrow
    concord_df['episode'] = row

    list_of_names = label_df.Contestant_names.apply(lambda k: k.split(" "))

    for textblob in tokens_list:  # tokens_list is a list of tokens. each one is a episode subsetted by time and season
        first_con, last_con = concordance(textblob.tokens, list_of_names, minrange=5, maxrange=5)
        firstname_concordance.append(first_con)
        lastname_concordance.append(last_con)

    concord_df['season'] = aggrow
    concord_df['episode'] = row
    # ******** commented out ******

    concord_df['Firstname_concordance'] = firstname_concordance
    concord_df['Lastname_concordance'] = lastname_concordance

    '''
    frames = [top_df,MiddleDf]
    top_df = pd.concat(frames)
    '''

    concord_df = concord_df[["season", "episode", "Firstname_concordance", "Lastname_concordance", "sentiment_score"]]
    return concord_df


def concordance_sentiment_packager(label_df, tokens_list):
    firstname_concordance = []
    lastname_concordance = []

    columns = ["season", "episode", "Contestant", "Firstname_concordance", "Lastname_concordance", "sentiment_score"]
    concord_df = pd.DataFrame(columns=columns)

    top_df = pd.DataFrame(columns=["Firstname_concordance"])

    episode_list = [13, 15, 15, 14, 14, 14, 14, 16, 14, 14, 14, 15, 15, 14, 14, 14, 13, 14, 15, 15, 15, 13, 15, 14, 14, 14 ,14, 13, 14, 14, 11]
    season_list = [i for i in range(1, 32)]
    episode_ord_dict = collections.OrderedDict()
    for x, y in zip(season_list, episode_list):
        episode_ord_dict[int(x)] = y

    def contestant_index(label_df):
        pass
    '''
    contestant_index = []
    for i in label_df.Season.unique(): #subsets contestants based on season
        subset = label_df.Contestant_names[label_df.Season == i]
        list_of_names = subset.apply(lambda k: k.split(" "))
        #contestant_index = [list_of_names for i in range(1,18)]
        episode_length = 17
        for i in range(1,len(episode_length)+1):
            contestant_index.append(list_of_names)

    '''
    def season_episode_contestant_index(label_df):
        row = []
        aggrow = []
        contestant_index = []
        a = 0
        index = []
        for season in label_df.Season.unique():
            contestants_per_season = label_df.Contestant_names[label_df.Season == season]
            episode_length = episode_ord_dict[season]

            for episode in range(1, episode_length + 1):

                subset_contestant = label_df.Contestant_names[label_df.Season == season]
                list_of_names = label_df.Contestant_names.apply(lambda k: k.split(" "))
                row.extend([episode for i in range(1, len(subset_contestant) + 1)])
                index.extend([a for i in range(1, len(subset_contestant) + 1)])
                aggrow.extend([season for i in range(1, len(subset_contestant) + 1)])

                contestant_index.extend(subset_contestant)
                a += 1
        return (row, aggrow, contestant_index, index)

    episode, season, contestant_index, index = season_episode_contestant_index(label_df)
    concord_df['season'] = season
    concord_df['episode'] = episode
    concord_df['Contestant_names'] = contestant_index
    concord_df['Index'] = index

    concord_df["Contestant"] = concord_df.Contestant_names.apply(lambda k: k.split(" "))

    def search(row):
        return concordance(tokens_list[row.Index].tokens, row.Contestant, minrange=5, maxrange=5)

    concord_df["Firstname_concordance"] = concord_df.apply(search, axis=1)

    def searchl(row):
        return last_concordance(tokens_list[row.Index].tokens, row.Contestant, minrange=5, maxrange=5)

    concord_df["Lastname_concordance"] = concord_df.apply(searchl, axis=1)

    def sent_search(row):

        return sentiment_score_fn(tokens_list[row.Index].tokens, row.Contestant, minrange=5, maxrange=5)
    concord_df["sentiment_score"] = concord_df.apply(sent_search, axis=1)

    def merge_col(row):
        # for seasons 28,29,30 merge happens at 6,7,6 respectively

        if (row.season == 28) & (row.episode < 6): return "before_merge"
        elif (row.season == 29) & (row.episode < 7): return "before_merge"
        elif (row.season == 30) & (row.episode < 7): return "before_merge"
        else: return "after_merge"
    concord_df["Merge"] = concord_df.apply(merge_col, axis=1)

    '''
    specification: I want to add a target variable 1 if voted out, 0 otherwise
     for each episode in concord_df I want to find the corresponding episode in label_df
     and find who was voted out
    '''
    # label_df.Contestant_names = label_df.Contestant_names.apply(lambda k: str(k))

    concord_df = pd.merge(concord_df, label_df[["Contestant_names", "episode_voted_out"]], how='left', on="Contestant_names")
    '''
    #sns.stripplot(x="episode",y="sentiment_score",data = concord_df)
    #sns.stripplot(x="episode",y="sentiment_score",data = concord_df[concord_df.season == 28])
    #sns.stripplot(x="episode",y="sentiment_score",data = concord_df[concord_df.season == 29])

    #nested categorial feature of voted out or not
    sns.stripplot(x="episode",y="sentiment_score",hue="target",data = concord_df)

    #sns.plt.show()
    '''

    def target_col(row):
        if row.episode == row.episode_voted_out:
            return 1
        else:
            return 0

    concord_df["target"] = concord_df.apply(target_col, axis=1)

    concord_df = concord_df[["season", "episode", "Contestant_names", "Firstname_concordance", "Lastname_concordance", "sentiment_score", "episode_voted_out", "Merge"]]

    return concord_df


def feature_df_index(label_df):
    row = []

    episode_list = [13, 15, 15, 14, 14, 14, 14, 16, 14, 14, 14, 15, 15, 14, 14, 14, 13, 14, 15, 15, 15, 13, 15, 14, 14, 14, 14, 13, 14, 14, 11]
    season_list = [i for i in range(1, 32)]
    episode_ord_dict = collections.OrderedDict()
    for x, y in zip(season_list, episode_list):
        episode_ord_dict[int(x)] = y

    for season in label_df.Season.unique():
        contestants_per_season = label_df.Contestant_names[label_df.Season == season]
        episode_length = episode_ord_dict[season]

        for episode in range(1, episode_length + 1):
            for name in contestants_per_season:
                row.append([name, episode])
    return(row)


def main():

    master_df = cleaning_masterdf()
    label_df = cleaning_labels()

    #changing columns to datetime type
    master_df.end = master_df.end.apply(sub_datetime)
    master_df.begin = master_df.begin.apply(sub_datetime)

    #getting rid of contestents in new season
    # ****remove this bit later on when updated new data*****
    label_df = label_df[:550]

    #clean some various columns
    label_df = cleaning_finish_col(label_df)
    label_df = cleaning_indiv_col(label_df)
    label_df = cleaning_tribal_col(label_df)
    label_df = cleaning_season_label(label_df)

    # model 1:
    # fit_model_1(label_df)

    # model 2:(WIP)

    # remove noise

    # ******removed timestamps, add back later*******

    subset_1 = master_df[(master_df.begin > sub_datetime("00:00:00")) & (master_df.end < sub_datetime("00:36:14")) ]

    label_df = label_df[["Contestant_names", "Finish", "Season"]]

    # subset of data to work on at first
    label_sub_season = label_subset_by_season(28, 30, label_df)
    master_sub_season = master_subset_by_season(28, 30, subset_1)

    label_df = label_df_target_variable(label_sub_season)
    ep_unique = sorted(master_sub_season.EpisodeId.unique())

    tokens_list = []

    for ep_id in ep_unique:
        tokens_list.append(get_ep_tokens(ep_id, master_sub_season))

    rows = feature_df_index(label_df)

    '''
    model 3: sentiment score
    '''
    # concord_df = concordance_packager(label_df,tokens_list)
    sentiment_df = concordance_sentiment_packager(label_df,tokens_list)

    # ranked sentiment scores
    # sentiment_df[sentiment_df.episode == 1].sort('sentiment_score',ascending = True)
    file_object = open("pickle_sentiment_df", 'wb')
    pickle.dump(sentiment_df, file_object)
    file_object.close()


main()
