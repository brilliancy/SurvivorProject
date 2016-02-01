# SurvivorProject
This is an attempt to apply some ML and NLP to the tv shower from CBS, survivor. I do not own any of the materials. It belongs to CBS.
This is still a work in progress and the code is very messy and needs to be cleaned up. I will work on cleaning it up sometime in the future.

Here is a list of files and their description:
Data Preprocessing Folder:
* xml_scrape_loop.py = the webscraper for the subtitles
* test_combine.py = combines the csv's into one large dataframe, and cleans it up a little bit.
* label_scraper.py = scrapes labels/target variable for the dataset from wikia
* season_scraper.py = scrapes wikipedia to map season name into season number for label_scraper.py
GADS_Project Folder:
* engine.py = loops through xml_scrape_loop and handles any exceptions that arise. Returns list of csv's with subtitles
* Model.py = contains a lot of feature engineering and the models. (model 2 & 3 are seasons 28-30 subsetted data from masterDf)
  * Model 1: Estimators is individual/tribal challenge, estimand is how they finished, bucketed into 3 categories.
  * Model 2: Contestant names in relation to each other (WIP)
  * Model 3: Grab a concordance around a contestant name then average the sentiment score for the concordance. Sentiment scores
* EDA.py = some exploratory data analysis of MasterDf
  * Exploratory_Data_Analysis.md =  exploratory data analysis in a prettier format