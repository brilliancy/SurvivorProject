# SurvivorProject
This is an attempt to apply some ML and NLP to the tv shower from CBS, survivor. I do not own any of the materials. It belongs to CBS.
This is still a rough draft and the code is very messy and needs to be cleaned up. I will work on cleaning it up sometime in the future.

Here is a list of files and their description:
Data Preprocessing:
* xml_scrape_loop.py = the webscraper for the subtitles
* combine_dataframe.py = combines the csv's into one large dataframe, and cleans it up a little bit.
* label_scraper.py = scrapes labels for the dataset from wikia
* season_scraper.py = scrapes wikipedia to map season name into season number for label_scraper.py
GADS_Project:
* engine.py = loops through xml_scrape_loop and handles any exceptions that arise. Returns list of csv's with subtitles
* Model.py = contains a lot of feature engineering and the model itself
* EDA.py = some exploratory data anlysis of MasterDf
