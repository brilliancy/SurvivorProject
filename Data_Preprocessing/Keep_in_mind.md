#Goals to keep in mind
* What is the problem you want to solve?
  * predict who will be voted out each episode
    * labels come from who will be voted out each episode
  * predict how well the contestants will do based on past behavior
    * how far they got into their own season, how well they do on challenges etc.
* What data will you use and where will it from?
  * cbs
* What modeling approach will you use?
  * feature engineering. 
    * binary text classification. Have a list of good and bad words from NLTK and see bigrams or ngrams how close it is to someones name and its connotation
    * find survivor specific key words like 'blindsided','immunity idol' and add it to the lexicon subjectivity list
      * some teams might win immunity how does that change the model?
    * might be a dead end: find words that corresponds with NLTK lexicon lists and find/predict which subtitles lines might have more signal. (how to know when I've done well?)
      * then just feed the new list with survivor specific subjectivity lexicon ? might be a recursive loop or am i feature engineering correctly? 
    * but i might not have too since the subtitles are already split into different rows so i just look for someones name and the connotation of that line and get a dictionary of somesones name and their connotation score.
  * can create labels by subsetting the last few timestamps where they vote and translate the number of votes into a numeric data type and match it to the name 
    * be careful in comparison to ' i hate that' vs ' i dont hate that'
  * for unsupervised you can do string similarity or tfidf or lda to see what's important
  * build a parser and feed in a list words that correspond with voting out people or loyalty
* What exploratory steps do you think will be important?
  * tfidf, clustering, still have ranks so i can make histograms, 
  * fix spelling errors with spellcheck  python levenshten fuzzymuzzy - string similairity if above a certain threshold then change the word
* How will you evaluate your results?
  * How do you know when you've done a good job?
    * F score - harmonic mean of precision and recall
    * accuracy 


* extras 
  * start simple then build complexity
  
  
  
* things to do in order
  * clean the rest of the data
    * combined into one large df
    * turn subtitles time into date time library 
    * clean out the >> and unique characters
    * I still have some combined words. Perhaps just add a space after every text before reading it into a csv  
  * EDA
    * tfidf
  * model 
    * feed in list of name and have a fixed ngrams of how many positive or negative words are near that name.
      * need to SCRAPE list of names
    * feed in a suvivor specific list of words that are likely to predict if someone will be voted out
    * subset based on time
      * need to change time to date time format
  * how to know when I did a good job
    * have a list of who was voted out. (NEED TO SCRAPE)
