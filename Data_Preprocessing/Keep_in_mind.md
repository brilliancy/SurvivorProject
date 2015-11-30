#Goals to keep in mind
* What is the problem you want to solve?
  * predict who will be voted out each episode
    * labels come from who will be voted out each episode
  * predict how well the contestants will do based on past behavior
    * how far they got into their own season, how well they do on challenges etc.
* What data will you use and where will it from?

* What modeling approach will you use?
  * feature engineering. Have a list of good and bad words from NLTK and see bigrams or ngrams how close it is to someones name and its connotation
    * but i might not have too since the subtitles are already split into different rows so i just look for someones name and the connotation of that line and get a dictionary of somesones name and their connotation score.
  * can create labels by subsetting the last few timestamps where they vote and translate the number of votes into a numeric data type and match it to the name 
    * be careful in comparison to ' i hate that' vs ' i dont hate that'
  * for unsupervised you can do string similairty or tfidf or lda to see what's important
* What exploratory steps do you think will be important?
  * tfidf, clustering, still have ranks so i can make histograms, 
  * fix spelling errors with spellcheck  python levenshten fuzzymuzzy - string similairity if above a certain threshold then change the word
* How will you evaluate your results?
  * How do you know when you've done a good job?
    * F score - harmonic mean of precision and recall
    * accuracy 


* extras 
  * start simple then build complexity