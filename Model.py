#token = word
#ngram =character or word
def main():
	'''
	Before Model:
	1. Clean up dataset to fit this format:
		SentenceId,EpisodeId,Season,Episode,begin,endSentence
		https://raw.githubusercontent.com/mneedham/neo4j-himym/master/data/import/sentences.csv
		******#PROBLEM: makes an assumption that the dictionary is in order, however it isn't****
		
	2. Have a list of labels: who was voted out each episode

	During Model:
	1. Feed in list of names of survivor contestents
	2. For each episode look at the ngrams around each 'name'
		2.1
	3. gather the sentiment of positive and negative words around the 'name'
		3.1 I could hand code it and look up the word and its corresponding positive or negative value
			then average the values

	4. can cross val by taking out some episodes
	
	Ways of improving Model(scores):
	5. Look at high information features instead of using all the words as features
		http://streamhacker.com/2010/06/16/text-classification-sentiment-analysis-eliminate-low-information-features/

	****random forest
	base classifiers n_estimators = several hundred up to 1000
	njobs =-1
	records = episode
	features = ngrams start with ngrams =5 then move up
	target variable = bucket their where they placed

	'''
main()