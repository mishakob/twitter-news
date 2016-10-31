from twython import Twython
import pandas as pd
import unicodecsv as csv
import time
from datetime import datetime
from dateutil import *
from dateutil.relativedelta import *
import glob
import os
import re
import operator
import sys
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity 
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn import metrics
import string
import numpy as np

handles_list = ["nytimes", "TheSun", "thetimes",
				 "AP", "CNN", "BBCNews", "CNET", "msnuk",
				"Telegraph", "USATODAY", "WSJ", "washingtonpost", "BostonGlobe", "newscomauHQ",
				"SkyNews", "SFGate", "AJEnglish", "Independent", "guardian", "latimes", 
				"ReutersAgency", "ABC", "BW", "TIME", "business"]

######### CHANGE HERE
num_files = 6 # max number of files to read (6 = 1 hour worth)
num_tweets = 100 # number of tweets sufficient to begin analysis
keyword = 'BREAKING' # keyword for breaking news
cluster_method = 'AP' # clustering method ('AP' = affinity propagation, 'KM' = KMeans, 'SC' = spectral clustering)
damping = 0.75 # affinity propagation damping parameter
n_clusters = 25 # number of clusters (for KM or SC clustering)
threshold = 4 # count of unique agencies in a cluster to be considered
retweet_threshold = 0.3 # retweet per second ratio threshold
max_results = 5 # number of results to send per day
###################################################
###################################################
# Information loading from previous runs uses filename structure: tweets.batch_id.maxID.csv
# batch_id refers to completed batches of downloaded tweets
# maxID refers to twitter id - used as a parameter for getting new tweets
###################################################

# set maxID, batch_id and sent_result to 0 for the 1st run
tweet_ids = []
for name in glob.glob('tweets.*.csv'):
	tweet_ids.append(int(name.split(".")[2]))
if tweet_ids:
	maxID = max(tweet_ids)
else:
	maxID = 0

file_numbers = []
for name in glob.glob('tweets.*.csv'):
	file_numbers.append(int(name.split(".")[1]))
if file_numbers:
	batch_id = max(file_numbers) +1
else:
	batch_id = 0

###################################################
# sent_result variable keeps track of number of tweets sent so far
sent_result = 0
for name in glob.glob('results[0-9].csv'):
	sent_result +=1
###################################################
# get tweets function
def get_tweets(handle,writer,maxID):
	with open('keys.txt') as k:
		CONSUMER_KEY = k.readline().rstrip()
		CONSUMER_SECRET = k.readline().rstrip()
		ACCESS_KEY = k.readline().rstrip()
		ACCESS_SECRET = k.readline().rstrip()
	twitter = Twython(CONSUMER_KEY,CONSUMER_SECRET,ACCESS_KEY,ACCESS_SECRET)
	
	if maxID == 0:
		tweet_list = twitter.get_user_timeline(screen_name = handle, count=1) # get first tweets
	else:
		tweet_list = twitter.get_user_timeline(screen_name = handle, since_id = maxID)
	
	if tweet_list:
		tweets = []
		for num in range(0,len(tweet_list)):
			current_tweet = []
			current_tweet.append(tweet_list[num].get('id'))
			current_tweet.append(tweet_list[num].get('created_at'))
			current_tweet.append(str(datetime.now()))
			current_tweet.append(tweet_list[num].get('user').get('name'))
			current_tweet.append(tweet_list[num].get('text'))
			urls = tweet_list[num].get('entities').get('urls')
			if urls:
				url = urls[0].get('url')
			else:
				url = ''
			current_tweet.append(url)
			current_tweet.append(tweet_list[num].get('retweet_count'))
			tweets.append(current_tweet)
		maxID = max([tweet.get('id') for tweet in tweet_list])
		writer.writerows(tweets)
		print "Fetched %d tweets from %s" % (len(tweets), handle)
	else:
		print "No tweets in %s" % handle
	return maxID

###################################################
# write tweets function
def write_tweets(maxID, batch_id):
	maxIDs = []
	if __name__ == '__main__':
		with open('tweets.' + str(batch_id) + '.' + str(maxID) + '.csv' , 'w') as f_all:
			writer = csv.writer(f_all)
			writer.writerow(["id","created_at","downloaded_at","agency","text","url","retweet_count"])
			for handle in handles_list:
				maxIDs.append(get_tweets(handle, writer, maxID))
			print 'Done writing tweets'
		os.rename('tweets.' + str(batch_id) + '.' +str(maxID)+ '.csv', 
			'tweets.' + str(batch_id) + '.' +str(max(maxIDs))+ '.csv')
	return max(maxIDs)

###################################################
#################### Analyzing tweets
tknzr = TweetTokenizer()
def tokenize(text):
	text = re.sub(r'\b-\b', " ", text) # removing hyphens
	text = re.sub(r'\'s\b', '', text) # removing "'s"
	text = re.sub(r'((JUST IN)|BBCBreaking|SkyNewsBreak|cnnbrk)', "BREAKING", text)
	text = "".join([ch for ch in text if ch not in string.punctuation])
	tokens = tknzr.tokenize(text)
	tokens = [word for word in tokens if word.lower() not in stopwords.words('english')]
	tokens = [word for word in tokens if word.lower() not in ['rt','via','ajenews','ap']]
	return tokens

### make dataframe from 3 last batches (30 min)
# get last batch number
def analyze_tweets(sent_result):
	global n_clusters, damping, num_files, keyword, cluster_method, threshold, retweet_threshold, max_results
	file_numbers = []
	for name in glob.glob('tweets.*.csv'):
		file_numbers.append(int(name.split(".")[1]))
	last_batch = max(file_numbers)

	# reading files
	batch_nums = [int(name.split(".")[1]) for name in glob.glob('tweets.*[0-9].*.csv') if int(name.split(".")[1]) <= last_batch]
	last_tweets = []
	for batchnum in sorted(batch_nums, reverse=True):
		currentfile = glob.glob('tweets.'+str(batchnum)+'.*.csv')[0]
		last_tweets.append(currentfile)
		if len(last_tweets) == 1:
			df_tweets = pd.read_csv(currentfile)
		else:
			df_tweets = pd.concat([df_tweets, pd.read_csv(currentfile)], ignore_index=True)
		if len(df_tweets) >= num_tweets or len(last_tweets) >= num_files: break	

	if len(df_tweets) >= num_tweets or len(last_tweets) >= num_files:
		print 'processing %d tweets' %len(df_tweets)
		clean_df = df_tweets.copy()

		### this section calculates retweet ratio per second
		# by getting time difference between GMT ('Twitter' time) and PDT, 
		# and between tweet creation and tweet downloading
		clean_df['time_created'] = clean_df['created_at'].apply(parser.parse, ignoretz=True)
		clean_df['time_created_local'] = clean_df['time_created'].apply(lambda x: x-relativedelta(hours=+7))
		clean_df['downloaded_at'] = clean_df['downloaded_at'].apply(parser.parse, ignoretz=True)
		clean_df['delta'] = clean_df['downloaded_at'] - clean_df['time_created_local']
		clean_df['delta'] = clean_df['delta'].apply(lambda x: x.total_seconds())
		clean_df['retweet_ratio'] = clean_df['retweet_count'] / clean_df['delta']

		# if previous result files exist, make sure sent tweets are in current dataset
		result_id = []
		for name in glob.glob('results[0-9].csv'):
			result_df = pd.read_csv(name)
			result_id.append(int(result_df['id'][0]))
			if int(result_df['id'][0]) not in clean_df['id'].tolist():
				# add tweet to dataset
				clean_df = pd.concat([result_df,clean_df])

		### preprocessing
		clean_df['text'] = clean_df['text'].apply(lambda x: re.sub(r'(\s?https?:\S+\s?)', '', x)) # removing urls
		clean_df['utext'] = clean_df['text'].apply(lambda x: unicode(x, 'utf-8')) # convert to unicode
		clean_df['utext'] = clean_df['utext'].apply(lambda x: re.sub(r'[^\x00-\x7F]+', "", x)) # remove non-ascii characters
		clean_df = clean_df[clean_df['utext'].map(len) > 0].reset_index() # remove empty rows

		### tokenizing
		clean_df['tokens'] = clean_df['utext'].apply(tokenize)

		# tfidf and cosine similarity
		tfidf = TfidfVectorizer()
		text = [' '.join(i) for i in clean_df['tokens']]
		vectors = tfidf.fit_transform(text)
		cosine_matrix = cosine_similarity(vectors)

		if cluster_method == 'AP':
			# clustering: affinity propagation
			af = AffinityPropagation(affinity='precomputed', damping=damping).fit(cosine_matrix)
			cluster_centers_indices = af.cluster_centers_indices_
			labels = af.labels_
			n_clusters = len(cluster_centers_indices)
			print 'found %d clusters' %n_clusters
			clean_df['cluster'] = labels

		if cluster_method == 'KM':
			# clustering: KMeans
			km = KMeans(n_clusters = n_clusters).fit(cosine_matrix)
			cluster_centers = km.cluster_centers_
			labels = km.labels_
			clean_df['cluster'] = labels
			n_clusters = len(cluster_centers)
			clean_df['cluster'] = labels

		if cluster_method == 'SC':
			# clustering: spectral clustering
			sc = SpectralClustering(affinity = 'precomputed', n_clusters = n_clusters).fit(cosine_matrix)
			labels = sc.labels_
			clean_df['cluster'] = labels

		# remove previously sent clusters, if any
		result_df = clean_df[clean_df['id'].isin(result_id)]
		result_cluster = result_df['cluster'].tolist()
		clean_df = clean_df[~clean_df['cluster'].isin(result_cluster)]

		### choosing cluster
		chosen_cluster = None
		big_clusters = []
		breaking_clusters = []
		high_retweet_clusters = []

		#######################################################
		### this section deals with cluster and tweet selection
		### the steps are as follows:
		### First, identify clusters that contain a default of 5 unique agencies (report diversity)
		### Among these, select only clusters in which the retweet ratio is relatively high 
		### (calculated by averaging the top 3 retweeted tweets), the default is 0.2 tweets per second
		### Among these, choose the cluster with highest retweet ratio *
		### Finally, select the tweet with highest retweet ratio
		### * in case there are worthy clusters among the discarded ones, those will likely be selected in subsequent runs

		# 1. Count distinct agencies in a cluster
		cluster_users = clean_df.groupby('cluster').agency.nunique().reset_index()
		max_users_cluster = max(cluster_users['agency'])
		print 'max unique agencies in a cluster: %d' %max_users_cluster
		big_clusters = cluster_users['cluster'][cluster_users['agency'] >= threshold].tolist()
		df_chosen = clean_df[clean_df['cluster'].isin(big_clusters)]
		if big_clusters:
			print 'identified %d clusters with more than %d unique agencies' %(len(big_clusters),threshold)
			# 2. At least 1 instances of "BREAKING"
			for cluster in big_clusters:
				for entry in df_chosen[df_chosen['cluster'] == cluster]['tokens']:
					if keyword in ' '.join(entry):
						if cluster not in breaking_clusters:
							breaking_clusters.append(cluster)
			if breaking_clusters:
				print 'identified %d breaking-news clusters' % len(breaking_clusters)
				df_chosen = df_chosen[df_chosen['cluster'].isin(breaking_clusters)]
				# 3. Retweet_ratio is high
				for cluster in breaking_clusters:
					if np.mean(df_chosen[df_chosen['cluster'] == cluster]
						.sort_values('retweet_ratio', ascending = False)[:3]['retweet_ratio']) >= retweet_threshold:
						high_retweet_clusters.append(cluster)
				if high_retweet_clusters:
					df_chosen = df_chosen[df_chosen['cluster'].isin(high_retweet_clusters)]
					print 'identified %d clusters with high retweet ratio' % len(high_retweet_clusters)
					average_ratios = []
					for cluster in high_retweet_clusters:
						average = np.mean(df_chosen[df_chosen['cluster'] == cluster]
							.sort_values('retweet_ratio', ascending = False)[:3]['retweet_ratio'])
						average_ratios.append(average)
					print 'retweet ratios (top 3 tweets):', average_ratios
					# choose cluster with highest retweet ratio
					chosen_cluster = int(df_chosen.sort_values('retweet_ratio', ascending = False)[:1]['cluster'])
					df_chosen = df_chosen[df_chosen['cluster'] == chosen_cluster]
					# choosing best tweet
					df_chosen = df_chosen.sort_values('retweet_ratio', ascending = False)[:1]
					result = df_chosen[['id','text','url','agency','created_at']][df_chosen['retweet_ratio'] == max(df_chosen['retweet_ratio'])]
					sent_result +=1
					result_name = 'results' +str(sent_result)
					result.to_csv(result_name +'.csv')
					for entry in result['text']:
						print 'selected tweet: %s' % (entry)
				else: print 'no clusters with high enough retweet ratio identified'
			else: print 'no breaking-news clusters identified'
		else: print 'no large enough clusters identified'
	return sent_result
while True:
	print "Starting batch %d: fetching tweets for maxID %d" % (batch_id, maxID)
	maxID = write_tweets(maxID, batch_id)
	batch_id += 1
	#if sent_result < max_results: 
	sent_result = analyze_tweets(sent_result)
	print "Going to sleep..."
	time.sleep(600)