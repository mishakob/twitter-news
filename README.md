# twitter-news
Hot news aggregator based on tweets from leading news agencies  
Tool output is (only) important & breaking news  
  
The tool works as follows:  
1. Connects to Twitter API and downloads the last tweets from a list of Twitter handles for the last 10 minutes.  
2. Builds a dataframe out of the last 100 tweets (or the last hour, if there weren't enough tweets).  
3. Preprocessing: tokenizing, dealing with time and date, calculating retweets-per-second ratio.  
4. Runs TFIDF on the tokenized tweet texts.  
5. Calculates similarity matrix (cosine similarity)  
6. Clusters tweets (the default is Affinity Propagation, Kmeans and Spectral Clustering are other options)  
7. Checks whether there are potential "hot news" clusters, using the following criteria:  
 - diversity in reporting agencies
 - breaking news (according to keywords)
 - high enough retweet ratio
