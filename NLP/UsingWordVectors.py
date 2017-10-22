from gensim.models import Word2Vec
import numpy as np
import pandas as pd
import WordVectors as wordVecs
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import time


model = Word2Vec.load("300features_40minwords_10context")

train = pd.read_csv( "resources/labeledTrainData.tsv", header=0,
 delimiter="\t", quoting=3 )

test = pd.read_csv( "resources/testData.tsv", header=0, delimiter="\t", quoting=3 )

def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")

    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.index2word)

    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    nwords = 0
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec, model[word])


    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec, nwords)
    return featureVec

def getAvgFeatureVecs(reviews, model, num_features):

    # Given a set of reviews (each one a list of words), calculate
    # the average feature vector for each one and return a 2D numpy array
    counter = 0.

    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")

    # Loop through the reviews
    for review in reviews:

        # Print a status message every 1000th review
        if counter % 1000. == 0.:
            print "Review %d of %d" % (counter, len(reviews))

        # Call the function (defined above) that makes average feature vectors
        reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)

        # Increment the counter
        counter = counter + 1.

    return reviewFeatureVecs

# ****************************************************************
# From Words To Paragraphs, Attempt 1: Vector Averaging
# ****************************************************************

clean_train_reviews = []
for review in train["review"]:
    clean_train_reviews.append( wordVecs.review_to_wordlist( review, remove_stopwords=True ))

trainDataVecs = getAvgFeatureVecs( clean_train_reviews, model, wordVecs.num_features )
#
# print "Creating average feature vecs for test reviews"
clean_test_reviews = []
for review in test["review"]:
    clean_test_reviews.append( wordVecs.review_to_wordlist( review, remove_stopwords=True ))

testDataVecs = getAvgFeatureVecs( clean_test_reviews, model, wordVecs.num_features )

# # perhaps we could do it in a more intelligent way?
# # A standard way of weighting word vectors is to apply "tf-idf" weights
# # One way to extract tf-idf weights in Python is by using scikit-learn's TfidfVectorizer
#
# # *** Random forest ***
#
#
# forest = RandomForestClassifier( n_estimators = 100 )
#
# print "Fitting a random forest to labeled training data..."
# forest = forest.fit( trainDataVecs, train["sentiment"] )
#
# # Test & extract results
# result = forest.predict( testDataVecs )
#
# # Write the test results
# output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
# output.to_csv( "Word2Vec_AverageVectors.csv", index=False, quoting=3 )


# ****************************************************************
# From Words to Paragraphs, Attempt 2: Clustering
# ****************************************************************

start = time.time() # Start time

# Set "k" (num_clusters) to be 1/5th of the vocabulary size, or an
# average of 5 words per cluster
word_vectors = model.wv.syn0
num_clusters = word_vectors.shape[0] / 5

# Initalize a k-means object and use it to extract centroids
kmeans_clustering = KMeans( n_clusters = num_clusters )
idx = kmeans_clustering.fit_predict( word_vectors )

# Get the end time and print how long the process took
end = time.time()
elapsed = end - start
print "Time taken for K Means clustering: ", elapsed, "seconds."

# Create a Word / Index dictionary, mapping each vocabulary word to
# a cluster number
word_centroid_map = dict(zip( model.index2word, idx ))

def create_bag_of_centroids( wordlist, word_centroid_map ):

    # The number of clusters is equal to the highest cluster index
    # in the word / centroid map
    num_centroids = max( word_centroid_map.values() ) + 1

    # Pre-allocate the bag of centroids vector (for speed)
    bag_of_centroids = np.zeros( num_centroids, dtype="float32" )

    # Loop over the words in the review. If the word is in the vocabulary,
    # find which cluster it belongs to, and increment that cluster count
    # by one
    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1
    #
    # Return the "bag of centroids"
    return bag_of_centroids

# Pre-allocate an array for the training set bags of centroids (for speed)
train_centroids = np.zeros( (train["review"].size, num_clusters), dtype="float32" )

# Transform the training set reviews into bags of centroids
counter = 0
for review in clean_train_reviews:
    train_centroids[counter] = create_bag_of_centroids( review, word_centroid_map )
    counter += 1

# Repeat for test reviews
test_centroids = np.zeros(( test["review"].size, num_clusters), \
    dtype="float32" )

counter = 0
for review in clean_test_reviews:
    test_centroids[counter] = create_bag_of_centroids( review, \
        word_centroid_map )
    counter += 1

# *** Random forest ***
# Fit a random forest and extract predictions
forest = RandomForestClassifier(n_estimators = 100)

# Fitting the forest may take a few minutes
print "Fitting a random forest to labeled training data..."
forest = forest.fit(train_centroids,train["sentiment"])
result = forest.predict(test_centroids)

# Write the test results
output = pd.DataFrame(data={"id":test["id"], "sentiment":result})
output.to_csv( "BagOfCentroids.csv", index=False, quoting=3 )




