import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from KaggleWord2VecUtility import KaggleWord2VecUtility
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
import nltk
# nltk.download()
from nltk.corpus import stopwords # Import the stop word list




def review_to_words (raw_review): # Data cleaning


    # Remove html tags
    review_text = BeautifulSoup(raw_review).get_text()

    # Remove punctuation and numbers
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)

    lower_case = letters_only.lower()
    words = lower_case.split()

    # In Python a set is faster than a list
    stops = set(stopwords.words("english"))


    # Dealing with stop words:
    meaningful_words = [w for w in words if not w in stops]

    return (" ".join(meaningful_words))


train = pd.read_csv("resources/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)

clean_review = review_to_words(train["review"][0])


# clean all of the training set at once
num_reviews = train["review"].size


clean_train_reviews = []

for i in xrange(0, num_reviews):
    if ((i + 1) % 1000 == 0): # Print a message every 1000 reviews
        print "Review %d of %d\n" % (i + 1, num_reviews)
    clean_train_reviews.append(review_to_words(train["review"][i]))


# Create Features from Bag of Words

#scikit-learn bag of words
# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.

def training(reviews, vectorizer) :


    train_data_features = vectorizer.fit_transform(reviews)
    train_data_features = train_data_features.toarray()

    # Random Forest (100 trees)
    forest = RandomForestClassifier(n_estimators = 100)
    forest = forest.fit( train_data_features, train["sentiment"] )

    return forest


vectorizer = CountVectorizer(analyzer = 'word',
                            tokenizer = None,
                             preprocessor = None,
                             stop_words = None,
                             max_features = 5000)

theForest = training(clean_train_reviews, vectorizer)


# Read the test data
test = pd.read_csv("resources/testData.tsv", header=0, delimiter="\t", \
                   quoting=3 )

# Verify that there are 25,000 rows and 2 columns
print test.shape

# Create an empty list and append the clean reviews one by one
num_reviews = len(test["review"])
clean_test_reviews = []

for i in xrange(0,num_reviews):
    if( (i+1) % 1000 == 0 ):
        print "Review %d of %d\n" % (i+1, num_reviews)
    clean_review = review_to_words( test["review"][i] )
    clean_test_reviews.append( clean_review )

# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

# Use the random forest to make sentiment label predictions
result = theForest.predict(test_data_features)


# Copy the results to a pandas dataframe with an "id" column and
# a "sentiment" column
output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )

# Use pandas to write the comma-separated output file
output.to_csv( "resources/Bag_of_Words_model.csv", index=False, quoting=3 )







