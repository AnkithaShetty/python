""" Sentimental classifier that uses Naives Bayes Library
 Generates a csv  file "baiyesClassifier.csv" used for kaggle submissions
 Python Version: 2.7
"""

from nltk.classify import NaiveBayesClassifier
from nltk.tokenize import *

# Using the inbuilt punctuation tokenizer.
word_tokenizer=WordPunctTokenizer()

filehandler_positive_sentiment = open('pos.txt')
filehandler_negative_sentiment = open('neg.txt')
# Path of the test data
test_data = open('test.txt')
#Path of the output file
outfile = open('baiyesClassifier.txt', 'w')

pos_reviews = filehandler_positive_sentiment.readlines()
neg_reviews = filehandler_negative_sentiment.readlines()

pos_train = [(sentence,'pos') for sentence in pos_reviews]
neg_train = [(sentence,'neg') for sentence in neg_reviews]
print "Constructed positive and negative tagged data set"

train = pos_train + neg_train

all_words = set(word.lower() for passage in train for word in word_tokenizer.tokenize(passage[0]))
print "Constructed vocabulary"

print "Constructing training_features ... (This will take time)"
training_features = [({word: (word in word_tokenizer.tokenize(x[0])) for word in all_words}, x[1]) for x in train]
print "Constructed training_features"
print "Training the classifier ... (This will take time)" 
classifier = NaiveBayesClassifier.train(training_features)

test_reviews = test_data.readlines()

# A small features function just for converting a sentence (movie review) to a format in which the inbuilt classifer takes input as
def features(sentence):
    return {word: (word in word_tokenizer.tokenize(sentence.lower())) for word in all_words}

print "Vending out predictions for test data ... (This will take time)"
predictions = []
for review in test_reviews:
    result = classifier.classify(features(review))
    if result == 'pos':
        predictions.append(0)
    else:
        predictions.append(1)


## Putting it all in an outout file as expected.
counter=1
outfile.write("Id,Prediction\n")
for item in predictions:
    outfile.write("%s,%s\n"%(counter,item))
    counter+=1
outfile.close()

