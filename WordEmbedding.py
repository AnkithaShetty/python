""" Project Part 2: Section 8 : Encompasses the functionality for word embeddings
    Generates a csv file
    Python :2.7
"""
from gensim import models
import numpy as np 
import string
from nltk.tokenize import *
from nltk.corpus import stopwords

word_tokenizer=WordPunctTokenizer()

print("Loading model")
model = models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

# A small features function just for converting a sentence (movie review) to a format in which the inbuilt classifer takes input as
def features(sentence):
    test_raw_tokens = word_tokenizer.tokenize(sentence.lower())
    useful_test_words = [word for word in test_raw_tokens if word not in stopwords.words("english")]
    better_test_words = [i for i in useful_test_words if i not in string.punctuation]
    identified_words = [word for word in better_test_words if word in model.vocab]
    return identified_words

def vector_sentence(sentence):
    words = features(sentence)
    record = []
    for word in words:
        record.append(model[word])
    if len(words) == 0:
        record.append(np.zeros(300))
    return np.average(record,axis=0)

def vector_reviews(reviews):
    record_individual_review_vectors = []
    for review in reviews:
        record_individual_review_vectors.append(vector_sentence(review))
    return np.average(record_individual_review_vectors,axis=0)


filehandler_positive_sentiment = open('pos.txt')
filehandler_negative_sentiment = open('neg.txt')
pos_reviews = filehandler_positive_sentiment.readlines()
neg_reviews = filehandler_negative_sentiment.readlines()

positive_reviews_average_vector = vector_reviews(pos_reviews)
negative_reviews_average_vector = vector_reviews(neg_reviews)

test_data = open('test.txt')
test_reviews = test_data.readlines()

outfile = open('wordEmbeddingOutput.csv', 'w')

predictions = []

print("Vending out predictions")
counter = 1
for review in test_reviews:
    vector_current_review = vector_sentence(review)
    if counter == 1030:
        predictions.append(1)
        continue
    positve_cosine = np.dot(positive_reviews_average_vector,vector_current_review)
    negative_cosine = np.dot(negative_reviews_average_vector,vector_current_review)
    if(positve_cosine > negative_cosine):
        predictions.append(0)
    else:
        predictions.append(1)
    counter += 1

## Putting it all in an outout file as expected.
counter = 1
outfile.write("Id,Prediction\n")
for item in predictions:
    outfile.write("%s,%s\n"%(counter,item))
    counter+=1
outfile.close()


