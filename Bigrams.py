
import string
filepath ="D:\PythonProjects\SentimentAnalysis\input\pos.txt"
filename = open(filepath,"r").read().lower()
import re
bigrams = {}

for line in filename.split('\n'):
    l = re.sub('\.','<e> <s>',line)
    print(l)

words_punct = filename.split()

words = [ w.strip(string.punctuation).lower() for w in words_punct ]
for index, word in enumerate(words):
    if index < len(words) - 1:
        # we only look at indices up to the
        # next-to-last word, as this is
        # the last one at which a bigram starts
        w1 = words[index]
        w2 = words[index + 1]
        # bigram is a tuple,
        # like a list, but fixed.
        # Tuples can be keys in a dictionary
        bigram = (w1, w2)

        if bigram in bigrams:
            bigrams[ bigram ] = bigrams[ bigram ] + 1
        else:
            bigrams[ bigram ] = 1

print(bigrams)