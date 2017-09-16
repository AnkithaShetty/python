from collections import Counter
from nltk.tokenize import RegexpTokenizer
filepath ="D:\PythonProjects\SentimentAnalysis\input\pos.txt"

def tokenizer_fun(fpath):
    filename = open(fpath,"r").read()
    type(filename)
    tokenizer = RegexpTokenizer(r'\w+')
   # tokens_lower = word_tokenize(filename.lower())
    tokens = tokenizer.tokenize(filename.lower())
   # countLower = Counter(tokens_lower)
    print (len(tokens))
    unigrams_count = Counter(tokens)
    print(dict(unigrams_count))

   # print(tokens.count("..."))
   # print(len(filename))
    return tokens

def tokenizer_func(fpath):
    import string
    fileStream = open(fpath,"r").read().lower()
    occurrence = Counter(fileStream.translate(None, string.punctuation).split())
    print(occurrence)


#unigrams_list = tokenizer_func(filepath)


#print(count)

