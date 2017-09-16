from numpy.random import choice
from nltk import RegexpTokenizer

filename = open('D:\PythonProjects\SentimentAnalysis\input\pos.txt',"r").read()
tokenizer = RegexpTokenizer(r'\w+')
tokens = tokenizer.tokenize(filename.lower())

unigram_frequency = {}
for token in tokens:
    if token in unigram_frequency:
        unigram_frequency[token] += 1
    else:
        unigram_frequency[token] = 1
total_tokens=sum(unigram_frequency.values())
unigram_count = dict(unigram_frequency)

def probability_for_unigram():
    ## Updating the frequencies to corresponding probability of occurence
    for word in unigram_frequency:
        unigram_frequency[word] = unigram_frequency[word] / float(total_tokens)
    return unigram_frequency

def generate_unigram_random_sentence(length_of_the_output_sentence,start_word):
    probabilities = probability_for_unigram()
    text = (" ".join(choice(list(probabilities.keys()), length_of_the_output_sentence, probabilities.values())))
    return start_word + " " +text

def probability_for_bigrams(tokens):
    index = 0
    Bigrams = {}
    for index, token in enumerate(tokens):
        if index < len(tokens)-1:
            w1 = tokens[index + 1]
            w2 = tokens[index]
            bigram = (w1, w2)

            if bigram in Bigrams:
                Bigrams[bigram] = Bigrams[bigram] + 1
            else:
                Bigrams[bigram] = 1

    for ((x, y), value) in Bigrams.items():
        Bigrams[(x, y)] = value / float(unigram_count[y])
    return Bigrams

def nested_bigram(bigrams,start_word):
    x_bigrams = {}
    for ((x, start_word), value) in bigrams.items():
        x_bigrams[x]= value
    return  x_bigrams


def generate_bigrams_random_sentence(length_sentence,start_word):
    list_words = [start_word]
    if( length_sentence > 0):
        bigrams = probability_for_bigrams(tokens)
        i = 0
        next_word = start_word
        for i in range(length_sentence):
            i += 1
            x_bigrams = nested_bigram(bigrams, next_word)
            next_word = "".join(choice(list(x_bigrams.keys()), 1, x_bigrams.values()))
            list_words.append(next_word)
    return  list_words

def generate_bigrams_random_sentence_without_seed(length_sentence):
    import random
    bigrams = probability_for_bigrams(tokens)
    seeding_word = random.choice(list(bigrams.keys()))
    list_words = [seeding_word]
    if (length_sentence > 0):
        bigrams = probability_for_bigrams(tokens)
        i = 0
        next_word = seeding_word
        for i in range(length_sentence):
            i += 1
            x_bigrams = nested_bigram(bigrams, next_word)
            next_word = "".join(choice(list(x_bigrams.keys()), 1, x_bigrams.values()))
            list_words.append(next_word)
    return list_words

unigram_without_seeding =generate_unigram_random_sentence(5," ")
print("Unigram without seeding:" +unigram_without_seeding)

unigram_with_seeding =generate_unigram_random_sentence(5,"the")
print("Unigram with seeding:" +unigram_with_seeding)

bigram_sentence_with_seeding = generate_bigrams_random_sentence(5,"with")
print(bigram_sentence_with_seeding)

bigram_sentence_without_seeding = generate_bigrams_random_sentence_without_seed(5)
print( bigram_sentence_without_seeding)



