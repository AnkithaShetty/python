from numpy.random import choice
from nltk import RegexpTokenizer,word_tokenize,sent_tokenize


def fancy_tokenizer(file_path):
    import string
    file_stream = open(file_path, "r").read()
    #tokenizer = RegexpTokenizer(r'\w+')
    tokens = sent_tokenize(file_stream.lower())
    #tokens = word_tokenize(file_stream.lower())
    #x = [''.join(c for c in s if c not in string.punctuation) for s in x]
    print(tokens)
    print("stripping punctuation")
    without_punc_tokens = [''.join(sentence for sentence in token if sentence not in string.punctuation) for token in tokens]
    new_tokens = ' '.join('<str>' + item.lstrip().rstrip() + '<end>' for item in without_punc_tokens)
    test = new_tokens.split(" ")
    print(test)
    #without_punc_tokens = [sentence for sentence in tokens if sentence not in string.punctuation]
    #print(without_punc_tokens)
    return tokens


def default_tokenizer(file_path):
    file_stream = open(file_path, "r").read()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(file_stream.lower())
    return tokens


def ngram_frequency_builder(tokens):
    unigram_frequency = {}
    for token in tokens:
        if token in unigram_frequency:
            unigram_frequency[token] += 1
        else:
            unigram_frequency[token] = 1

    #new_dict = {key:val for key,val in unigram_frequency.items() if len(key) == 1}
    #print(new_dict)

    total_tokens = sum(unigram_frequency.values())
    print(total_tokens)
    return unigram_frequency, total_tokens


def probability_for_unigram(unigram_frequency, total_tokens):
    # Updating the frequencies to corresponding probability of occurence
    for word in unigram_frequency:
        unigram_frequency[word] = unigram_frequency[word] / float(total_tokens)
    return unigram_frequency


def generate_unigram_random_sentence(sentence_length, start_word, unigram_frequency, total_tokens):
    probabilities = probability_for_unigram(unigram_frequency, total_tokens)
    text = (" ".join(choice(list(probabilities.keys()), sentence_length, probabilities.values())))
    if start_word is None:
        return text
    else:
        return start_word + " " + text


def probability_for_bigrams(tokens, unigram_count):
    bigrams = {}
    for index, token in enumerate(tokens):
        if index < len(tokens) -1:
            w1 = tokens[index + 1]
            w2 = tokens[index]
            bigram = (w1, w2)

            if bigram in bigrams:
                bigrams[bigram] = bigrams[bigram] + 1
            else:
                bigrams[bigram] = 1
    bigrams_count = len(bigrams)
    bigram_frequency = dict(bigrams)
    for ((x, y), value) in bigrams.items():
        bigrams[(x, y)] = value / float(unigram_count[y])
    return bigrams, bigrams_count, bigram_frequency


def nested_bigram(bigrams, start_word):
    x_bigrams = {}
    bigram_dict = {(x1, y1): value for (x1, y1), value in bigrams.items() if y1 == start_word}
    for ((x, y), value) in bigram_dict.items():
        x_bigrams[x] = value
    return x_bigrams


def generate_bigrams_random_sentence(sentence_length, start_word, tokens, unigram_count):
    import random
    bigrams, _, _ = probability_for_bigrams(tokens, unigram_count)
    if start_word is None:
        start_word = random.choice(list(bigrams.keys()))[0]
    list_words = [start_word]
    if sentence_length > 0:
        next_word = start_word
        for i in range(sentence_length - 1):
            i += 1
            x_bigrams = nested_bigram(bigrams, next_word)
            next_word = max(x_bigrams, key=x_bigrams.get)
            list_words.append(next_word)
    return " ".join(list_words)


def handle_unknowns(unigram_frequency, held_out_tokens):
    unigram_frequency["UNK"] = 0
    for token in held_out_tokens:
        if token not in unigram_frequency:
            unigram_frequency["UNK"] +=1
        else:
            unigram_frequency[token] +=1

    total_tokens = sum(unigram_frequency.values())
    unigrams_count = dict(unigram_frequency)
    #retraining with unknowns
    for word in unigram_frequency:
        unigram_frequency[word] = unigram_frequency[word] / float(total_tokens)
    print(unigram_frequency["UNK"])
    return unigram_frequency, unigrams_count


def get_GoodTuring_counts(unigram_frequency, tokens):
    total_unigrams = len(unigram_frequency)
    total_bigrams = total_unigrams * total_unigrams
    bigrams, bigrams_count, bigrams_frequency = probability_for_bigrams(tokens, unigram_frequency)
    bigrams_unseen_count = total_bigrams - bigrams_count
    #print("bigrams_unseen_count " + str(bigrams_unseen_count))

    bigrams_count_list = [bigrams_unseen_count, 0, 0, 0, 0, 0]
    for key, value in bigrams_frequency.items():
        if value < 6:
            bigrams_count_list[value] += 1

    #print(bigrams_count_list)
    goodTuring_counts = [0, 0, 0, 0, 0]
    for x in range(0, 5):
        goodTuring_counts[x] = (x+1) * (bigrams_count_list[x+1] * 1.0000) / (bigrams_count_list[x] * 1.0000)

    print(goodTuring_counts)
    return goodTuring_counts


def get_Good_Turing_probability(w1, w2, unigram_frequency, tokens, GT_counts):
    adj_count = 0
    bigrams, bigrams_count, bigrams_frequency = probability_for_bigrams(tokens, unigram_frequency)
    if (w1, w2) in bigrams.keys():
        adj_count = bigrams_frequency[(w1, w2)]
    if adj_count < 5:
        adj_count = GT_counts[adj_count]
    if w1 in unigram_frequency.keys():
        t = adj_count / (unigram_frequency[w1] * 1.0)
        #print(t)
    else:
        t = adj_count / (unigram_frequency["UNK"] * 1.0)
        #print(t)
    return t
    #get_Good_Turing_probability("Ankita", "Nayan", unigram_frequency_pos_unk,tokens_pos,GT_counts)


def calculate_perplexity(test_token, unigrams_count, tokens, GT_count):
    import math
    #print(unigrams_count[test_token[0]])
    total_tokens = len(tokens)
    len_token = len(test_token)
    for i in range(0, len_token, 1):
        if test_token[i] not in unigrams_count:
            test_token[i] = u'UNK'

        sum = math.log(unigrams_count[test_token[0]] * 1.0 / total_tokens) * (-1.0)
        for x in range(0, len_token - 1):
            prob = get_Good_Turing_probability(test_token[x], test_token[x + 1], unigrams_count, tokens, GT_count)
            sum += ((math.log(prob)) * (-1))

        return math.exp((sum / len_token))

def main():
    file_path_training_pos = "D:\PythonProjects\SentimentAnalysis\input\Train\pos.txt"
    #file_path_training_neg = "D:\PythonProjects\SentimentAnalysis\input\Train\\neg.txt"
    tokens_pos = default_tokenizer(file_path_training_pos)
    #tokens_neg = default_tokenizer(file_path_training_neg)

    unigram_frequency_pos, total_tokens_pos = ngram_frequency_builder(tokens_pos)
    #print(len(unigram_frequency_pos))

    #unigram_frequency_neg, total_tokens_neg = ngram_frequency_builder(tokens_neg)

    #unigram_count_pos = dict(unigram_frequency_pos)
    #unigram_count_neg = dict(unigram_frequency_neg)


    #unigram_without_seeding = generate_unigram_random_sentence(5, None, unigram_frequency_pos, total_tokens_pos)
    #print("Unigram without seeding  for postive corpus-> " + unigram_without_seeding)
    """
    unigram_without_seeding = generate_unigram_random_sentence(5, None, unigram_frequency_neg, total_tokens_neg)
    print("Unigram without seeding  for negative corpus-> " + unigram_without_seeding)

    unigram_with_seeding = generate_unigram_random_sentence(5, "with", unigram_frequency_pos, total_tokens_pos)
    print("Unigram with seeding for positive corpus-> " + unigram_with_seeding)

    unigram_with_seeding = generate_unigram_random_sentence(5, "with", unigram_frequency_neg, total_tokens_neg)
    print("Unigram with seeding for negative corpus-> " + unigram_with_seeding)

    bigram_sentence_with_seeding = generate_bigrams_random_sentence(5, "purely", tokens_pos, unigram_frequency_pos)
    print("Bigram with seeding for positive corpus- > " + bigram_sentence_with_seeding)

    bigram_sentence_with_seeding = generate_bigrams_random_sentence(5, "heist", tokens_neg, unigram_frequency_neg)
    print("Bigram with seeding for negative corpus- > " + bigram_sentence_with_seeding)

    bigram_sentence_without_seeding = generate_bigrams_random_sentence(5, None, tokens_pos, unigram_frequency_pos)
    print("Bigram without seeding for positive corpus -> " + bigram_sentence_without_seeding)

    bigram_sentence_without_seeding = generate_bigrams_random_sentence(5, None, tokens_neg, unigram_frequency_neg)
    print("Bigram without seeding for negative corpus-> " + bigram_sentence_without_seeding)
    """

    file_path_held_out_pos = "D:\PythonProjects\SentimentAnalysis\input\Dev\pos.txt"
    #file_path_held_out_neg = "D:\PythonProjects\SentimentAnalysis\input\Dev\\neg.txt"

    held_out_tokens_pos = default_tokenizer(file_path_held_out_pos)
    #held_out_tokens_neg = default_tokenizer(file_path_held_out_neg)

    unigram_frequency_pos_unk, unigram_count_pos_unk = handle_unknowns(unigram_frequency_pos, held_out_tokens_pos)
    #unigram_frequency_neg_unk = handle_unknowns(unigram_frequency _neg, held_out_tokens_neg)

    #unigram_frequency_pos_unk, total_tokens_pos = ngram_frequency_builder(tokens_pos)
    #print(len(unigram_frequency_pos))

    #unigram_frequency_neg_unk, total_tokens_neg = ngram_frequency_builder(tokens_neg)

    #GT_counts = get_GoodTuring_counts(unigram_frequency_pos_unk, tokens_pos)
    #get_Good_Turing_probability("Ankita", "Nayan", unigram_frequency_pos_unk,tokens_pos,GT_counts)

    #test_bigram = ["most","part"]


    """

    print(bigrams_count)

    bigrams, bigrams_count_unk, bigrams_frequency = probability_for_bigrams(tokens_with_unknown, unigram_count_pos_unk)
    print(bigrams_count_unk)
    #perplexity_score = calculate_perplexity(test_bigram, unigram_frequency_pos_unk, tokens_pos, GT_counts)
    #print(perplexity_score)
     """

if __name__ == "__main__":
        main()




