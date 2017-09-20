from nltk import RegexpTokenizer, sent_tokenize

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


def default_tokenizer(file_stream):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(file_stream.lower())
    return tokens


def compute_unigram_frequency(tokens):
    unigram_frequency = {}
    for token in tokens:
        if token in unigram_frequency:
            unigram_frequency[token] += 1
        else:
            unigram_frequency[token] = 1
    return unigram_frequency


def compute_unigram_probability(unigram_frequency):
    unigram_probability = {}
    total_tokens = sum(unigram_frequency.values())
    for word in unigram_frequency:
        unigram_probability[word] = unigram_frequency[word] / float(total_tokens)
    return unigram_probability


def compute_bigram_probability(tokens, unigram_frequency):
    bigram_frequency = {}
    for index, token in enumerate(tokens):
        if index < len(tokens) -1:
            w1 = tokens[index + 1]
            w2 = tokens[index]
            bigram = (w1, w2)

            if bigram in bigram_frequency:
                bigram_frequency[bigram] = bigram_frequency[bigram] + 1
            else:
                bigram_frequency[bigram] = 1

    #bigram_count = len(bigram_frequency)
    bigram_probability = dict(bigram_frequency)
    for ((x, y), value) in bigram_probability.items():
        bigram_probability[(x, y)] = value / float(unigram_frequency[y])

    return bigram_probability, bigram_frequency


def nested_bigram(bigrams, start_word):
    x_bigrams = {}
    bigram_dict = {(x1, y1): value for (x1, y1), value in bigrams.items() if y1 == start_word}
    for ((x, y), value) in bigram_dict.items():
        x_bigrams[x] = value
    return x_bigrams


def handle_unknowns_for_unigrams(unigram_frequency):
    average_freq_in_unigram = sum(unigram_frequency.values())/len(unigram_frequency)
    unigram_frequency["UNK"] = int(average_freq_in_unigram)
    """keys = [key for key, value in unigram_frequency.items() if value == int(average_freq_in_unigram)]
    if keys is not None:
        unigram_frequency.pop(random.choice(keys), None)
        unigram_frequency["UNK"] = int(average_freq_in_unigram)
    else:
        print("fix this soon")"""

    return unigram_frequency


def get_GoodTuring_counts(unigram_frequency, tokens):
    total_unigrams = len(unigram_frequency)
    total_bigrams = total_unigrams * total_unigrams
    bigrams_prob, bigrams_frequency = compute_bigram_probability(tokens, unigram_frequency)
    bigrams_unseen_count = total_bigrams - sum(bigrams_frequency.values())
    print("bigrams_unseen_count " + str(bigrams_unseen_count))

    bigrams_count_list = [bigrams_unseen_count, 0, 0, 0, 0, 0]
    for key, value in bigrams_frequency.items():
        if value < 6:
            bigrams_count_list[value] += 1

    goodTuring_counts = [0, 0, 0, 0, 0]
    for x in range(0, 5):
        goodTuring_counts[x] = (x+1) * (bigrams_count_list[x+1] * 1.0000) / (bigrams_count_list[x] * 1.0000)
    return goodTuring_counts


def get_Good_Turing_probability(w1, w2, unigram_frequency, tokens, GT_counts):
    adj_count = 0
    bigrams_prob, bigrams_frequency = compute_bigram_probability(tokens, unigram_frequency)
    if (w2, w1) in bigrams_prob.keys():
        adj_count = bigrams_frequency[(w2, w1)]
    if adj_count < 5:
        adj_count = GT_counts[adj_count]
    if w1 in unigram_frequency.keys():
        t = adj_count / (unigram_frequency[w1] * 1.0)
        print(t)
    else:
        t = adj_count / (unigram_frequency["UNK"] * 1.0)
        print(t)
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

def sentiment_classifier():
    pass


def main():
    file_path_training_pos = "D:\PythonProjects\SentimentAnalysis\input\Train\pos.txt"
    file_path_training_neg = "D:\PythonProjects\SentimentAnalysis\input\Train\\neg.txt"
    tokens_pos = default_tokenizer(open(file_path_training_pos,"r").read())
    tokens_neg = default_tokenizer(open(file_path_training_neg,"r").read())

    unigram_frequency_pos = compute_unigram_frequency(tokens_pos)
    unigram_frequency_neg = compute_unigram_frequency(tokens_neg)

    file_path_dev_pos = "D:\PythonProjects\SentimentAnalysis\input\Dev\pos.txt"
    file_path_dev_neg = "D:\PythonProjects\SentimentAnalysis\input\Dev\\neg.txt"
    tokens_dev_pos = default_tokenizer(open(file_path_dev_pos,"r").readline())
    tokens_dev_neg = default_tokenizer(open(file_path_dev_neg,"r").readline())

    #unigram_frequency_pos_dev = compute_unigram_frequency(tokens_dev_pos)
    #unigram_frequency_neg_dev = compute_unigram_frequency(tokens_dev_neg)

    positive_token_count = sum(unigram_frequency_pos.values())
    negative_token_count = sum(unigram_frequency_neg.values())

    print("positive_token_count " + str(positive_token_count))
    print("negative_token_count " + str(negative_token_count))

    unigram_freq_with_unk_pos = handle_unknowns_for_unigrams(unigram_frequency_pos)
    unigram_probabilities_pos = compute_unigram_probability(unigram_freq_with_unk_pos)

    unigram_freq_with_unk_neg = handle_unknowns_for_unigrams(unigram_frequency_neg)
    unigram_probabilities_neg = compute_unigram_probability(unigram_freq_with_unk_neg)

    print("prob of unknowns in positive " + str(unigram_probabilities_pos["UNK"]))
    print("prob of unknowns in negative " + str(unigram_probabilities_neg["UNK"]))

    GT_counts_pos = get_GoodTuring_counts(unigram_freq_with_unk_pos, tokens_pos)
    get_Good_Turing_probability("a", "movie", unigram_freq_with_unk_pos,tokens_pos,GT_counts_pos)

    #test_bigram = ["most","part"]


    #sentiment_classifier(unigram_freq_with_unk_pos,tokens_dev_pos)

    #perplexity_score = calculate_perplexity(test_bigram, unigram_frequency_pos_unk, tokens_pos, GT_counts)
    #print(perplexity_score)


if __name__ == "__main__":
        main()




