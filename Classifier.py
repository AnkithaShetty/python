""" Project Part 2: File contains the below functionality:
    --handle unknowns,
    --implement Good-Turing Smoothening for both unigram and bigrams Models
    --sentiment classifier for Bigrams and Unigrams model using Perplexity for classification
Generates two csv files 'classified_bigrams.csv' and 'classified_unigrams.csv'. These files were used for kaggle submission
 Python Version : 3.6
"""
from nltk import RegexpTokenizer

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


def handle_unknowns_for_unigrams(unigram_frequency,tokens):
    maximum_key = max(unigram_frequency, key=unigram_frequency.get)
    tokens = ["UNK" if token == maximum_key else token for token in tokens]
    unigram_frequency["UNK"] = unigram_frequency.pop(maximum_key)
    return unigram_frequency, tokens


def get_Good_Turing_counts_Unigrams(unigram_frequency):
    unigram_frequency_GT_count = dict(unigram_frequency)
    total_count = 0
    for key, value in unigram_frequency.items():
        n_count = sum(1 for x in unigram_frequency.values() if x == (value + 1))
        count_d = sum(1 for x in unigram_frequency.values() if x == value)
        new_count = (value + 1) * (n_count * 1.0000) / (count_d * 1.0000)
        total_count += new_count
        unigram_frequency_GT_count[key] = new_count
    return unigram_frequency_GT_count, total_count


def get_Good_Turing_probability_unigrams(w1, unigrams_frequency,total_count):
    # If unigram has occurred, return good turing probability
    if w1 in unigrams_frequency.items():
        if unigrams_frequency[w1] > 0.0:
            return unigrams_frequency[w1] / total_count

    # Otherwise, return N1/N
    count_n = sum(1 for x in unigrams_frequency.values() if int(x) == 1)
    return count_n/total_count


def compute_perplexity_unigrams(test_token, unigrams_frequency, total_count):
    import math
    len_token = len(test_token)
    for i in range(0, len_token, 1):
        if test_token[i] not in unigrams_frequency:
            test_token[i] = u'UNK'

    test_sum =0
    for x in range(0, len_token - 1):
        prob = get_Good_Turing_probability_unigrams(test_token[x], unigrams_frequency, total_count)
        test_sum += ((math.log(prob)) * (-1))
    return math.exp((test_sum / len_token))


def get_GoodTuring_counts(unigram_frequency, bigrams_frequency):
    total_unigrams = len(unigram_frequency)
    total_bigrams = total_unigrams * total_unigrams
    bigrams_unseen_count = total_bigrams - len(bigrams_frequency)

    bigrams_count_list = [bigrams_unseen_count, 0, 0, 0, 0, 0]
    for key, value in bigrams_frequency.items():
        if value < 6:
            bigrams_count_list[value] += 1

    goodTuring_counts = [0, 0, 0, 0, 0]
    for x in range(0, 5):
        goodTuring_counts[x] = (x+1) * (bigrams_count_list[x+1] * 1.0000) / (bigrams_count_list[x] * 1.0000)
    return goodTuring_counts


def get_Good_Turing_probability(w1, w2, unigram_frequency, bigrams_frequency, GT_counts):
    adj_count = 0
    if (w2, w1) in bigrams_frequency.keys():
        adj_count = bigrams_frequency[(w2, w1)]
    if adj_count < 5:
        adj_count = GT_counts[adj_count]
    if w1 in unigram_frequency.keys():
        return adj_count / (unigram_frequency[w1] * 1.0)


def calculate_perplexity(test_token, unigrams_frequency, bigrams_frequency, tokens, GT_count):
    import math
    total_tokens = len(tokens)
    len_token = len(test_token)
    for i in range(0, len_token, 1):
        if test_token[i] not in unigrams_frequency:
            test_token[i] = u'UNK'

    test_sum = math.log(unigrams_frequency[test_token[0]] * 1.0 / total_tokens) * (-1.0)
    for x in range(0, len_token - 1):
        prob = get_Good_Turing_probability(test_token[x], test_token[x+1], unigrams_frequency, bigrams_frequency, GT_count)
        test_sum += ((math.log(prob)) * (-1))
    return math.exp((test_sum / len_token))


def main():
    file_path_training_pos = "D:\PythonProjects\SentimentAnalysis\input\Train\pos.txt"
    file_path_training_neg = "D:\PythonProjects\SentimentAnalysis\input\Train\\neg.txt"
    file_path_test ="D:\PythonProjects\SentimentAnalysis\input\Test\\test.txt"

    # tokenizes the files
    tokens_pos = default_tokenizer(open(file_path_training_pos, "r").read())
    tokens_neg = default_tokenizer(open(file_path_training_neg, "r").read())
    test_corpus = default_tokenizer(open(file_path_test, "r").read())

    # calculate unigram frequency for both positive and negative tokens
    unigram_frequency_pos = compute_unigram_frequency(tokens_pos)
    unigram_frequency_neg = compute_unigram_frequency(tokens_neg)

    file_path_dev_pos = "D:\PythonProjects\SentimentAnalysis\input\Dev\pos.txt"
    file_path_dev_neg = "D:\PythonProjects\SentimentAnalysis\input\Dev\\neg.txt"
    tokens_pos_dev = default_tokenizer(open(file_path_dev_pos, "r").read())
    tokens_neg_dev = default_tokenizer(open(file_path_dev_neg, "r").read())

    # handling unknowns for both positive and negative
    unigram_freq_with_unk_pos, tokens_pos = handle_unknowns_for_unigrams(unigram_frequency_pos, tokens_pos)
    unigram_freq_with_unk_neg, tokens_neg = handle_unknowns_for_unigrams(unigram_frequency_neg, tokens_neg)

    # calculating bigram frequency and probabilities for both positive and negative
    bigrams_prob_pos, bigrams_frequency_pos = compute_bigram_probability(tokens_pos, unigram_freq_with_unk_pos)
    bigrams_prob_neg, bigrams_frequency_neg = compute_bigram_probability(tokens_neg, unigram_freq_with_unk_neg)

    # Smoothening  for bigrams step 1 : Getting good-turing counts for both negative and positive for bigrams
    GT_counts_pos_bigrams = get_GoodTuring_counts(unigram_freq_with_unk_pos, bigrams_frequency_pos)
    GT_counts_neg_bigrams = get_GoodTuring_counts(unigram_freq_with_unk_neg, bigrams_frequency_neg)

    # Perplexity calculation for bigrams dev(held-out) corpus
    positive_perplexity = calculate_perplexity(tokens_pos_dev, unigram_freq_with_unk_pos, bigrams_frequency_pos, tokens_pos, GT_counts_pos_bigrams)
    print("Perplexity for test data with postive development corpus for bigrams : " + str(positive_perplexity))
    negative_perplexity = calculate_perplexity(tokens_neg_dev, unigram_freq_with_unk_neg, bigrams_frequency_neg,tokens_neg, GT_counts_neg_bigrams)
    print("Perplexity for test data with negative development  corpus for bigrams : " + str(negative_perplexity))


    # Perplexity calculation for bigrams test corpus
    positive_perplexity = calculate_perplexity(test_corpus, unigram_freq_with_unk_pos, bigrams_frequency_pos, tokens_pos, GT_counts_pos_bigrams)
    print("Perplexity for test data with postive corpus for bigrams : " + str(positive_perplexity))
    negative_perplexity = calculate_perplexity(test_corpus, unigram_freq_with_unk_neg, bigrams_frequency_neg, tokens_neg, GT_counts_neg_bigrams)
    print("Perplexity for test data with negative corpus for bigrams : " + str(negative_perplexity))

    goodTuring_count_unigram_pos, total_count_pos = get_Good_Turing_counts_Unigrams(unigram_freq_with_unk_pos)
    goodTuring_count_unigram_neg, total_count_neg = get_Good_Turing_counts_Unigrams(unigram_freq_with_unk_neg)

    """ # Perplexity_calculation for unigrams test corpus
    perplexity_unigrams_pos = compute_perplexity_unigrams(test_corpus, goodTuring_count_unigram_pos, total_count_pos)
    print("Perplexity for test data with postive corpus for unigrams : " + str(perplexity_unigrams_pos))
    perplexity_unigrams_neg = compute_perplexity_unigrams(test_corpus, goodTuring_count_unigram_neg, total_count_neg)
    print("Perplexity for test data with negative corpus for unigrams : " + str(perplexity_unigrams_neg))"""

    # Perplexity_calculation for unigrams dev corpus

    perplexity_unigrams_pos = compute_perplexity_unigrams(tokens_pos_dev, goodTuring_count_unigram_pos, total_count_pos)
    print("Perplexity for dev with postive corpus for unigrams : " + str(perplexity_unigrams_pos))
    perplexity_unigrams_neg = compute_perplexity_unigrams(tokens_neg_dev, goodTuring_count_unigram_neg, total_count_neg)
    print("Perplexity for dev with negative corpus for unigrams : " + str(perplexity_unigrams_neg))
    # sentiment clasiifier for bigrams model

    sentiment_classifier_bigrams = {}
    i = 1
    with open(file_path_test) as f:
        for line in f:
            tokens_test = default_tokenizer(line)
            positive_perplexity = calculate_perplexity(tokens_test, unigram_freq_with_unk_pos,bigrams_frequency_pos, tokens_pos, GT_counts_pos_bigrams)
            negative_perplexity = calculate_perplexity(tokens_test, unigram_freq_with_unk_neg, bigrams_frequency_neg, tokens_neg, GT_counts_neg_bigrams)

            if positive_perplexity < negative_perplexity:
                sentiment_classifier_bigrams[i] = 0
            else:
                sentiment_classifier_bigrams[i] = 1
            i += 1

    import csv
    with open('classified_bigrams.csv', 'w') as csvfile:
        header = ['Id', 'Prediction']
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for key, value in sentiment_classifier_bigrams.items():
            writer.writerow([key, value])

    sentiment_classifier_unigrams = {}
    i = 1
    with open(file_path_test) as f:
        for line in f:
            tokens_test = default_tokenizer(line)
            positive_perplexity = compute_perplexity_unigrams(tokens_test, goodTuring_count_unigram_pos, total_count_pos)
            negative_perplexity = compute_perplexity_unigrams(tokens_test, goodTuring_count_unigram_neg, total_count_neg)

            if positive_perplexity < negative_perplexity:
                sentiment_classifier_unigrams[i] = 0
            else:
                sentiment_classifier_unigrams[i] = 1
            i += 1

    import csv
    with open('classified_unigrams.csv', 'w') as csvfile:
        header = ['Id', 'Prediction']
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for key, value in sentiment_classifier_unigrams.items():
            writer.writerow([key, value])


if __name__ == "__main__":
        main()