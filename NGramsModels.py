""" Project Part 1: Generates Random sentence for both negative and positive corpura
 Python Version: 3.6"""
from numpy.random import choice
from nltk import RegexpTokenizer

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


def main():
    file_path_training_pos = "D:\PythonProjects\SentimentAnalysis\input\Train\pos.txt"
    file_path_training_neg = "D:\PythonProjects\SentimentAnalysis\input\Train\\neg.txt"
    tokens_pos = default_tokenizer(file_path_training_pos)
    tokens_neg = default_tokenizer(file_path_training_neg)

    unigram_frequency_pos, total_tokens_pos = ngram_frequency_builder(tokens_pos)
    unigram_frequency_neg, total_tokens_neg = ngram_frequency_builder(tokens_neg)

    unigram_without_seeding = generate_unigram_random_sentence(5, None, unigram_frequency_pos, total_tokens_pos)
    print("Unigram without seeding  for postive corpus-> " + unigram_without_seeding)

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


if __name__ == "__main__":
        main()




