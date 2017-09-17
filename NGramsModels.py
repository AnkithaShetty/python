from numpy.random import choice
from nltk import RegexpTokenizer


def probability_for_unigram(unigram_frequency,total_tokens):
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
        if index < len(tokens)-1:
            w1 = tokens[index + 1]
            w2 = tokens[index]
            bigram = (w1, w2)

            if bigram in bigrams:
                bigrams[bigram] = bigrams[bigram] + 1
            else:
                bigrams[bigram] = 1

    for ((x, y), value) in bigrams.items():
        bigrams[(x, y)] = value / float(unigram_count[y])
    return bigrams


def nested_bigram(bigrams, start_word):
    x_bigrams = {}
    for ((x, start_word), value) in bigrams.items():
        x_bigrams[x] = value
    return x_bigrams


def generate_bigrams_random_sentence(sentence_length, start_word, tokens, unigram_count):
    import random
    bigrams = probability_for_bigrams(tokens, unigram_count)
    if start_word is None:
        start_word = random.choice(list(bigrams.keys()))[0]
    list_words = [start_word]
    if sentence_length > 0:
        next_word = start_word
        for i in range(sentence_length - 1):
            i += 1
            x_bigrams = nested_bigram(bigrams, next_word)
            next_word = "".join(choice(list(x_bigrams.keys()), 1, x_bigrams.values()))
            list_words.append(next_word)
    return " ".join(list_words)


def main():
    filepath = "D:\PythonProjects\SentimentAnalysis\input\pos.txt"
    file_stream = open(filepath, "r").read()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(file_stream.lower())

    unigram_frequency = {}
    for token in tokens:
        if token in unigram_frequency:
            unigram_frequency[token] += 1
        else:
            unigram_frequency[token] = 1
    total_tokens = sum(unigram_frequency.values())
    unigram_count = dict(unigram_frequency)

    unigram_without_seeding = generate_unigram_random_sentence(5, None, unigram_frequency, total_tokens)
    print("Unigram without seeding -> " + unigram_without_seeding)

    unigram_with_seeding = generate_unigram_random_sentence(5, "with", unigram_frequency, total_tokens)
    print("Unigram with seeding -> " + unigram_with_seeding)

    bigram_sentence_with_seeding = generate_bigrams_random_sentence(5, "the", tokens, unigram_count)
    print("Bigram with seeding - > " + bigram_sentence_with_seeding)

    bigram_sentence_without_seeding = generate_bigrams_random_sentence(5, None,tokens, unigram_count)
    print("Bigram without seedinhg -> " + bigram_sentence_without_seeding)


if __name__ == "__main__":
    main()

