import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

# the three different linestyle to distinguish different subjects
LINES = ["-", ":", "--"]

# import text file into string format
def text_to_string(filename):
    with open(filename, encoding = "utf-8") as file:
        return file.read()

# filter strings into pure words
def make_word_dict(string_by_author):
    words_by_author = dict()
    for author in string_by_author:
        tokens = nltk.word_tokenize(string_by_author[author])
        words_by_author[author] = ([token.lower() for token in tokens if token.isalpha()])
    
    return words_by_author

# get shortest length of the text files
def find_shortest_corpus(words_by_author):
    word_count = []
    for author in words_by_author:
        word_count.append(len(words_by_author[author]))
        print(f"Number of words for {author} = {len(words_by_author[author])}\n")
    
    len_shortest_corpus = min(word_count)
    print(f"length shortest corpus = {len_shortest_corpus}\n")
    return len_shortest_corpus

# record the words' length frequency
def word_length_test(words_by_author, len_shortest_corpus):
    by_author_length_freq_dist = dict()
    plt.figure(1)
    plt.ion()

    for i, author in enumerate(words_by_author):
        word_lengths = [len(word) for word in words_by_author[author][: len_shortest_corpus]]
        by_author_length_freq_dist[author] = nltk.FreqDist(word_lengths)
        by_author_length_freq_dist[author].plot(15, label = author, linestyle = LINES[i], title = "Word Length")
    
    plt.legend()
    plt.ioff()
    plt.show()

# record the frequency of all stopwords in dictionary respectively
def stopwords_test(words_by_author, len_shortest_corpus):
    stopwords_by_author_freq_dist = dict()
    plt.figure(2)
    plt.ion()
    stop_words = set(stopwords.words("english"))

    # the length limit is the shortest length of all samples
    for i, author in enumerate(words_by_author):
        stopwords_by_author = [word for word in words_by_author[author][: len_shortest_corpus] if word in stop_words]
        stopwords_by_author_freq_dist[author] = nltk.FreqDist(stopwords_by_author)
        stopwords_by_author_freq_dist[author].plot(50, label = author, linestyle = LINES[i], title = "50 Most Common Stopwords")
    
    plt.legend()
    plt.ioff()
    plt.show()

# record frequency of part of speech of each samples
def parts_of_speech_test(words_by_author, len_shortest_corpus):
    by_author_pos_freq_dist = dict()
    plt.figure(3)
    plt.ion()
    for i, author in enumerate(words_by_author):
        pos_by_author = [pos[1] for pos in nltk.pos_tag(words_by_author[author][: len_shortest_corpus])]
        by_author_pos_freq_dist[author] = nltk.FreqDist(pos_by_author)
        by_author_pos_freq_dist[author].plot(35, label = author, linestyle = LINES[i], title = "Part of Speech")
    
    plt.legend()
    plt.ioff()
    plt.show()

def vocab_test(words_by_author):
    chisquared_by_author = dict()
    for author in words_by_author:
        if (author != "unknown"):
            # calculate the proportion of current sample
            combined_corpus = words_by_author[author] + words_by_author["unknown"]
            author_proportion = len(words_by_author[author]) / len(combined_corpus)

            # get the frequency of occurence of samples' combination
            combined_freq_dist = nltk.FreqDist(combined_corpus)
            # get the first 1000 words occur most frequently
            most_common_words = list(combined_freq_dist.most_common(1000))

            chisquared = 0
            for word, combined_count in most_common_words:
                # the occurence time of certain word in current sample
                observed_count_author = words_by_author[author].count(word)
                # the expected occurence time of the current sample
                expected_count_author = combined_count * author_proportion
                chisquared += (observed_count_author - expected_count_author) ** 2 / expected_count_author
            chisquared_by_author[author] = chisquared
            print("Chi-squared for {} = {:.1f}".format(author, chisquared))

    # get the author who has the lowest chi-square score
    most_likely_author = min(chisquared_by_author, key = chisquared_by_author.get)
    print(f"Most-likely author by vocabulary is {most_likely_author}\n")

def jaccard_test(words_by_author, len_shortest_corpus):
    jaccard_by_author = dict()
    
    # convert the unknown sample into set to delete duplicate words
    unique_words_unknown = set(words_by_author["unknown"][: len_shortest_corpus])

    authors = (author for author in words_by_author if author != "unknown")
    for author in authors:
        # convert the current sample into set to delete duplicate words
        unique_words_author = set(words_by_author[author][: len_shortest_corpus])
        # get the intersection of current sample ans unknown sample
        shared_words = unique_words_author.intersection(unique_words_unknown)

        # get the Jaccard similarity
        jaccard_sim = float(len(shared_words)) / (len(unique_words_author) + len(unique_words_unknown) - len(shared_words))

        jaccard_by_author[author] = jaccard_sim
        print(f"Jaccard Similarity for {author} = {jaccard_sim}")
    
    # get the author who has the highest Jaccard similarity
    most_likely_author = max(jaccard_by_author, key = jaccard_by_author.get)
    print(f"Most-likely author by similarity is {most_likely_author}")

def main():
    # import text file into dictionary
    string_by_author = dict()
    string_by_author["doyle"] = text_to_string("hound.txt")
    string_by_author["wells"] = text_to_string("war.txt")
    string_by_author["unknown"] = text_to_string("lost.txt")

    words_by_author = make_word_dict(string_by_author)
    len_shortest_corpus = find_shortest_corpus(words_by_author)

    word_length_test(words_by_author, len_shortest_corpus)
    stopwords_test(words_by_author, len_shortest_corpus)
    parts_of_speech_test(words_by_author, len_shortest_corpus)
    vocab_test(words_by_author)
    jaccard_test(words_by_author, len_shortest_corpus)


if __name__ == "__main__":
    main()