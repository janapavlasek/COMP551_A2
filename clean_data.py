#!/usr/bin/env python
import re
import os
import csv
from random import shuffle
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Uncomment if you haven't downloaded NLTK data yet.
# import nltk
# nltk.download('all')

__author__ = "Jana Pavlasek, Yacine Sibous"


class CleanData(object):
    """Cleans the given data into a usable form."""
    def __init__(self, max_features=5000, tfidf=False, max_train_size=None, n_grams=1):
        """Initializes CleanData object.

        Args:
            max_features: Maximum number of features for bag of words.
            tfidf: Uses TF/IDF features if True, count features otherwise.
            max_train_size: If set, cuts the training data set.
            n_grams: The value of n for n_grams.
        """
        self.data = []
        self.max_train_size = max_train_size
        self.categories = {'hockey': 0,
                           'movies': 1,
                           'nba': 2,
                           'news': 3,
                           'nfl': 4,
                           'politics': 5,
                           'soccer': 6,
                           'worldnews': 7}
        # Initialize count vectorizer. Only max_features most frequent words
        # will be analyzed.
        if not tfidf:
            self.vectorizer = CountVectorizer(analyzer="word",
                                              tokenizer=None,
                                              preprocessor=None,
                                              stop_words=None,
                                              max_features=max_features,
                                              ngram_range=(1, n_grams))
        else:
            self.vectorizer = TfidfVectorizer(analyzer="word",
                                              tokenizer=None,
                                              preprocessor=None,
                                              stop_words=None,
                                              max_features=max_features,
                                              ngram_range=(1, n_grams))

    def clean_data(self, in_file, out_file=None):
        """Cleans the data.

        Args:
            input_file: The path to the file to clean.
            output_file: If provided, file to save the output to.
        """
        with open(in_file, 'rb') as f:
            reader = csv.reader(f)
            self.data = list(reader)

            for element in self.data:
                if element[0] == "id":
                    continue

                element[1] = self.clean_post(element[1])

                if int(element[0]) % 1000 == 0:
                    print "Element {} of {}".format(element[0], len(self.data))

        if out_file is not None:
            with open(out_file, "w") as f:
                writer = csv.writer(f)
                for row in self.data:
                    writer.writerow(row)

    def clean_post(self, post):
        """Apply various cleaning rules to a post.

        Args:
            post: A string containing text to clean.

        Returns:
            Cleaned post as string.
        """
        # Remove the tags.
        post = re.sub(r'<.+?>', "", post)

        # Retain only letter characters.
        post = re.sub("[^a-zA-Z]", " ", post)

        # Strip off white space.
        post = post.replace("\n", "")

        # Change all the characters to lower case.
        post = post.lower()

        # Tokenize sentences.
        tokenizer = RegexpTokenizer(r'\w+')
        post = tokenizer.tokenize(post)

        # Remove stop words.
        post = [word for word in post if word not in set(stopwords.words('english'))]

        # Remove single characters.
        post = [word for word in post if len(word) != 1]

        # Stem words.
        snowball_stemmer = SnowballStemmer('english')
        post = [snowball_stemmer.stem(word) for word in post]

        # Stitch back into a string.
        post = " ".join(post)

        return post

    def bag_of_words(self, in_file=None, y_file="data/train_output.csv", sparse=False):
        """Returns the bag of words training set in the form of a list of
        tuples, with tuples of the form:

            (ID, [features], category)

        If the in_file arg is set, the original post data will be loaded from
        a CSV file. If sparse is true, the X and y matrices, with X as a sparse
        matrix, are returned.

        Use this if you do not wish to run clean_data (recommended if you
        haven't made any changes). You probably want the in_file to be
        data/clean_train_input.csv.

        Args:
            in_file: Input file containing the cleaned data. If left blank, the
                     saved data will be used.
            y_file: File in which to save the output. If left blank, output will
                    not be saved to a file.
            sparse: If True, the data returned will be X, y where X is a sparse
                    matrix, instead of the form above.
        """
        # Create the bag of words list.
        if in_file is not None:
            with open(in_file, 'rb') as f:
                reader = csv.reader(f)
                bow_data = list(reader)
        else:
            bow_data = self.data

        # Remove the header row.
        bow_data.pop(0)

        # Shuffle the data.
        shuffle(bow_data)

        # Limit the number of training examples if necessary.
        if self.max_train_size is not None:
            bow_data = bow_data[0:self.max_train_size]

        # Extract just the posts from the array.
        posts = [ele[1] for ele in bow_data]

        # Get bag of words features.
        features = self.vectorizer.fit_transform(posts)

        if not sparse:
            features = features.toarray()

        # Get output.
        with open(y_file, 'rb') as f:
            reader = csv.reader(f)
            y_data = list(reader)

        # Remove the header row.
        y_data.pop(0)

        bow = []
        y = []

        # Put the data in the correct form.
        for i, row in enumerate(features):
            y.append(self.categories[y_data[int(bow_data[i][0])][1]])
            bow.append((int(bow_data[i][0]), row, self.categories[y_data[int(bow_data[i][0])][1]]))

        if sparse:
            return features, y

        return bow

    def get_data(self):
        """Returns a list of lists of the data, where each sub-list is a row of
        data."""
        if not self.data:
            print "Data is empty. You need to run clean_data first."
            return

        return self.data

    def get_data_from_csv(self, in_file="data/clean_train_input.csv"):
        """Returns a list of lists of the data, where each sub-list is a row of
        data, by loading past results from a CSV file (so you don't need to run
        the algorithm)."""
        if not self.check_existance(in_file):
            return

        with open(in_file, 'rb') as f:
            reader = csv.reader(f)
            data = list(reader)

        return data

    def get_x_in(self, in_file="data/clean_test_input.csv", sparse=False):
        """Gets the test data for prediction in the correct form:

                (ID, [features])

        The data is a list of such tuples. If sparse is True, returns a sparse
        matrix of features.

        Args:
            in_file: The CSV file containing the test data.
            sparse: If true, a sparse feature matrix is returned.
        """
        if not self.check_existance(in_file):
            return

        with open(in_file, 'rb') as f:
            reader = csv.reader(f)
            data = list(reader)

        # Remove the header row.
        data.pop(0)

        # Extract just the posts from the array.
        posts = [ele[1] for ele in data]

        # Get bag of words features.
        features = self.vectorizer.transform(posts)

        # Return just the sparse matrix if the flag is set.
        if sparse:
            return features

        features = features.toarray()

        bow = []

        # Put the data in the correct form.
        for i, row in enumerate(features):
            bow.append((i, row))

        return bow

    def check_existance(self, path):
        """Check whether a path exists and print a warning if not."""
        if not os.path.exists(path):
            print "File", path, "does not exist. You need to run clean_data with this path as the out_file."
            return False

        return True


if __name__ == '__main__':
    cd = CleanData()
    # cd.clean_data("data/train_input.csv", out_file="data/clean_train_input.csv")
    # cd.clean_data("data/test_input.csv", out_file="data/clean_test_input.csv")
    print cd.bag_of_words(in_file="data/clean_train_input.csv")[0:5]
