#!/usr/bin/env python

import csv
import numpy as np
import matplotlib.pyplot as plt
from clean_data import CleanData
from sklearn.cross_validation import train_test_split

__author__ = "Jana Pavlasek"


class NaiveBayes(object):

    def __init__(self):
        self.class_probs = []  # Probability of each class.
        self.feature_probs = None  # Relative probability of each feature for a given class.
        self.N = None  # Number of samples.

    def train(self, X, y):
        """Trains the Naive Bayes classifier by learning the
        distributions of each feature.

        Args:
            X: Feature matrix.
            y: Labels.
        """
        self.N = len(y)

        # This will be an array where each row represents values for each
        # class, and the columns within the row are the P(x | c).
        probs = np.zeros((8, X.shape[1]))

        # There are 8 classes. The index of the element corresponds to the
        # class number.
        class_freq = [0] * 8

        # Count the number of times each class appears. At the same time, for
        # that class, count the number of times a word appears.
        for i in range(0, len(y)):
            class_freq[int(y[i])] += 1
            for j in range(0, X.shape[1]):
                probs[int(y[i]), j] += X[i, j]

        # Transform the frequencies into log probabilities by dividing instance
        # appearances by total number of instances.
        for freq in class_freq:
            self.class_probs.append(np.log(freq / float(self.N)))

        for row in range(0, probs.shape[0]):
            for col in range(0, probs.shape[1]):
                # Calculate each probability using Laplace smoothing.
                probs[row, col] = np.log((probs[row, col] + 1) / float(class_freq[row] + probs.shape[1]))

        self.feature_probs = probs

    def classify(self, X):
        """Classify data on a previously learned Naive Bayes algorithm.

        Args:
            X: Feature matrix.
        """
        results = []

        # Iterate through each value of the dataset.
        for element in X:
            probs = []
            # Put the features into a usable array form.
            x_i = np.matrix(element).T
            # Iterate through each class and find the probability that the
            # given feature array represents an element of that class.
            for i in range(0, 8):
                # Put the weights into a usable array form.
                w_i = np.asmatrix(self.feature_probs[i, :])
                # Calculate the probability and add it to the array.
                prob = self.class_probs[i] + w_i.dot(x_i)[0, 0]
                probs.append(prob)

            # The result is the class with the highest probability.
            results.append(np.argmax(probs))

        return results

    def compute_error(self, Y, Y_expect):
        """Computes the error.

        Args:
            Y: The calculated output.
            Y_expect: The desired output.
        Returns:
            The percentage of incorrect results.
        """
        if len(Y) != len(Y_expect):
            print("Matrices must be the same size.")
            return

        count = 0
        for i in range(0, len(Y)):
            if Y[i] != Y_expect[i]:
                count += 1

        return count / float(len(Y))


def get_numpy_matrices(training_data):
    """Converts training data into numpy matrices."""
    y_train = np.zeros(len(training_data))
    X_train = np.zeros((len(training_data), len(training_data[0][1])))
    ids = []

    idx = 0
    while not len(training_data) == 0:
        ids.append(training_data[0][1])
        y_train[idx] = training_data[0][2]
        X_train[idx] = training_data.pop(0)[1]
        idx += 1

    return ids, X_train, y_train


def plot_max_train_size(num_iter):
    """Tests various training samples sizes and plots the error.

    Args:
        num_iter: Number of times to test for each point.
    """
    points = [10, 50, 100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000,
              8000, 9000, 10000, 15000, 30000]
    errors = []
    train_errors = []

    # Iterate over all points defined.
    for point in points:
        print "Testing for point", point, "training examples."
        error = 0
        train_error = 0

        # Repeat the test the desired number of times.
        for i in range(0, num_iter):
            cd = CleanData(tfidf=True, max_train_size=int(point / 0.6))

            try:
                # Get and train data.
                training_data = cd.bag_of_words(in_file="data/clean_train_input.csv")

                ids, X, y = get_numpy_matrices(training_data)

                del training_data

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

                del X, y, ids

                nb = NaiveBayes()
                nb.train(X_train, y_train)

                # Calculate training and validation errors.
                out = nb.classify(X_test)
                error += nb.compute_error(out, y_test)

                train_out = nb.classify(X_train)
                train_error += nb.compute_error(train_out, y_train)
            except MemoryError:
                print "Memory error. Continuing."
                continue

            del X_train, X_test, y_train, y_test

        print "Training:", train_error / num_iter, "Validation:", error / num_iter
        errors.append(error / num_iter)
        train_errors.append(train_error / num_iter)

    # PLOT.
    plt.figure(1)

    plt.title("Error vs Training Examples")
    plt.xlabel("Number of training examples")
    plt.ylabel("Error")
    # plt.xscale('log')
    plt.plot(points, errors, '-ro')
    plt.plot(points, train_errors, '-bo')
    plt.show()


def plot_feature_size(num_iter):
    """Tests various feature sizes and plots the error.

    Args:
        num_iter: Number of times to test for each point.
    """
    points = [100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000,
              8000, 9000, 10000]
    errors = []
    train_errors = []

    # Iterate over all points defined.
    for point in points:
        print "Testing for point", point, "features."
        error = 0
        train_error = 0

        # Repeat the test the desired number of times.
        for i in range(0, num_iter):
            cd = CleanData(tfidf=True, max_train_size=25000, max_features=point)

            try:
                # Get and train data.
                training_data = cd.bag_of_words(in_file="data/clean_train_input.csv")

                ids, X, y = get_numpy_matrices(training_data)

                del training_data

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

                del X, y, ids

                nb = NaiveBayes()
                nb.train(X_train, y_train)

                # Calculate training and validation errors.
                out = nb.classify(X_test)
                error += nb.compute_error(out, y_test)

                train_out = nb.classify(X_train)
                train_error += nb.compute_error(train_out, y_train)
            except MemoryError:
                print "Memory error. Continuing."
                continue

            del X_train, X_test, y_train, y_test

        errors.append(error / num_iter)
        train_errors.append(train_error / num_iter)

    # PLOT.
    plt.figure(2)

    plt.title("Error vs Features")
    plt.xlabel("Number of features")
    plt.ylabel("Error")
    # plt.xscale('log')
    plt.plot(points, errors, '-ro')
    plt.plot(points, train_errors, '-bo')
    plt.show()


def classify_test_data(cd, nb, results_file):
    """Classify the data for final testing."""
    print "Getting the testing data."
    test_data = cd.get_x_in()

    X_test = np.zeros((len(test_data), len(test_data[0][1])))
    ids = []

    idx = 0
    while not len(test_data) == 0:
        ids.append(test_data[0][0])
        X_test[idx] = test_data.pop(0)[1]
        idx += 1

    print "Done collecting data."

    print "Classifying the testing data."
    out = nb.classify(X_test)
    print "Done classifying."

    print "Creating and saving the results."
    results = [["id", "category"]]
    categories = ['hockey',
                  'movies',
                  'nba',
                  'news',
                  'nfl',
                  'politics',
                  'soccer',
                  'worldnews']

    for id, element in zip(ids, out):
        results.append([id, categories[element]])

    with open(results_file, "w") as f:
        writer = csv.writer(f)
        for row in results:
            writer.writerow(row)


def train_naive_bayes(cd, nb):
    """Train the Naive Bayes classifier and compute training error."""
    print "Getting the training data."
    training_data = cd.bag_of_words(in_file="data/clean_train_input.csv")

    y_train = np.zeros(len(training_data))
    X_train = np.zeros((len(training_data), len(training_data[0][1])))

    idx = 0
    while not len(training_data) == 0:
        y_train[idx] = training_data[0][2]
        X_train[idx] = training_data.pop(0)[1]
        idx += 1

    del training_data

    print "Done collecting data."

    print "Training Naive Bayes classifier."
    nb.train(X_train, y_train)
    print "Done training."

    print "Classifying training input."
    out = nb.classify(X_train)

    print "Training error:", nb.compute_error(out, y_train)

    # Clean up unused arrays.
    del out
    del y_train
    del X_train


if __name__ == '__main__':
    cd = CleanData(tfidf=True, max_train_size=15000, max_features=7000)
    nb = NaiveBayes()

    train_naive_bayes(cd, nb)

    # Classify the test data.
    classify_test_data(cd, nb, "results/nb_predictions2.csv")

    # Tests.
    # plot_max_train_size(3)
    # plot_feature_size(3)

    print "Completed successfully."
