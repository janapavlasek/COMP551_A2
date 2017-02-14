#!/usr/bin/env python

import csv
import numpy as np
from clean_data import CleanData


class NaiveBayes(object):

    def __init__(self):
        self.class_probs = []  # Probability of each class.
        self.feature_probs = None  # Relative probability of each feature for a given class.
        self.N = None  # Number of samples.

    def train(self, train_data):
        """Trains the Naive Bayes classifier by learning the
        distributions of each feature."""
        self.N = len(train_data)

        # This will be an array where each row represents values for each
        # class, and the columns within the row are the P(x | c).
        probs = np.zeros((8, len(train_data[0][1])))

        # There are 8 classes. The index of the element corresponds to the
        # class number.
        class_freq = [0] * 8

        # Count the number of times each class appears. At the same time, for
        # that class, count the number of times a word appears.
        for element in train_data:
            class_freq[element[2]] += 1
            for i, word in enumerate(element[1]):
                probs[element[2], i] += word

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
        """Classify data on a previously learned Naive Bayes algorithm."""
        results = []

        # Iterate through each value of the dataset.
        for element in X:
            probs = []
            # Put the features into a useable array form.
            x_i = np.matrix(element[1]).T
            # Iterate through each class and find the probability that the
            # given feature array represents an element of that class.
            for i in range(0, 8):
                # Put the weights into a usable array form.
                w_i = np.asmatrix(self.feature_probs[i, :])
                # Calculate the probability and add it to the array.
                prob = self.class_probs[i] + w_i.dot(x_i)[0, 0]
                probs.append(prob)

            # The result is the class with the highest probability.
            results.append((element[0], np.argmax(probs)))

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
            if Y[i][1] != Y_expect[i][1]:
                count += 1

        return count / float(len(Y))

    def compute_confusion(self, Y, Y_expect):
        """Returns the confusion matrix.

        Args:
            Y: The calculated output.
            Y_expect: The desired output.
        """
        if Y.shape != Y_expect.shape:
            print("Matrices must be the same size.")
            return

        # Initialize confusion matrix of the form:
        #     [m_00, m_01]
        #     [m_10, m_11]
        confusion = np.zeros((2, 2))

        for i in range(0, len(Y)):
            if Y[i] == 0 and Y_expect[i] == 0:    # True -ve
                confusion[0, 0] += 1
            elif Y[i] == 1 and Y_expect[i] == 0:  # False +ve
                confusion[0, 1] += 1
            elif Y[i] == 0 and Y_expect[i] == 1:  # False -ve
                confusion[1, 0] += 1
            elif Y[i] == 1 and Y_expect[i] == 1:  # True +ve
                confusion[1, 1] += 1

        return confusion


if __name__ == '__main__':
    cd = CleanData()

    print "Getting the training data."
    training_data = cd.bag_of_words(in_file="data/clean_train_input.csv")
    print "Done collecting data."

    nb = NaiveBayes()
    print "Training Naive Bayes classifier."
    nb.train(training_data)
    print "Done training."

    print "Classifying training input."
    out = nb.classify(training_data)

    expect = cd.get_y_train()
    print "Training error:", nb.compute_error(out, expect)

    print "Getting the testing data."
    test_data = cd.get_x_in()
    print "Done collecting data."

    out = nb.classify(test_data)

    with open("data/test_input.csv", "r") as f:
        reader = csv.reader(f)
        data = list(reader)

    data.pop(0)

    results = [["id", "conversation", "category"]]
    categories = {0: 'hockey',
                  1: 'movies',
                  2: 'nba',
                  3: 'news',
                  4: 'nfl',
                  5: 'politics',
                  6: 'soccer',
                  7: 'worldnews'}

    for element in out:
        results.append([data[element[0]][0], data[element[0]][1], categories[element[1]]])

    with open("results/naive_bayes_results.csv", "w") as f:
        writer = csv.writer(f)
        for row in results:
            writer.writerow(row)
