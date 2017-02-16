#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from clean_data import CleanData
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split


def compute_error(Y, Y_expect):
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


def plot_k(cd, num_iter):
    points = [3, 5, 10, 15, 20]
    errors = []
    train_errors = []

    for point in points:
        print "Testing for k =", point
        error = 0
        train_error = 0

        for i in range(0, num_iter):

            nn = KNeighborsClassifier(point, weights='uniform')

            X, y = cd.bag_of_words(in_file="data/clean_train_input.csv", sparse=True)

            # nn.fit(X, y)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
            # print "Error:", c_validation.mean()

            nn.fit(X_train, y_train)

            print "Predicting on validation set."
            out = nn.predict(X_test)
            error += compute_error(out, y_test)

            print "Predicting on training set."
            train_out = nn.predict(X_train)
            train_error += compute_error(train_out, y_train)

            del X_train, X_test, y_train, y_test, out, train_out

        print point, "Training:", train_error / num_iter, "Validation:", error / num_iter
        errors.append(error / num_iter)
        train_errors.append(train_error / num_iter)

    print errors
    print train_errors

    # PLOT.
    plt.figure(2)

    plt.title("Error vs K")
    plt.xlabel("K (number of neighbours)")
    plt.ylabel("Error")
    plt.plot(points, errors, '-ro')
    plt.plot(points, train_errors, '-bo')
    plt.show()


if __name__ == '__main__':
    cd = CleanData(tfidf=True, max_train_size=30000, max_features=5000)  # 15000, 7000

    # n_neighbours = 15
    # nn = KNeighborsClassifier(n_neighbours, weights='uniform')  # weights: 'uniform', 'distance'

    # print "Getting the training data."
    # training_data = cd.bag_of_words(in_file="data/clean_train_input.csv")

    # y_train = np.zeros(len(training_data))
    # X_train = np.zeros((len(training_data), len(training_data[0][1])))

    # idx = 0
    # while not len(training_data) == 0:
    #     y_train[idx] = training_data[0][2]
    #     X_train[idx] = training_data.pop(0)[1]
    #     idx += 1

    # del training_data

    # print "Done getting training data."

    # print "Training..."
    # nn.fit(X_train, y_train)

    # print "Classifying..."
    # results = nn.predict(X_train)

    # print "Training error:", compute_error(results, y_train)

    plot_k(cd, 1)
