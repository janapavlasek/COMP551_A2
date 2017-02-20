#!/usr/bin/env python
"""
This module combines the results of various algorithms and uses a voting system
to select the final results. In the default implementation, SVM gets 3 votes
and Naive Bayes gets 2 votes.
"""
import csv
from collections import Counter

__author__ = "Jana Pavlasek"


def get_results_from_csv(in_file):
    with open(in_file, 'rb') as f:
        reader = csv.reader(f)
        data = list(reader)

    data.pop(0)

    return [categories_map[x[1]] for x in data]


def append_column(x, col):
    """Append column col onto matrix x."""
    for i, element in enumerate(col):
        if type(x[i]) == int:
            x[i] = [x[i]]
        x[i].append(element)

    return x


if __name__ == '__main__':
    categories_map = {'hockey': 0,
                      'movies': 1,
                      'nba': 2,
                      'news': 3,
                      'nfl': 4,
                      'politics': 5,
                      'soccer': 6,
                      'worldnews': 7}
    categories = ['hockey',
                  'movies',
                  'nba',
                  'news',
                  'nfl',
                  'politics',
                  'soccer',
                  'worldnews']
    predictions = []

    # Collect all the data.
    all_data = get_results_from_csv("results/svm_predictions1.csv")
    all_data = append_column(all_data, get_results_from_csv("results/svm_predictions2.csv"))
    all_data = append_column(all_data, get_results_from_csv("results/svm_predictions3.csv"))
    all_data = append_column(all_data, get_results_from_csv("results/nb_predictions1.csv"))
    all_data = append_column(all_data, get_results_from_csv("results/nb_predictions2.csv"))

    # Calculate the number of votes for each category, and pick the category
    # with the most votes.
    for i, votes in enumerate(all_data):
        data = Counter(votes)
        predictions.append([i, categories[data.most_common(1)[0][0]]])

    with open("results/combined_predictions.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "category"])
        for row in predictions:
            writer.writerow(row)
