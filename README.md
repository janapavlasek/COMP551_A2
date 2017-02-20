#  COMP 551 - Mini-Project 2: Classifying Reddit Conversations

The second project for COMP 551 - Applied Machine Learning.

This repository contains various algorithms used to classify Reddit posts into
eight categories: hockey, movies, nba, news, nfl, politics, soccer or worldnews.
Using the Sci-Kit Learn SVM model, we achieved a score of 96.633% accuracy on
unseen testing data.

## Dependencies

Before running any of the models, install:
* nltk
* sklearn
* pandas
* numpy

For the convolutional models, install:
* Theano
* Lasagne
* scipy
* matplotlib

## Running the models

To run the Naive Bayes model, run:
```
python naive_bayes.py
```

To run the SVM model, run:
```
python scisvm.py
```

To run the Nearest Neighbour model, run:
```
python nearest_neighbour.py
```

To run the Decision Tree model, run:
```
python decision_tree.py
```

To run the voting algorithm, run:
```
python combine_results.py
```

To run the 1-dimensional convolutional network, run:
```
python conv1d.py
```

To run the 2-dimensional convolutional network, run:
```
python conv2d.py
```

## Included directories

The data directory contains the given data as well as the cleaned data,
generated by clean_data.py. You can regenerate the data using this module. The
clean data is stored to avoid the need to rerun the computationally intensive
cleaning functions each time a model is used.

The results directory contains the most relevant results as well as results
needed for the combine_results.py module to run.
