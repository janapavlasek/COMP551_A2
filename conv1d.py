#!/usr/bin/env python
import time
import numpy
import theano
import pandas
import pickle
import lasagne
import theano.tensor as T
from clean_data import CleanData
from collections import OrderedDict

__author__ = "Robert Fratilla"

max_features = 2000
train_time = 1
num_batches = 10
training_reserve = 0.8
subset = 1000


def create_trainer(network, input_var, y):
    '''
    Responsible for setting up the network with trainable parameters using ADAM optimization
    '''
    print("Creating Trainer...")
    # Output of network
    out = lasagne.layers.get_output(network)
    # Get all parameters from network
    params = lasagne.layers.get_all_params(network, trainable=True)
    # Calculate a loss function which has to be a scalar
    cost = T.nnet.categorical_crossentropy(out, y).mean()
    # Calculate updates using ADAM optimization gradient descent
    updates = lasagne.updates.adam(cost, params, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08)
    # Theano function to compare brain to their masks with ADAM optimization
    train_function = theano.function([input_var, y], updates=updates)  # omitted (, allow_input_downcast=True)
    return train_function


def create_validator(network, input_var, y):
    '''
    Used for calculating the validation error and accuracy of the validation set
    '''
    print ("Creating Validator...")

    # Create prediction
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    # Check how much error in prediction
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, y).mean()
    # Check the accuracy of the prediction
    test_accuracy = T.mean(T.eq(T.argmax(test_prediction, axis=1), T.argmax(y, axis=1)), dtype=theano.config.floatX)
    # Check for error and accuracy percentage
    validate_fn = theano.function([input_var, y], [test_loss, test_accuracy])
    return validate_fn


def create_network(shape=(None, 1, max_features), input_var=T.tensor3()):
    '''
    Responsible for creating the network
    '''
    network = lasagne.layers.InputLayer(shape=(None, 1, max_features), input_var=input_var)

    network = lasagne.layers.Conv1DLayer(network, num_filters=20, filter_size=(5),
                                         pad='same', nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool1DLayer(network, pool_size=(2))
    print "\t", lasagne.layers.get_output_shape(network)

    network = lasagne.layers.Conv1DLayer(network, num_filters=40, filter_size=(5),
                                         pad='same', nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool1DLayer(network, pool_size=(2))
    print "\t", lasagne.layers.get_output_shape(network)

    network = lasagne.layers.Conv1DLayer(network, num_filters=40, filter_size=(5),
                                         pad='same', nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool1DLayer(network, pool_size=(2))
    print "\t", lasagne.layers.get_output_shape(network)

    network = lasagne.layers.DenseLayer(network, num_units=8, nonlinearity=lasagne.nonlinearities.softmax)
    print "\t", lasagne.layers.get_output_shape(network)
    return network


train_input = pandas.read_csv('./data/clean_train_input.csv')
train_output = pandas.read_csv('./data/train_output.csv')
# test_input = pandas.read_csv('./data/test_input.csv')

print ('Getting data...')
cd = CleanData(max_features=max_features, tfidf=True)
train_data = cd.bag_of_words(in_file='./data/clean_train_input.csv')

X = numpy.array([x[1] for x in train_data], dtype='float32')
y = numpy.array([y[2] for y in train_data], dtype='float32')

X = X[:subset]
y = y[:subset]

new_y = numpy.array([numpy.zeros(8)])
for categ in y:
    temp = numpy.array([numpy.zeros(8)])
    numpy.put(temp, int(categ), 1)
    new_y = numpy.concatenate((new_y, temp), axis=0)
new_y = new_y[1:]

train_data = X[:int(X.shape[0] * training_reserve)]
train_truth = new_y[:int(new_y.shape[0] * training_reserve)]

train_test = X[int(X.shape[0] * training_reserve):]
train_test_truth = new_y[int(new_y.shape[0] * training_reserve):]


input_x = T.tensor3('input')
truth_y = T.dmatrix('truth')
network = create_network(shape=(None, 1, max_features), input_var=input_x)
trainer = create_trainer(network, input_x, truth_y)
validator = create_validator(network, input_x, truth_y)


'''
out = lasagne.layers.get_out(network)
fn = theano.function([input_x],out)
train_in = train_data.reshape([train_data.shape[0]] + [1] + [train_data.shape[1]])
trainer(train_in[:1000], train_truth[:1000])
error, accuracy = validator(train_in, train_truth)
'''

record = OrderedDict(epoch=[], error=[], accuracy=[])

print("Training for %s hour(s) with %s batches per epoch" % (train_time, num_batches))
epoch = 0
start_time = time.time()
time_elapsed = time.time() - start_time

while time_elapsed / 3600 < train_time:     # use for time training
    epoch_time = time.time()
    print("--> Epoch: %d | Time left: %.2f hour(s)" % (epoch, train_time - time_elapsed / 3600))
    num_per_division = train_data.shape[0] / num_batches

    for i in xrange(num_batches):

        data = train_data[i * num_batches:i * num_batches + num_per_division]

        train_in = data.reshape([data.shape[0]] + [1] + [data.shape[1]])
        truth_in = train_truth[i * num_batches:i * num_batches + num_per_division]
        if train_in.shape[0] == 0:
            break
        trainer(train_in, truth_in)

        train_in = train_test.reshape([train_test.shape[0]] + [1] + [train_test.shape[1]])
    error, accuracy = validator(train_in, train_test_truth)   # pass modified data through network
    record['error'].append(error)
    record['accuracy'].append(accuracy)
    record['epoch'].append(epoch)
    time_elapsed = time.time() - start_time
    epoch_time = time.time() - epoch_time
    print ("    error: %s and accuracy: %s in %.2fs\n" % (error, accuracy, epoch_time))
    epoch += 1

with open('newstats.pickle', 'w') as output:
    pickle.dump(record, output)
