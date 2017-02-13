import theano
import theano.tensor as T
import lasagne
import numpy
import pandas

l_in= lasagne.layers.InputLayer(shape=(None,1,1))

network = lasagne.layers.RecurrentLayer(l_in,num_units=520,nonlinearity=lasagne.nonlinearities.rectify)

l_out = lasagne.layers.DenseLayer(network,num_units=8,nonlinearity=lasagne.nonlinearities.softmax)



train_input = pandas.read_csv('./data/clean_train_input.csv')
train_output = pandas.read_csv('./data/train_output.csv')
test_input = pandas.read_csv('test_input.csv')
input_x = T.tensor3()
truth = T.matrix()

out = lasagne.layers.get_output(l_out)
fn = theano.function([l_in.input_var],out)

import pudb; pu.db
