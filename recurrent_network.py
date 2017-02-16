import theano
import theano.tensor as T
import lasagne
import numpy
import pandas
import nltk
from clean_data import CleanData


MAX_LENGTH = 400 #max tokens per sample is 362
BATCH_SIZE = None
N_HIDDEN = 100
GRAD_CLIP = 100

def get_mask(lst):
	return [1 for i in range(len(lst))] + [0 for i in range(MAX_LENGTH-len(lst))]

def find_max_length(train):
	max_length = 0
	for sample in train:
		if max_length < len(sample):
			max_length = len(sample)
	print ("Max token length: %d"%max_length)
	return max_length


l_in= lasagne.layers.InputLayer(shape=(BATCH_SIZE,MAX_LENGTH,8))

l_mask = lasagne.layers.InputLayer(shape=(BATCH_SIZE, MAX_LENGTH))

l_forward = lasagne.layers.RecurrentLayer(
        l_in, N_HIDDEN, mask_input=l_mask, grad_clipping=GRAD_CLIP,
        W_in_to_hid=lasagne.init.HeUniform(),
        W_hid_to_hid=lasagne.init.HeUniform(),
        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True)

l_backward = lasagne.layers.RecurrentLayer(
        l_in, N_HIDDEN, mask_input=l_mask, grad_clipping=GRAD_CLIP,
        W_in_to_hid=lasagne.init.HeUniform(),
        W_hid_to_hid=lasagne.init.HeUniform(),
        nonlinearity=lasagne.nonlinearities.tanh,
        only_return_final=True, backwards=True)

l_concat = lasagne.layers.ConcatLayer([l_forward, l_backward])

l_out = lasagne.layers.DenseLayer(l_concat,num_units=8,nonlinearity=lasagne.nonlinearities.tanh)


#train_input = pandas.read_csv('./data/clean_train_input.csv')
#train_output = pandas.read_csv('./data/train_output.csv')
#test_input = pandas.read_csv('./data/test_input.csv')

cd = CleanData(max_features=1000,tfidf=True)
train_data = cd.bag_of_words(in_file='./data/clean_train_input.csv')

X = [x[1] for x in train_data]
y = [y[2] for y in train_data]

print ('Getting data...')
#train = numpy.array([nltk.word_tokenize(train_input.iloc[sampleID][1]) for sampleID in train_input['id']])
MAX_LENGTH = find_max_length(X)

import pudb; pu.db
new_train=numpy.array([])
for sample in train[:10]:
	new_train = numpy.concatenate((new_train,sample + [""] * (MAX_LENGTH - len(sample))),axis=0)


print ('Getting masks...')
train_mask = []
for l in train:
	train_mask.append(get_mask(l))
train_mask = numpy.array(train_mask)

input_x = T.fvector()
truth = T.ivector()

out = lasagne.layers.get_output(l_out)

fn = theano.function([l_in.input_var],out)


