import theano
import theano.tensor as T
import lasagne
import numpy
import pandas
import nltk
from clean_data import CleanData
import time
from collections import OrderedDict


max_features = 5000
train_time = 1
num_batches = 1000
training_reserve = 0.8


def create_trainer(network,input_var,y):
	print ("Creating Trainer...")
	#output of network
	out = lasagne.layers.get_output(network)
	#get all parameters from network
	params = lasagne.layers.get_all_params(network, trainable=True)
	#calculate a loss function which has to be a scalar
	cost = T.nnet.categorical_crossentropy(out, y).mean()
	#calculate updates using ADAM optimization gradient descent
	updates = lasagne.updates.adam(cost, params, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08)
	#theano function to compare brain to their masks with ADAM optimization
	train_function = theano.function([input_var, y], updates=updates) # omitted (, allow_input_downcast=True)
	return train_function

def create_validator(network, input_var, y):
	print ("Creating Validator...")
	#We will use this for validation
	testPrediction = lasagne.layers.get_output(network, deterministic=True)			#create prediction
	testLoss = lasagne.objectives.categorical_crossentropy(testPrediction,y).mean()   #check how much error in prediction
	testAcc = T.mean(T.eq(T.argmax(testPrediction, axis=1), T.argmax(y, axis=1)),dtype=theano.config.floatX)	#check the accuracy of the prediction

	validateFn = theano.function([input_var, y], [testLoss, testAcc])	 #check for error and accuracy percentage
	return validateFn

def create_network(shape=(None,1,max_features),input_var=T.tensor3()):
	network= lasagne.layers.InputLayer(shape=(None,1,max_features),input_var=input_var)


	network = lasagne.layers.Conv1DLayer(network, num_filters=20, filter_size=(5), pad ='same',nonlinearity=lasagne.nonlinearities.rectify)
	network = lasagne.layers.MaxPool1DLayer(network,pool_size=(2))
	print '	',lasagne.layers.get_output_shape(network)

	network = lasagne.layers.Conv1DLayer(network, num_filters=40, filter_size=(5), pad ='same',nonlinearity=lasagne.nonlinearities.rectify)
	network = lasagne.layers.MaxPool1DLayer(network,pool_size=(2))
	print '	',lasagne.layers.get_output_shape(network)

	#network = lasagne.layers.Conv1DLayer(network, num_filters=80, filter_size=(5), pad ='same',nonlinearity=lasagne.nonlinearities.rectify)
	#network = lasagne.layers.MaxPool1DLayer(network,pool_size=(2))
	#print '	',lasagne.layers.get_output_shape(network)

	#network = lasagne.layers.DenseLayer(network,num_units=400,nonlinearity=lasagne.nonlinearities.rectify)
	#print '	',lasagne.layers.get_output_shape(network)

	network = lasagne.layers.DenseLayer(network,num_units=8,nonlinearity=lasagne.nonlinearities.softmax)
	print '	',lasagne.layers.get_output_shape(network)
	return network

train_input = pandas.read_csv('./data/clean_train_input.csv')
train_output = pandas.read_csv('./data/train_output.csv')
#test_input = pandas.read_csv('./data/test_input.csv')

alphabet = [' ','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
max_length = 300
def get_2drep(in_string):
	temp = numpy.zeros((max_length,len(alphabet)))
	for index,ch in enumerate(in_string[:max_length]):
		temp[index,alphabet.index(ch)] = 1.0
	return [temp]

new_train = numpy.zeros((1,max_length,len(alphabet)))
for sentence in train_input['conversation'][:10]:
	new_train = numpy.concatenate((new_train,get_2drep(sentence)),axis=0)
new_train = new_train[1:]

import pudb; pu.db
print ('Getting data...')
cd = CleanData(max_features=max_features,tfidf=True)
train_data = cd.bag_of_words(in_file='./data/clean_train_input.csv')
import pudb; pu.db 
X = numpy.array([x[1] for x in train_data],dtype='float32')
y = numpy.array([y[2] for y in train_data],dtype='float32')
#X = X[:100000]
#y = y[:100000]
new_y = numpy.array([numpy.zeros(8)])
for categ in y:
	temp = numpy.array([numpy.zeros(8)])
	numpy.put(temp,int(categ),1)
	new_y = numpy.concatenate((new_y,temp),axis=0)
new_y = new_y[1:]

train_data = X[:int(X.shape[0]*training_reserve)]
train_truth = new_y[:int(new_y.shape[0]*training_reserve)]

train_test = X[int(X.shape[0]*training_reserve):] 
train_test_truth =  new_y[int(new_y.shape[0]*training_reserve):] 


#train = numpy.array([nltk.word_tokenize(train_input.iloc[sampleID][1]) for sampleID in train_input['id']])



input_x = T.tensor3('input')
truth_y = T.dmatrix('truth')
network = create_network(shape=(None,1,max_features),input_var=input_x)
trainer = create_trainer(network,input_x,truth_y)
validator = create_validator(network,input_x,truth_y)


'''
out = lasagne.layers.get_out(network)
fn = theano.function([input_x],out)
train_in = train_data.reshape([train_data.shape[0]] + [1] + [train_data.shape[1]])
trainer(train_in[:1000], train_truth[:1000])
error, accuracy = validator(train_in, train_truth)	
'''

record = OrderedDict(epoch=[],error=[],accuracy=[])

print ("Training for %s hour(s) with %s batches per epoch"%(train_time,num_batches))
epoch = 0
start_time = time.time()
time_elapsed = time.time() - start_time
#for epoch in xrange(epochs):            #use for epoch training
while time_elapsed/3600 < train_time :     #use for time training
	epoch_time = time.time()
	print ("--> Epoch: %d | Time left: %.2f hour(s)"%(epoch,train_time-time_elapsed/3600))
	num_per_division = train_data.shape[0]/num_batches

	for i in xrange(num_batches):
		
		data = train_data[i*num_batches:i*num_batches+num_per_division]

		train_in = data.reshape([data.shape[0]] + [1] + [data.shape[1]])
		truth_in = train_truth[i*num_batches:i*num_batches+num_per_division]
		if train_in.shape[0]==0:
			break
		trainer(train_in, truth_in)

	
	#trainIn = data['input'].reshape([1,1] + list(data['input'].shape))
	train_in = train_test.reshape([train_test.shape[0]] + [1] + [train_test.shape[1]])
	error, accuracy = validator(train_in, train_test_truth)			     #pass modified data through network
	record['error'].append(error)
	record['accuracy'].append(accuracy)
	record['epoch'].append(epoch)
	time_elapsed = time.time() - start_time
	epoch_time = time.time() - epoch_time
	print ("	error: %s and accuracy: %s in %.2fs\n"%(error,accuracy,epoch_time))
	epoch+=1

import pickle
#data = {'data':record}
with open('newstats.pickle','w') as output:
	#import pudb; pu.db
	pickle.dump(record,output)

import pudb; pu.db