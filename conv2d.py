import theano
import theano.tensor as T
import lasagne
import numpy
import pandas
import nltk
from clean_data import CleanData
import time
from collections import OrderedDict


train_time = 1           #in hours
num_batches = 10         #how many folds in the data when training
training_reserve = 0.8   #how much to save for training
subset = 100             #max number of sample conversations
max_length = 400         #max string length

def create_trainer(network,input_var,y):
	'''
	Responsible for setting up the network with trainable parameters using ADAM optimization
	'''
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
	'''
	Used for calculating the validation error and accuracy of the validation set
	'''
	print ("Creating Validator...")
	
	test_prediction = lasagne.layers.get_output(network, deterministic=True)			#create prediction
	test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,y).mean()   #check how much error in prediction
	test_accuracy = T.mean(T.eq(T.argmax(test_prediction, axis=1), T.argmax(y, axis=1)),dtype=theano.config.floatX)	#check the accuracy of the prediction
	validate_fn = theano.function([input_var, y], [test_loss, test_accuracy])	 #check for error and accuracy percentage
	return validate_fn

def create_network(shape=(None,None,None,None),input_var=T.tensor3()):
	'''
	Responsible for creating the network
	'''
	network= lasagne.layers.InputLayer(shape=shape,input_var=input_var)

	network = lasagne.layers.Conv2DLayer(network, num_filters=256, filter_size=(7,7), pad ='same',nonlinearity=lasagne.nonlinearities.rectify)
	network = lasagne.layers.MaxPool2DLayer(network,pool_size=(2,2))
	print '	',lasagne.layers.get_output_shape(network)

	network = lasagne.layers.Conv2DLayer(network, num_filters=256, filter_size=(7,7), pad ='same',nonlinearity=lasagne.nonlinearities.rectify)
	network = lasagne.layers.MaxPool2DLayer(network,pool_size=(2,2))
	print '	',lasagne.layers.get_output_shape(network)

	network = lasagne.layers.Conv2DLayer(network, num_filters=256, filter_size=(3,3), pad ='same',nonlinearity=lasagne.nonlinearities.rectify)
	network = lasagne.layers.MaxPool2DLayer(network,pool_size=(2,2))
	print '	',lasagne.layers.get_output_shape(network)

	network = lasagne.layers.DenseLayer(network,num_units=1024,nonlinearity=lasagne.nonlinearities.rectify)
	print '	',lasagne.layers.get_output_shape(network)

	network = lasagne.layers.DenseLayer(network,num_units=1024,nonlinearity=lasagne.nonlinearities.rectify)
	print '	',lasagne.layers.get_output_shape(network)

	network = lasagne.layers.DenseLayer(network,num_units=8,nonlinearity=lasagne.nonlinearities.softmax)
	print '	',lasagne.layers.get_output_shape(network)

	return network

def get_2drep(in_string):
	'''
	This will return the 2d representation of the given string by mapping each character to a spot in 2d space
	'''
	temp = numpy.zeros((max_length,len(alphabet)))
	for index,ch in enumerate(in_string[:max_length]):
		temp[index,alphabet.index(ch)] = 1.0
	return numpy.array([temp],dtype='float32')

print ('Getting data...')
train_input = pandas.read_csv('./data/clean_train_input.csv')
train_output = pandas.read_csv('./data/train_output.csv')
#test_input = pandas.read_csv('./data/test_input.csv')

alphabet = [' ','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
category = {'hockey': 0,'movies': 1,'nba': 2,'news': 3,'nfl': 4,'politics': 5,'soccer': 6,'worldnews': 7}


train_input = train_input['conversation'][:subset]
train_output = train_output['category'][:subset]

#Set up training data
new_train = numpy.array(numpy.zeros((1,max_length,len(alphabet))),dtype='float32')
for sentence in train_input:
	new_train = numpy.concatenate((new_train,get_2drep(sentence)),axis=0)
X = new_train[1:]

#Set up truth values
new_y = numpy.array([numpy.zeros(8)],dtype='float32')
for categ in train_output:
	temp = numpy.array([numpy.zeros(8)],dtype='float32')
	numpy.put(temp,category[categ],1)
	new_y = numpy.concatenate((new_y,temp),axis=0)
new_y = new_y[1:]


#Modify training and validation sets
train_data = X[:int(X.shape[0]*training_reserve)]
train_truth = new_y[:int(new_y.shape[0]*training_reserve)]
train_test = X[int(X.shape[0]*training_reserve):] 
train_test_truth =  new_y[int(new_y.shape[0]*training_reserve):] 




print ('Initializing network and functions...')
input_x = T.tensor4('input')
truth_y = T.dmatrix('truth')
network = create_network(shape=(None,1,train_data.shape[1],train_data.shape[2]),input_var=input_x)
trainer = create_trainer(network,input_x,truth_y)
validator = create_validator(network,input_x,truth_y)

'''
#Use for debugging network one layer at a time

out = lasagne.layers.get_output(network)
fn = theano.function([input_x],out)
train_in = train_data.reshape([train_data.shape[0]] + [1] + [train_data.shape[1],train_data.shape[2]])
trainer(train_in, train_truth)
error, accuracy = validator(train_in, train_truth)	
'''

record = OrderedDict(epoch=[],error=[],accuracy=[])

print ("Training for %s hour(s) with %s batches per epoch"%(train_time,num_batches))
epoch = 0
start_time = time.time()
time_elapsed = time.time() - start_time

while time_elapsed/3600 < train_time :     #use for time training
	epoch_time = time.time()
	print ("--> Epoch: %d | Time left: %.2f hour(s)"%(epoch,train_time-time_elapsed/3600))
	num_per_division = train_data.shape[0]/num_batches

	for i in xrange(num_batches):
		
		data = train_data[i*num_batches:i*num_batches+num_per_division]

		train_in = data.reshape([data.shape[0]] + [1] + [data.shape[1],data.shape[2]])
		truth_in = train_truth[i*num_batches:i*num_batches+num_per_division]
		if train_in.shape[0]==0:
			break
		
		trainer(train_in, truth_in)

	
	train_in = train_test.reshape([train_test.shape[0]] + [1] + [train_test.shape[1],train_test.shape[2]])
	error, accuracy = validator(train_in, train_test_truth)			     #pass modified data through network
	record['error'].append(error)
	record['accuracy'].append(accuracy)
	record['epoch'].append(epoch)
	time_elapsed = time.time() - start_time
	epoch_time = time.time() - epoch_time
	print ("	error: %s and accuracy: %s in %.2fs\n"%(error,accuracy,epoch_time))
	epoch+=1

#Save data to file for visualizing validation error over the amount of epochs
import pickle
with open('newstats.pickle','w') as output:
	pickle.dump(record,output)
