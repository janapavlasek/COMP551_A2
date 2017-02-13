import nltk
import pandas

train_input = pandas.read_csv('data/clean_train_input.csv')
train_output = pandas.read_csv('data/train_output.csv')
test_input = pandas.read_csv('data/test_input.csv')
#test_output = pandas.read_csv('test_output.csv')

training_reserve = 0.7

#Create binary features for each word in the sentence
def word_features(sentence):
    return dict([(word, True) for word in sentence])

def accuracy(classifier, test):
    count_right=count_wrong = 0
    for sample in test:
        print "Prediction:",classifier.classify(sample[0]),"| Truth:",sample[1],
        if classifier.classify(sample[0]) is sample[1]:
            print "right"
            count_right+=1
        else:
            print "wrong"
            count_wrong +=1
    print "Got right:",count_right, "Got wrong:",count_wrong
    print "Accuracy: ",float(count_right)/len(test)
    print ""

train = [(word_features(nltk.word_tokenize(train_input['conversation'][i])),
        train_output['category'][i]) for i in xrange(len(train_input))]    
   
train_data = train[:int(len(train)*training_reserve)]
train_test = train[int(len(train)*training_reserve):]  

test = [(word_features(nltk.word_tokenize(test_input['conversation'][i]))) for i in xrange(10)]
'''
train = [
    (word_features(nltk.word_tokenize('I love this sandwich.')), 'pos'),
    (word_features(nltk.word_tokenize('This is an amazing place!')), 'pos'),
    (word_features(nltk.word_tokenize('I feel very good about these beers.')), 'pos'),
    (word_features(nltk.word_tokenize('This is my best work.')), 'pos'),
    (word_features(nltk.word_tokenize("What an awesome view")), 'pos'),
    (word_features(nltk.word_tokenize('I do not like this restaurant')), 'neg'),
    (word_features(nltk.word_tokenize('I am tired of this stuff.')), 'neg'),
    (word_features(nltk.word_tokenize("I can't deal with this")), 'neg'),
    (word_features(nltk.word_tokenize('He is my sworn enemy!')), 'neg'),
    (word_features(nltk.word_tokenize('My boss is horrible.')), 'neg')
]
test = [
    (word_features(nltk.word_tokenize('The beer was good.'))),
    (word_features(nltk.word_tokenize('I do not enjoy my job'))),
    (word_features(nltk.word_tokenize("I ain't feeling dandy today."))),
    (word_features(nltk.word_tokenize("I feel amazing!"))),
    (word_features(nltk.word_tokenize('Gary is a friend of mine.'))),
    (word_features(nltk.word_tokenize("I can't believe I'm doing this.")))

]
'''
'''
    (word_features(nltk.word_tokenize('The beer was good.'), 'pos'),
    (word_features(nltk.word_tokenize('I do not enjoy my job'), 'neg'),
    (word_features(nltk.word_tokenize("I ain't feeling dandy today."), 'neg'),
    (word_features(nltk.word_tokenize("I feel amazing!"), 'pos'),
    (word_features(nltk.word_tokenize('Gary is a friend of mine.'), 'pos'),
    (word_features(nltk.word_tokenize("I can't believe I'm doing this."), 'neg')
    '''
#import pudb; pu.db

nltk.usage(nltk.classify.ClassifierI)
#classifier = nltk.classify.NaiveBayesClassifier.train(train)
classifier = nltk.classify.DecisionTreeClassifier.train(train_data, entropy_cutoff=0,support_cutoff=0)

print sorted(classifier.labels())

#print classifier.classify_many(test)
print (classifier)

accuracy(classifier,train_test)



def entropy(data, attribute):
    attribute_entropy = 0.0
    value_frequency = {}

    for sample in data:
        #if value_frequency.has_key(sample)
        pass