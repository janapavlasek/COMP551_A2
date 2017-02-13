import nltk
import pandas

train_input = pandas.read_csv('data/train_input.csv')
train_output = pandas.read_csv('data/train_output.csv')
test_input = pandas.read_csv('data/test_input.csv')
#test_output = pandas.read_csv('test_output.csv')


#Create binary features for each word in the sentence
def word_features(sentence):
    return dict([(word, True) for word in sentence])

train = [(word_features(nltk.word_tokenize(train_input['conversation'][i])),
        train_output['category'][i]) for i in xrange(10)]

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
classifier = nltk.classify.DecisionTreeClassifier.train(train, entropy_cutoff=0,support_cutoff=0)

sorted(classifier.labels())

print classifier.classify_many(test)
print (classifier)
