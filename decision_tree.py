import nltk
# import pandas
from clean_data import CleanData

__author__ = "Roboert Fratilla"

# train_input = pandas.read_csv('data/clean_train_input.csv')
# train_output = pandas.read_csv('data/train_output.csv')
# test_input = pandas.read_csv('data/test_input.csv')
# test_output = pandas.read_csv('test_output.csv')

training_reserve = 0.8

cd = CleanData(max_features=2000, tfidf=True)
training_data = cd.bag_of_words(in_file='./data/clean_train_input.csv')


# Create binary features for each word in the sentence
def word_features(sentence):
    return dict([(word, True) for word in sentence])


def features(sentence):
    return dict([(index, word) for index, word in enumerate(sentence)])


def accuracy(classifier, test):
    count_right = count_wrong = 0
    for sample in test:
        print "Prediction:", classifier.classify(sample[0]), "| Truth:", sample[1],
        if classifier.classify(sample[0]) is sample[1]:
            print "right"
            count_right += 1
        else:
            print "wrong"
            count_wrong += 1
    print "Got right:", count_right, "Got wrong:", count_wrong
    print "Accuracy: ", float(count_right) / len(test)
    print ""


# train = [(word_features(nltk.word_tokenize(train_input['conversation'][i])),train_output['category'][i]) for i in xrange(1000)]
train_bow = [(features(training_data[i][1]), training_data[i][2]) for i in xrange(50000)]

train_data = train_bow[:int(len(train_bow) * training_reserve)]
train_test = train_bow[int(len(train_bow) * training_reserve):]

# test = [(word_features(nltk.word_tokenize(test_input['conversation'][i]))) for i in xrange(10)]

nltk.usage(nltk.classify.ClassifierI)
# classifier = nltk.classify.NaiveBayesClassifier.train(train)
classifier = nltk.classify.DecisionTreeClassifier.train(train_data[:30000], entropy_cutoff=0, support_cutoff=0)

print sorted(classifier.labels())

# print classifier.classify_many(test)
print (classifier)

accuracy(classifier, train_test)
