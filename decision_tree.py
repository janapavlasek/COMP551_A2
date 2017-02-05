import nltk

def word_feats(words):
    return dict([(word, True) for word in words])

train = [
    (word_feats(nltk.word_tokenize('I love this sandwich.')), 'pos'),
    (word_feats(nltk.word_tokenize('This is an amazing place!')), 'pos'),
    (word_feats(nltk.word_tokenize('I feel very good about these beers.')), 'pos'),
    (word_feats(nltk.word_tokenize('This is my best work.')), 'pos'),
    (word_feats(nltk.word_tokenize("What an awesome view")), 'pos'),
    (word_feats(nltk.word_tokenize('I do not like this restaurant')), 'neg'),
    (word_feats(nltk.word_tokenize('I am tired of this stuff.')), 'neg'),
    (word_feats(nltk.word_tokenize("I can't deal with this")), 'neg'),
    (word_feats(nltk.word_tokenize('He is my sworn enemy!')), 'neg'),
    (word_feats(nltk.word_tokenize('My boss is horrible.')), 'neg')
]
test = [
    (word_feats(nltk.word_tokenize('The beer was good.'))),
    (word_feats(nltk.word_tokenize('I do not enjoy my job'))),
    (word_feats(nltk.word_tokenize("I ain't feeling dandy today."))),
    (word_feats(nltk.word_tokenize("I feel amazing!"))),
    (word_feats(nltk.word_tokenize('Gary is a friend of mine.'))),
    (word_feats(nltk.word_tokenize("I can't believe I'm doing this.")))

]
'''
    (word_feats(nltk.word_tokenize('The beer was good.'), 'pos'),
    (word_feats(nltk.word_tokenize('I do not enjoy my job'), 'neg'),
    (word_feats(nltk.word_tokenize("I ain't feeling dandy today."), 'neg'),
    (word_feats(nltk.word_tokenize("I feel amazing!"), 'pos'),
    (word_feats(nltk.word_tokenize('Gary is a friend of mine.'), 'pos'),
    (word_feats(nltk.word_tokenize("I can't believe I'm doing this."), 'neg')
    '''
#import pudb; pu.db

nltk.usage(nltk.classify.ClassifierI)
classifier = nltk.classify.NaiveBayesClassifier.train(train)
#classifier = nltk.classify.DecisionTreeClassifier.train(train, entropy_cutoff=0,support_cutoff=0)

sorted(classifier.labels())

print classifier.classify_many(test)
