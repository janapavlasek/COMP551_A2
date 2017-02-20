from sklearn import svm
from clean_data import CleanData

__author__ = "Yacine Sibous"

cd = CleanData()

print "Getting the training data."
training_data = cd.bag_of_words(in_file="data/clean_train_input.csv")
print "Done collecting data."

X = [x[1] for x in training_data]
y = [y[2] for y in training_data]

print X[0:5]
print y[0:5]
clf = svm.SVC()
clf.fit(X, y)
