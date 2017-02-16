from sklearn import svm
from sklearn.cross_validation import cross_val_score
from clean_data import CleanData
import numpy as np
import csv

cd = CleanData(tfidf=True, max_features=2000000, n_grams=3)

print 'Getting Training data.'
X, y = cd.bag_of_words(in_file="data/clean_train_input.csv", sparse=True)
print 'Done collecting data.'

# X = [x[1] for x in training_data]
# y = [y[2] for y in training_data]

# del training_data

print 'Training the model.'
lin_clf = svm.LinearSVC()
lin_clf.fit(X, y)
print 'Done training.'

print 'Cross Validation'
c_validation = cross_val_score(lin_clf, X, y, scoring='accuracy')
print c_validation.mean()


print 'Collecting test data.'
test = cd.get_x_in(sparse=True)
# test = [x[1] for x in test_data]
print 'Done collecting data.'

# del test_data

print 'Predicting results.'
out = lin_clf.predict(test)
print 'Done predicitng.'

del test, lin_clf, X, y, c_validation

out = np.array(out).tolist()

for i in range(0, len(out)):
    if out[i] == 0:
        out[i] = 'hockey'
    if out[i] == 1:
        out[i] = 'movies'
    if out[i] == 2:
        out[i] = 'nba'
    if out[i] == 3:
        out[i] = 'news'
    if out[i] == 4:
        out[i] = 'nfl'
    if out[i] == 5:
        out[i] = 'politics'
    if out[i] == 6:
        out[i] = 'soccer'
    if out[i] == 7:
        out[i] = 'worldnews'

print out[0:5]

print 'Writing results into CSV file'
with open('results/svm_final_predictions.csv', 'wb') as result:
    wr = csv.writer(result)
    wr.writerow(["id", "category"])
    for i, item in enumerate(out):
        wr.writerow([i, item])

print 'Completed successfully'
