from sklearn import svm
from sklearn.cross_validation import cross_val_score
from clean_data import CleanData
import numpy as np
import csv

# Initialize data for final submission.
cd = CleanData(tfidf=True, max_features=2500000, n_grams=3)

# Geat features and output.
print 'Getting Training data.'
X, y = cd.bag_of_words(in_file="data/clean_train_input.csv", sparse=True)
print 'Done collecting data.'

# Train.
print 'Training the model.'
lin_clf = svm.LinearSVC()
lin_clf.fit(X, y)
print 'Done training.'

# 3-fold cross validation.
print 'Cross Validation'
c_validation = cross_val_score(lin_clf, X, y, scoring='accuracy')
print c_validation.mean()

# Get and predict on the final test data.
print 'Collecting test data.'
test = cd.get_x_in(sparse=True)
print 'Done collecting data.'

print 'Predicting results.'
out = lin_clf.predict(test)
print 'Done predicitng.'

# Clean up to conserve memory.
del test, lin_clf, X, y, c_validation

# Format and save the files.
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

out_file = 'results/svm_final_predictions.csv'

print 'Writing results into CSV file:', out_file
with open(out_file, 'wb') as result:
    wr = csv.writer(result)
    wr.writerow(["id", "category"])
    for i, item in enumerate(out):
        wr.writerow([i, item])

print 'Completed successfully'
